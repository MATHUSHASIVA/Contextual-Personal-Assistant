"""
Thinking Agent for proactive analysis and intelligent suggestions using LLM
"""

# Standard library imports
import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

# Third-party imports
from sqlalchemy import desc, or_
from sqlalchemy.orm import Session

# Local imports
from ..models.schemas import InsightType, ThinkingInsightCreate
from ..storage.database import Card, DatabaseManager, Envelope, ThinkingInsight, UserContext
from ..utils.llm_manager import LLMConfig, LLMManager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AnalysisConfig:
    """Configuration for thinking agent analysis"""
    lookback_days: int = 30
    min_pattern_frequency: int = 2
    conflict_detection_days: int = 14
    max_insights_per_run: int = 20


class ThinkingAgent:
    """
    Proactive agent that analyzes Cards and Envelopes to generate insights and suggestions
    """
    
    def __init__(self, database_url: str = None, config: AnalysisConfig = None, api_key: str = None):
        self.db_manager = DatabaseManager(database_url or "sqlite:///./data/assistant.db")
        self.config = config or AnalysisConfig()
        
        # Get API key from environment if not provided
        if not api_key:
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("API key is required for LLM functionality")
        
        # Initialize LLM with API key
        llm_config = LLMConfig(api_key=api_key)
        self.llm = LLMManager(llm_config)
    
    def run_analysis(self) -> Dict[str, Any]:
        """
        Run complete analysis and generate insights with improved error handling
        """
        logger.info("Starting thinking agent analysis...")
        start_time = datetime.now()
        
        session = self.db_manager.get_session()
        try:
            # Initialize default results
            analysis_results = {
                "timestamp": start_time.isoformat(),
                "status": "success",
                "insights_generated": 0,
                "analysis_duration_ms": 0,
                "error": None,
                "categories": {
                    "next_step": 0,
                    "recommendation": 0,
                    "conflict": 0,
                    "optimization": 0
                }
            }
            
            # Clear old insights that have expired
            try:
                self._cleanup_expired_insights(session)
            except Exception as e:
                logger.error(f"Error cleaning up expired insights: {e}")
            
            # Generate all insights with single LLM call
            try:
                all_insights = self._generate_all_insights(session)
                
                # Categorize insights by type
                insights_by_type = {
                    "next_step": [],
                    "recommendation": [],
                    "conflict": [],
                    "optimization": []
                }
                
                for insight in all_insights:
                    if insight.insight_type == InsightType.NEXT_STEP:
                        insights_by_type["next_step"].append(insight)
                    elif insight.insight_type == InsightType.RECOMMENDATION:
                        insights_by_type["recommendation"].append(insight)
                    elif insight.insight_type == InsightType.CONFLICT:
                        insights_by_type["conflict"].append(insight)
                    elif insight.insight_type == InsightType.OPTIMIZATION:
                        insights_by_type["optimization"].append(insight)
                
            except Exception as e:
                logger.error(f"Error generating insights: {e}")
                analysis_results["error"] = f"Insight generation failed: {str(e)}"
                insights_by_type = {
                    "next_step": [],
                    "recommendation": [],
                    "conflict": [],
                    "optimization": []
                }
            
            # Combine all insights
            all_insights = []
            for insight_type, insights in insights_by_type.items():
                if insights:
                    all_insights.extend(insights)
                    analysis_results["categories"][insight_type] = len(insights)
            
            # Limit total insights
            limited_insights = all_insights[:self.config.max_insights_per_run]
            
            # Save insights to database
            for insight_create in limited_insights:
                insight = ThinkingInsight(
                    insight_type=insight_create.insight_type.value,
                    title=insight_create.title,
                    description=insight_create.description,
                    is_actionable=insight_create.is_actionable,
                    suggested_action=insight_create.suggested_action,
                    expires_at=insight_create.expires_at
                )
                
                insight.set_related_card_ids(insight_create.related_card_ids)
                insight.set_related_envelope_ids(insight_create.related_envelope_ids)
                
                session.add(insight)
                
                # Update category counts
                analysis_results["categories"][insight_create.insight_type.value] += 1
            
            session.commit()
            
            analysis_results["insights_generated"] = len(limited_insights)
            analysis_results["analysis_duration_ms"] = int(
                (datetime.now() - start_time).total_seconds() * 1000
            )
            
            if not limited_insights:
                if not analysis_results.get("error"):
                    analysis_results["error"] = "No insights generated"
                analysis_results["status"] = "warning"
            elif analysis_results.get("error"):
                analysis_results["status"] = "partial_success"
            
            logger.info(
                f"Generated {len(limited_insights)} insights in {analysis_results['analysis_duration_ms']}ms "
                f"(Status: {analysis_results['status']})"
            )
            return analysis_results
            
        except Exception as e:
            session.rollback()
            error_msg = f"Error in thinking agent analysis: {str(e)}"
            logger.error(error_msg)
            return {
                "timestamp": start_time.isoformat(),
                "status": "error",
                "insights_generated": 0,
                "analysis_duration_ms": int(
                    (datetime.now() - start_time).total_seconds() * 1000
                ),
                "error": error_msg,
                "categories": {
                    "next_steps": 0,
                    "recommendations": 0,
                    "conflicts": 0,
                    "optimizations": 0
                }
            }
        finally:
            session.close()
    
    def get_active_insights(self, limit: int = 10) -> List[ThinkingInsight]:
        """
        Get currently active insights
        """
        session = self.db_manager.get_session()
        try:
            insights = session.query(ThinkingInsight).filter(
                ThinkingInsight.is_dismissed == False,
                ThinkingInsight.is_acted_upon == False,
                or_(
                    ThinkingInsight.expires_at.is_(None),
                    ThinkingInsight.expires_at > datetime.utcnow()
                )
            ).order_by(
                desc(ThinkingInsight.created_at)
            ).limit(limit).all()
            
            return insights
            
        except Exception as e:
            logger.error(f"Error retrieving active insights: {e}")
            return []
        finally:
            session.close()
    
    def dismiss_insight(self, insight_id: int) -> bool:
        """
        Mark an insight as dismissed
        """
        session = self.db_manager.get_session()
        try:
            insight = session.query(ThinkingInsight).get(insight_id)
            if insight:
                insight.is_dismissed = True
                session.commit()
                return True
            return False
        finally:
            session.close()
    
    def mark_insight_acted_upon(self, insight_id: int) -> bool:
        """
        Mark an insight as acted upon
        """
        session = self.db_manager.get_session()
        try:
            insight = session.query(ThinkingInsight).get(insight_id)
            if insight:
                insight.is_acted_upon = True
                session.commit()
                return True
            return False
        finally:
            session.close()
    
    def _cleanup_expired_insights(self, session: Session):
        """Clean up expired insights"""
        expired_count = session.query(ThinkingInsight).filter(
            ThinkingInsight.expires_at < datetime.utcnow()
        ).delete()
        
        if expired_count > 0:
            logger.info(f"Cleaned up {expired_count} expired insights")
    
    def _generate_all_insights(self, session: Session) -> List[ThinkingInsightCreate]:
        """
        Generate all types of insights using single comprehensive LLM analysis
        """
        insights = []
        
        # Get all data needed for analysis
        cards = session.query(Card).all()
        envelopes = session.query(Envelope).filter(Envelope.is_active == True).all()
        
        # Prepare comprehensive context
        analysis_context = {
            "cards": [{
                "id": card.id,
                "type": card.card_type,
                "description": card.description,
                "status": card.status,
                "envelope_id": card.envelope_id,
                "assignee": card.assignee,
                "due_date": card.due_date.isoformat() if card.due_date else None,
                "keywords": [kw.word for kw in card.keywords],
                "created_at": card.created_at.isoformat(),
                "updated_at": card.updated_at.isoformat()
            } for card in cards],
            
            "envelopes": [{
                "id": env.id,
                "name": env.name,
                "card_count": env.card_count,
                "created_at": env.created_at.isoformat()
            } for env in envelopes]
        }
        
        # Single comprehensive prompt for all insight types
        prompt = f"""You are an intelligent workspace analysis AI. Analyze the workspace data and generate insights across 4 categories.

WORKSPACE DATA:
{json.dumps(analysis_context, indent=2)}

ANALYSIS REQUIREMENTS:
Generate insights for these 4 categories:

1. NEXT STEPS: Analyze task dependencies, priorities, deadlines. Identify tasks needing immediate attention or that should be worked together.

2. RECOMMENDATIONS: Look for grouping opportunities, workload distribution patterns, organization improvements, emerging themes, resource optimizations.

3. CONFLICTS: Detect scheduling conflicts, similar/duplicate tasks, resource allocation conflicts, priority conflicts, timeline overlaps.

4. OPTIMIZATIONS: Find aging tasks needing attention, inactive envelopes, workflow bottlenecks, resource allocation improvements, organizational structure optimizations.

CRITICAL RESPONSE FORMAT:
Return ONLY a valid JSON object with this EXACT structure:
{{
  "next_steps": [
    {{
      "title": "Brief title",
      "description": "Description - NO line breaks",
      "related_card_ids": [1, 2],
      "related_envelope_ids": [1],
      "is_actionable": true,
      "suggested_action": "Action - NO line breaks"
    }}
  ],
  "recommendations": [
    {{
      "title": "Brief title",
      "description": "Description - NO line breaks", 
      "related_card_ids": [1, 2],
      "related_envelope_ids": [1],
      "is_actionable": true,
      "suggested_action": "Action - NO line breaks"
    }}
  ],
  "conflicts": [
    {{
      "title": "Brief title",
      "description": "Description - NO line breaks",
      "related_card_ids": [1, 2], 
      "related_envelope_ids": [1],
      "is_actionable": true,
      "suggested_action": "Action - NO line breaks"
    }}
  ],
  "optimizations": [
    {{
      "title": "Brief title",
      "description": "Description - NO line breaks",
      "related_card_ids": [1, 2],
      "related_envelope_ids": [1], 
      "is_actionable": true,
      "suggested_action": "Action - NO line breaks"
    }}
  ]
}}

STRICT REQUIREMENTS:
- ONLY return the JSON object, no other text
- NO markdown formatting or code blocks
- related_card_ids and related_envelope_ids must be arrays of integers
- is_actionable must be true or false (boolean)
- NO line breaks in string values
- Maximum 5 insights per category
- Must be valid parseable JSON
"""
        
        try:
            llm_response = self.llm.generate_response(prompt)
            if not llm_response:
                raise ValueError("No response from LLM")
            
            # Clean and parse response
            analysis = self._clean_llm_response(llm_response)
            
            # Process next steps
            for insight_data in analysis.get("next_steps", [])[:5]:
                if self._validate_insight_data(insight_data):
                    insight = ThinkingInsightCreate(
                        insight_type=InsightType.NEXT_STEP,
                        title=str(insight_data["title"])[:200],
                        description=str(insight_data["description"]).replace('\n', ' '),
                        related_card_ids=insight_data.get("related_card_ids", []),
                        related_envelope_ids=insight_data.get("related_envelope_ids", []),
                        is_actionable=bool(insight_data.get("is_actionable", True)),
                        suggested_action=str(insight_data.get("suggested_action", "")).replace('\n', ' '),
                        expires_at=datetime.utcnow() + timedelta(days=7)
                    )
                    insights.append(insight)
            
            # Process recommendations
            for insight_data in analysis.get("recommendations", [])[:5]:
                if self._validate_insight_data(insight_data):
                    insight = ThinkingInsightCreate(
                        insight_type=InsightType.RECOMMENDATION,
                        title=str(insight_data["title"])[:200],
                        description=str(insight_data["description"]).replace('\n', ' '),
                        related_card_ids=insight_data.get("related_card_ids", []),
                        related_envelope_ids=insight_data.get("related_envelope_ids", []),
                        is_actionable=bool(insight_data.get("is_actionable", True)),
                        suggested_action=str(insight_data.get("suggested_action", "")).replace('\n', ' '),
                        expires_at=datetime.utcnow() + timedelta(days=14)
                    )
                    insights.append(insight)
            
            # Process conflicts
            for insight_data in analysis.get("conflicts", [])[:3]:
                if self._validate_insight_data(insight_data):
                    insight = ThinkingInsightCreate(
                        insight_type=InsightType.CONFLICT,
                        title=str(insight_data["title"])[:200],
                        description=str(insight_data["description"]).replace('\n', ' '),
                        related_card_ids=insight_data.get("related_card_ids", []),
                        related_envelope_ids=insight_data.get("related_envelope_ids", []),
                        is_actionable=True,
                        suggested_action=str(insight_data.get("suggested_action", "")).replace('\n', ' '),
                        expires_at=datetime.utcnow() + timedelta(days=7)
                    )
                    insights.append(insight)
            
            # Process optimizations
            for insight_data in analysis.get("optimizations", [])[:3]:
                if self._validate_insight_data(insight_data):
                    insight = ThinkingInsightCreate(
                        insight_type=InsightType.OPTIMIZATION,
                        title=str(insight_data["title"])[:200],
                        description=str(insight_data["description"]).replace('\n', ' '),
                        related_card_ids=insight_data.get("related_card_ids", []),
                        related_envelope_ids=insight_data.get("related_envelope_ids", []),
                        is_actionable=bool(insight_data.get("is_actionable", True)),
                        suggested_action=str(insight_data.get("suggested_action", "")).replace('\n', ' '),
                        expires_at=datetime.utcnow() + timedelta(days=14)
                    )
                    insights.append(insight)
        
        except Exception as e:
            logger.error(f"Error generating comprehensive insights with LLM: {e}")
        
        return insights
    
    def _validate_insight_data(self, insight_data: Dict[str, Any]) -> bool:
        """Validate insight data structure and required fields"""
        try:
            if not isinstance(insight_data, dict):
                return False
            
            # Check required fields
            required_fields = ["title", "description"]
            if not all(field in insight_data for field in required_fields):
                return False
            
            # Validate card IDs
            card_ids = insight_data.get("related_card_ids", [])
            if not isinstance(card_ids, list):
                return False
            
            # Validate envelope IDs  
            envelope_ids = insight_data.get("related_envelope_ids", [])
            if not isinstance(envelope_ids, list):
                return False
            
            return True
            
        except Exception:
            return False

    def _clean_llm_response(self, response: str) -> dict:
        """Clean and parse LLM response to ensure valid JSON"""
        try:
            if not response or not response.strip():
                raise ValueError("Empty response from LLM")

            # Extract JSON content from markdown if present
            response = response.strip()
            if '```' in response:
                parts = response.split('```')
                for part in parts:
                    cleaned = part.strip()
                    if cleaned.lower().startswith('json'):
                        cleaned = cleaned[4:].strip()
                    if cleaned.startswith('{'):
                        response = cleaned
                        break
                else:
                    # If no valid JSON found in code blocks, use the original
                    response = response.replace('```', '').strip()
            
            # Remove any remaining JSON language identifier
            if response.lower().startswith('json'):
                response = response[4:].strip()
            
            # Basic JSON structure validation
            if not (response.startswith('{') and response.endswith('}')):
                raise ValueError("Response is not a valid JSON object")
            
            # Fix common JSON formatting issues
            response = re.sub(r'[\n\r\t]', ' ', response)  # Normalize whitespace
            response = re.sub(r'\s+', ' ', response)  # Collapse multiple spaces
            response = re.sub(r',\s*([\]}])', r'\1', response)  # Remove trailing commas
            response = re.sub(r'"\s*:\s*"', '":"', response)  # Fix spacing in key-value pairs
            
            # Handle truncated or incomplete JSON
            if response.count('{') != response.count('}'):
                raise ValueError("Unmatched braces in JSON")
            
            try:
                return json.loads(response)
            except json.JSONDecodeError as e:
                # Try additional cleanup if initial parse fails
                response = re.sub(r'([{,])\s*([^"\s]+)\s*:', r'\1"\2":', response)  # Quote unquoted keys
                response = response.replace("'", '"')  # Replace single quotes with double quotes
                return json.loads(response)
                
        except Exception as e:
            logger.error(f"Error cleaning LLM response: {str(e)}")
            logger.debug(f"Original response: {response}")
            raise ValueError(f"Failed to clean and parse LLM response: {str(e)}")
    
