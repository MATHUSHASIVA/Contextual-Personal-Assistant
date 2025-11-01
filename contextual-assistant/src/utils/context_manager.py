"""
Context Management System for maintaining dynamic user context
"""

from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from sqlalchemy import func, desc
import logging
from collections import defaultdict, Counter

from ..storage.database import DatabaseManager, UserContext, Card, Envelope, Keyword
from ..models.schemas import (
    UserContextCreate, UserContextResponse, ContextType
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ContextManager:
    """
    Manages dynamic user context including projects, companies, people, and themes
    """
    
    def __init__(self, database_url: str = None):
        self.db_manager = DatabaseManager(database_url or "sqlite:///./data/assistant.db")
        
        # Context decay parameters
        self.relevance_decay_days = 30  # Days after which relevance starts decaying
        self.min_relevance_threshold = 0.1  # Minimum relevance to keep context active
    
    def get_current_context(self, limit_per_type: int = 10) -> Dict[str, List[UserContextResponse]]:
        """
        Get the current active user context organized by type
        """
        session = self.db_manager.get_session()
        try:
            # Get active contexts ordered by relevance and recency
            contexts = session.query(UserContext).filter(
                UserContext.is_active == True,
                UserContext.relevance_score >= self.min_relevance_threshold
            ).order_by(
                UserContext.context_type,
                desc(UserContext.relevance_score),
                desc(UserContext.last_referenced)
            ).all()
            
            # Group by type and limit
            context_by_type = defaultdict(list)
            type_counts = defaultdict(int)
            
            for context in contexts:
                if type_counts[context.context_type] < limit_per_type:
                    context_response = self._convert_context_to_response(context)
                    context_by_type[context.context_type].append(context_response)
                    type_counts[context.context_type] += 1
            
            return dict(context_by_type)
            
        finally:
            session.close()
    
    def update_context_from_card(self, card_data: Dict[str, Any]) -> List[UserContextResponse]:
        """
        Update user context based on a newly created card with enhanced context refinement
        """
        session = self.db_manager.get_session()
        try:
            updated_contexts = []
            
            # Extract entities and metadata from card
            entities = card_data.get("additional_entities", {})
            assignee = card_data.get("assignee")
            keywords = card_data.get("keywords", [])
            envelope_id = card_data.get("envelope_id")
            card_type = card_data.get("type", "").lower()
            description = card_data.get("description", "")
            due_date = card_data.get("date")
            
            # Update person contexts
            if assignee:
                context = self._update_or_create_person_context(session, assignee, envelope_id)
                updated_contexts.append(self._convert_context_to_response(context))
            
            # Update organization/company contexts from organizations and certain keywords
            organizations = entities.get("organizations", [])
            company_keywords = ["insurance", "company", "bank", "client", "vendor", "supplier", "business"]
            
            # Process organizations
            for org in organizations:
                context = self._update_or_create_company_context(session, org, envelope_id)
                updated_contexts.append(self._convert_context_to_response(context))
            
            # Process potential company-related keywords
            for keyword in keywords:
                # Check if it's a company-related term
                is_company = any(company_term in keyword.lower() for company_term in company_keywords)
                if is_company:
                    context = self._update_or_create_company_context(session, keyword, envelope_id)
                    if context:
                        updated_contexts.append(self._convert_context_to_response(context))
                elif len(keyword) > 3 and self._is_significant_keyword(keyword):
                    context = self._update_or_create_theme_context(session, keyword, envelope_id)
                    if context:
                        updated_contexts.append(self._convert_context_to_response(context))
            
            # Update project context if envelope represents a project
            if envelope_id:
                envelope = session.query(Envelope).get(envelope_id)
                if envelope:
                    if envelope.envelope_type == "project":
                        context = self._update_or_create_project_context(session, envelope.name, envelope_id)
                        updated_contexts.append(self._convert_context_to_response(context))
                    
                    # Cross-reference with other cards in the envelope
                    related_cards = session.query(Card).filter(
                        Card.envelope_id == envelope_id,
                        Card.id != card_data.get("id")  # Exclude current card
                    ).all()
                    
                    # Update context based on card relationships
                    if related_cards:
                        for related_card in related_cards:
                            # Check for shared themes or people
                            if related_card.assignee and related_card.assignee == assignee:
                                self._strengthen_person_context(session, assignee)
                            
                            # Check for related keywords
                            if related_card.keywords:
                                common_keywords = set(keywords).intersection(set(related_card.keywords))
                                for keyword in common_keywords:
                                    if self._is_significant_keyword(keyword):
                                        context = self._update_or_create_theme_context(session, keyword, envelope_id)
                                        if context:
                                            context.relevance_score = min(context.relevance_score + 0.1, 1.0)
                                            updated_contexts.append(self._convert_context_to_response(context))
            
            session.commit()
            return updated_contexts
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error updating context from card: {e}")
            raise
        finally:
            session.close()
    
    def analyze_context_patterns(self) -> Dict[str, Any]:
        """
        Analyze patterns in user context to identify trends and insights
        """
        session = self.db_manager.get_session()
        try:
            analysis = {
                "most_active_people": self._get_most_active_people(session),
                "trending_themes": self._get_trending_themes(session),
                "active_projects": self._get_active_projects(session),
                "context_evolution": self._analyze_context_evolution(session),
                "collaboration_patterns": self._analyze_collaboration_patterns(session)
            }
            
            return analysis
            
        finally:
            session.close()
    
    def refresh_context_relevance(self) -> Dict[str, int]:
        """
        Refresh context relevance scores based on recent activity and time decay
        """
        session = self.db_manager.get_session()
        try:
            updated_counts = {"updated": 0, "deactivated": 0, "deleted": 0}
            
            # Get all active contexts
            contexts = session.query(UserContext).filter(UserContext.is_active == True).all()
            
            for context in contexts:
                # Calculate time-based decay
                days_since_reference = (datetime.utcnow() - context.last_referenced).days
                
                if days_since_reference > self.relevance_decay_days:
                    # Apply decay
                    decay_factor = max(0.1, 1.0 - (days_since_reference - self.relevance_decay_days) * 0.02)
                    new_relevance = context.relevance_score * decay_factor
                    
                    if new_relevance < self.min_relevance_threshold:
                        # Deactivate context
                        context.is_active = False
                        updated_counts["deactivated"] += 1
                    else:
                        context.relevance_score = new_relevance
                        updated_counts["updated"] += 1
                
                # Boost relevance based on recent activity
                recent_activity = self._get_recent_activity_score(session, context)
                if recent_activity > 0:
                    context.relevance_score = min(1.0, context.relevance_score + recent_activity * 0.1)
                    context.last_referenced = datetime.utcnow()
                    updated_counts["updated"] += 1
            
            # Clean up old inactive contexts
            old_threshold = datetime.utcnow() - timedelta(days=90)
            deleted_count = session.query(UserContext).filter(
                UserContext.is_active == False,
                UserContext.updated_at < old_threshold
            ).delete()
            updated_counts["deleted"] = deleted_count
            
            session.commit()
            return updated_counts
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error refreshing context relevance: {e}")
            raise
        finally:
            session.close()
    
    # Category-specific indicators (simplified)
    GROCERY_INDICATORS = {'groceries', 'food', 'milk', 'bread', 'eggs', 'coffee', 'shopping'}
    WORK_INDICATORS = {'meeting', 'project', 'review', 'deadline', 'report', 'presentation'}
    TRAVEL_INDICATORS = {'flight', 'book', 'travel', 'ticket', 'conference'}
    
    def get_context_for_text(self, text: str, limit: int = 5) -> List[UserContextResponse]:
        """
        Get relevant context based on input text for better processing
        """
        session = self.db_manager.get_session()
        try:
            words = set(text.lower().split())
            relevant_contexts = []
            
            contexts = session.query(UserContext).filter(
                UserContext.is_active == True
            ).order_by(desc(UserContext.relevance_score)).all()
            
            for context in contexts:
                relevance_score = 0.0
                
                # Check for direct name match
                if context.name.lower() in text.lower():
                    relevance_score = 1.0
                    context.relevance_score = min(context.relevance_score + 0.1, 1.0)  # Boost for direct mention
                else:
                    # Check for keyword overlap
                    context_keywords = set(context.name.lower().split())
                    overlap = len(words.intersection(context_keywords))
                    if overlap > 0:
                        relevance_score = overlap / len(context_keywords)
                
                if relevance_score > 0:
                    relevant_contexts.append((context, relevance_score))
            
            # Sort by relevance and return top matches
            relevant_contexts.sort(key=lambda x: x[1], reverse=True)
            return [
                self._convert_context_to_response(context[0]) 
                for context in relevant_contexts[:limit]
            ]
            
        finally:
            session.close()
    
    def create_context(self, context_data: UserContextCreate) -> UserContextResponse:
        """
        Manually create a new context entry
        """
        session = self.db_manager.get_session()
        try:
            context = UserContext(
                context_type=context_data.context_type.value,
                name=context_data.name.lower(),
                description=context_data.description,
                relevance_score=context_data.relevance_score,
                related_envelope_id=context_data.related_envelope_id
            )
            
            context.set_metadata(context_data.metadata)
            
            session.add(context)
            session.commit()
            
            return self._convert_context_to_response(context)
            
        except Exception as e:
            session.rollback()
            logger.error(f"Error creating context: {e}")
            raise
        finally:
            session.close()
    
    def _strengthen_person_context(self, session: Session, person_name: str) -> None:
        """Strengthen the context for a person based on repeated interactions"""
        context = session.query(UserContext).filter(
            UserContext.context_type == ContextType.PERSON.value,
            UserContext.name == person_name.lower()
        ).first()
        
        if context:
            # Increase relevance score more significantly for repeated interactions
            context.relevance_score = min(context.relevance_score + 0.15, 1.0)
            context.last_referenced = datetime.utcnow()
    
    def _update_or_create_person_context(
        self, 
        session: Session, 
        person_name: str, 
        envelope_id: Optional[int]
    ) -> UserContext:
        """Update or create person context"""
        # Filter out non-person categories
        non_person_categories = {'groceries', 'shopping', 'tasks', 'items', 'list', 'reminder', 'bill', 'payment'}
        
        # Check if name is a non-person category
        if person_name.lower() in non_person_categories:
            return self._update_or_create_theme_context(session, person_name, envelope_id)
        
        # Additional filters for common non-person terms
        if any(term in person_name.lower() for term in ['list', 'task', 'note', 'reminder']):
            return self._update_or_create_theme_context(session, person_name, envelope_id)

        context = session.query(UserContext).filter(
            UserContext.context_type == ContextType.PERSON.value,
            UserContext.name == person_name.lower()
        ).first()
        
        if context:
            context.last_referenced = datetime.utcnow()
            context.relevance_score = min(context.relevance_score + 0.1, 1.0)
            if envelope_id and not context.related_envelope_id:
                context.related_envelope_id = envelope_id
        else:
            context = UserContext(
                context_type=ContextType.PERSON.value,
                name=person_name.lower(),
                description=f"Person: {person_name}",
                relevance_score=0.6,
                related_envelope_id=envelope_id
            )
            session.add(context)
        
        return context
    
    def _update_or_create_company_context(
        self, 
        session: Session, 
        company_name: str, 
        envelope_id: Optional[int]
    ) -> UserContext:
        """Update or create company context"""
        
        # Handle common company suffixes
        company_suffixes = ["inc", "corp", "ltd", "llc", "company", "co", "corporation"]
        company_keywords = ["insurance", "bank", "financial", "healthcare", "technology"]
        
        name_parts = company_name.lower().split()
        
        # Remove company suffixes for matching
        clean_name = ' '.join(word for word in name_parts 
                            if word not in company_suffixes)
        
        # Check if it's a business term without proper company context
        if clean_name in company_keywords:
            # This is a general business category, treat it as a theme instead
            return self._update_or_create_theme_context(session, company_name, envelope_id)
        
        context = session.query(UserContext).filter(
            UserContext.context_type == ContextType.COMPANY.value,
            UserContext.name == clean_name
        ).first()
        
        if context:
            context.last_referenced = datetime.utcnow()
            context.relevance_score = min(context.relevance_score + 0.15, 1.0)
        else:
            context = UserContext(
                context_type=ContextType.COMPANY.value,
                name=clean_name,
                description=f"Company: {company_name}",
                relevance_score=0.7,
                related_envelope_id=envelope_id
            )
            session.add(context)
        
        return context
    
    def _update_or_create_theme_context(
        self, 
        session: Session, 
        theme: str, 
        envelope_id: Optional[int]
    ) -> UserContext:
        """Update or create theme context with improved categorization"""
        
        # Define categories of themes
        financial_terms = {"budget", "payment", "invoice", "bill", "expense", "cost", "price", "money", "financial"}
        task_terms = {"report", "review", "meeting", "deadline", "schedule", "presentation"}
        status_terms = {"pending", "progress", "completed", "done", "todo", "waiting"}
        
        theme_lower = theme.lower()
        theme_words = set(theme_lower.split())
        
        # Skip if the theme is too generic or status-related
        if theme_lower in status_terms:
            return None
            
        # Check for specific category indicators
        is_grocery = bool(theme_words.intersection(self.GROCERY_INDICATORS))
        is_work = bool(theme_words.intersection(self.WORK_INDICATORS))
        is_travel = bool(theme_words.intersection(self.TRAVEL_INDICATORS))
            
        context = session.query(UserContext).filter(
            UserContext.context_type == ContextType.THEME.value,
            UserContext.name == theme_lower
        ).first()
        
        if context:
            context.last_referenced = datetime.utcnow()
            
            # Adjust relevance based on category
            if is_grocery or is_work or is_travel:
                context.relevance_score = min(context.relevance_score + 0.2, 1.0)
            elif theme_lower in financial_terms or theme_lower in task_terms:
                context.relevance_score = min(context.relevance_score + 0.15, 1.0)
            else:
                context.relevance_score = min(context.relevance_score + 0.05, 1.0)
        else:
            # Set initial relevance based on category
            if is_grocery:
                context_type = ContextType.PERSONAL.value
                initial_relevance = 0.7
                description = f"Personal: {theme}"
            elif is_work:
                context_type = ContextType.WORK.value
                initial_relevance = 0.8
                description = f"Work: {theme}"
            elif is_travel:
                context_type = ContextType.TRAVEL.value
                initial_relevance = 0.7
                description = f"Travel: {theme}"
            else:
                context_type = ContextType.THEME.value
                initial_relevance = 0.4 if theme_lower not in financial_terms else 0.6
                description = f"Theme: {theme}"
            
            context = UserContext(
                context_type=context_type,
                name=theme_lower,
                description=description,
                relevance_score=initial_relevance,
                related_envelope_id=envelope_id
            )
            session.add(context)
        
        return context
    
    def _update_or_create_project_context(
        self, 
        session: Session, 
        project_name: str, 
        envelope_id: Optional[int]
    ) -> UserContext:
        """Update or create project context"""
        
        context = session.query(UserContext).filter(
            UserContext.context_type == ContextType.PROJECT.value,
            UserContext.name == project_name.lower()
        ).first()
        
        if context:
            context.last_referenced = datetime.utcnow()
            context.relevance_score = min(context.relevance_score + 0.2, 1.0)
        else:
            context = UserContext(
                context_type=ContextType.PROJECT.value,
                name=project_name.lower(),
                description=f"Project: {project_name}",
                relevance_score=0.8,
                related_envelope_id=envelope_id
            )
            session.add(context)
        
        return context
    
    def _is_significant_keyword(self, keyword: str) -> bool:
        """Determine if a keyword is significant enough to track"""
        # Common words to filter out
        common_words = {
            "the", "and", "or", "but", "in", "on", "at", "to", "for", "of", "with", "by",
            "this", "that", "these", "those", "what", "when", "where", "why", "how",
            "can", "will", "should", "would", "could", "may", "might", "must",
            # Time-related terms
            "day", "week", "month", "year", "today", "tomorrow", "tonight",
            # General action words
            "get", "make", "do", "go", "come", "take", "give", "find",
            # Common task-related words
            "task", "todo", "done", "pending", "list", "item",
            # Quantity words
            "many", "much", "few", "several", "some", "any",
            # Status words (not meaningful context)
            "note", "reminder", "idea", "thing", "stuff", "need", "want"
        }
        
        return (
            len(keyword) > 3 and
            keyword.lower() not in common_words and
            keyword.isalpha()
        )
    
    def _get_recent_activity_score(self, session: Session, context: UserContext) -> float:
        """Get activity score based on recent cards related to this context"""
        recent_threshold = datetime.utcnow() - timedelta(days=7)
        
        # Count recent cards that mention this context
        activity_count = 0
        
        if context.context_type == ContextType.PERSON.value:
            activity_count = session.query(Card).filter(
                Card.assignee == context.name,
                Card.created_at >= recent_threshold
            ).count()
        
        elif context.related_envelope_id:
            activity_count = session.query(Card).filter(
                Card.envelope_id == context.related_envelope_id,
                Card.created_at >= recent_threshold
            ).count()
        
        # Normalize activity score
        return min(activity_count * 0.1, 0.5)
    
    def _get_most_active_people(self, session: Session) -> List[Dict[str, Any]]:
        """Get most active people based on recent interactions"""
        recent_threshold = datetime.utcnow() - timedelta(days=30)
        non_person_categories = {'groceries', 'shopping', 'tasks', 'items', 'list', 'reminder'}
        
        # Get only genuine person contexts (excluding non-person categories)
        people_contexts = session.query(UserContext).filter(
            UserContext.context_type == ContextType.PERSON.value,
            UserContext.is_active == True,
            UserContext.last_referenced >= recent_threshold,
            ~UserContext.name.in_([name for name in non_person_categories])  # Exclude non-person categories
        ).order_by(desc(UserContext.relevance_score)).limit(10).all()
        
        result = []
        for context in people_contexts:
            # Count related cards including case-insensitive matches
            card_count = session.query(Card).filter(
                func.lower(Card.assignee) == context.name.lower(),
                Card.created_at >= recent_threshold
            ).count()
            
            if card_count > 0 or context.relevance_score > 0.5:  # Only include if there are cards or high relevance
                result.append({
                    "name": context.name.title(),
                    "relevance_score": context.relevance_score,
                    "card_count": card_count,
                    "last_referenced": context.last_referenced
                })
        
        return result
    
    def _get_trending_themes(self, session: Session) -> List[Dict[str, Any]]:
        """Get trending themes based on keyword frequency and actual occurrences"""
        recent_threshold = datetime.utcnow() - timedelta(days=14)
        
        # Get keywords and their actual occurrences in recent cards
        subquery = session.query(
            Keyword.word,
            func.count(Card.id).label('actual_count')
        ).join(
            Keyword.cards
        ).filter(
            Card.created_at >= recent_threshold
        ).group_by(
            Keyword.word
        ).subquery()
        
        # Get keywords with their base frequency and actual counts
        keywords = session.query(
            Keyword,
            subquery.c.actual_count
        ).join(
            subquery,
            Keyword.word == subquery.c.word
        ).order_by(desc(subquery.c.actual_count)).limit(20).all()
        
        result = []
        for keyword, actual_count in keywords:
            # Use actual count instead of stored frequency
            result.append({
                "theme": keyword.word.title(),
                "frequency": actual_count or 0,  # Fallback to 0 if None
                "importance_score": keyword.importance_score
            })
        
        return result
    
    def _get_active_projects(self, session: Session) -> List[Dict[str, Any]]:
        """Get active projects based on recent activity"""
        recent_threshold = datetime.utcnow() - timedelta(days=30)
        
        project_contexts = session.query(UserContext).filter(
            UserContext.context_type == ContextType.PROJECT.value,
            UserContext.is_active == True
        ).order_by(desc(UserContext.relevance_score)).limit(10).all()
        
        result = []
        for context in project_contexts:
            # Get related envelope and card count
            card_count = 0
            envelope_name = context.name
            
            if context.related_envelope_id:
                envelope = session.query(Envelope).get(context.related_envelope_id)
                if envelope:
                    envelope_name = envelope.name
                    card_count = session.query(Card).filter(
                        Card.envelope_id == envelope.id,
                        Card.created_at >= recent_threshold
                    ).count()
            
            result.append({
                "name": envelope_name.title(),
                "relevance_score": context.relevance_score,
                "recent_activity": card_count,
                "last_referenced": context.last_referenced
            })
        
        return result
    
    def _analyze_context_evolution(self, session: Session) -> Dict[str, Any]:
        """Analyze how context has evolved over time with category tracking"""
        try:
            now = datetime.utcnow()
            week_ago = now - timedelta(days=7)
            month_ago = now - timedelta(days=30)
            
            # Get active contexts with their types for different time periods
            recent_contexts = session.query(
                UserContext.context_type,
                func.count(UserContext.id).label('count')
            ).filter(
                UserContext.created_at >= week_ago,
                UserContext.is_active == True
            ).group_by(UserContext.context_type).all()
            
            monthly_contexts = session.query(
                UserContext.context_type,
                func.count(UserContext.id).label('count')
            ).filter(
                UserContext.created_at >= month_ago
            ).group_by(UserContext.context_type).all()
            
            total_by_type = session.query(
                UserContext.context_type,
                func.count(UserContext.id).label('count')
            ).group_by(UserContext.context_type).all()
            
            # Convert to dictionary for easier access
            recent_counts = {str(t): c for t, c in recent_contexts}
            monthly_counts = {str(t): c for t, c in monthly_contexts}
            total_counts = {str(t): c for t, c in total_by_type}
            
            # Calculate growth rates and trends by type
            growth_rates = {}
            trend_data = {}
            
            for context_type in total_counts.keys():
                current_count = monthly_counts.get(context_type, 0)
                old_count = total_counts[context_type] - current_count
                
                # Calculate growth rate safely
                if old_count > 0:
                    growth_rate = (current_count / old_count) * 100
                elif current_count > 0:
                    growth_rate = 100  # New type with some contexts
                else:
                    growth_rate = 0    # No growth
                
                growth_rates[context_type] = round(growth_rate, 2)
                
                # Calculate trend score
                recent_count = recent_counts.get(context_type, 0)
                trend_score = (
                    growth_rates[context_type] * 0.6 +  # Weight growth rate at 60%
                    recent_count * 0.4                  # Weight recent activity at 40%
                )
                
                trend_data[context_type] = {
                    'type': context_type,
                    'growth_rate': growth_rates[context_type],
                    'recent_count': recent_count,
                    'monthly_count': current_count,
                    'total_count': total_counts[context_type],
                    'trend_score': round(trend_score, 2)
                }
            
            # Sort trending categories by trend score
            trending_categories = sorted(
                trend_data.values(),
                key=lambda x: x['trend_score'],
                reverse=True
            )
            
            return {
                "total_contexts": sum(total_counts.values()),
                "contexts_by_type": {
                    "recent": dict(recent_counts),
                    "monthly": dict(monthly_counts),
                    "total": dict(total_counts)
                },
                "growth_rates": growth_rates,
                "trending_categories": trending_categories,
                "trend_data": trend_data
            }
            
        except Exception as e:
            logger.error(f"Error analyzing context evolution: {e}")
            return {
                "total_contexts": 0,
                "contexts_by_type": {
                    "recent": {},
                    "monthly": {},
                    "total": {}
                },
                "growth_rates": {},
                "trending_categories": [],
                "error": str(e)
            }
    
    def _analyze_collaboration_patterns(self, session: Session) -> Dict[str, Any]:
        """Analyze collaboration patterns between people"""
        recent_threshold = datetime.utcnow() - timedelta(days=30)
        
        # Get cards with assignees from recent period
        cards_with_people = session.query(Card).filter(
            Card.assignee.isnot(None),
            Card.created_at >= recent_threshold
        ).all()
        
        # Count collaborations (simplified - people mentioned together)
        collaborations = Counter()
        for card in cards_with_people:
            if card.assignee:
                collaborations[card.assignee] += 1
        
        most_collaborative = collaborations.most_common(5)
        
        return {
            "most_mentioned_people": [{"name": name.title(), "mentions": count} for name, count in most_collaborative],
            "total_people_mentioned": len(collaborations),
            "average_mentions_per_person": sum(collaborations.values()) / max(len(collaborations), 1)
        }
    
    def _convert_context_to_response(self, context: UserContext) -> UserContextResponse:
        """Convert database context to response model"""
        return UserContextResponse(
            id=context.id,
            context_type=ContextType(context.context_type),
            name=context.name,
            description=context.description,
            relevance_score=context.relevance_score,
            metadata=context.get_context_data(),
            related_envelope_id=context.related_envelope_id,
            created_at=context.created_at,
            updated_at=context.updated_at,
            last_referenced=context.last_referenced,
            is_active=context.is_active
        )