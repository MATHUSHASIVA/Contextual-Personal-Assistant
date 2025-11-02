# Standard library imports
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

# Third-party imports
from sqlalchemy import func, text
from sqlalchemy.orm import Session

# Local imports
from ..models.schemas import (
    CardResponse,
    CardType,
    EnvelopeResponse,
    EnvelopeType,
    ProcessingResult,
    Status
)
from ..nlp.processor import NLPPipeline
from ..storage.database import Card, DatabaseManager, Envelope, Keyword, UserContext

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.WARNING)


class IngestionAgent:
    """
    Main agent responsible for processing raw notes into structured cards
    """
    
    def __init__(self, database_url: str = None, groq_api_key: str = None):
        from dotenv import load_dotenv
        load_dotenv()
        
        self.db_manager = DatabaseManager(database_url or "sqlite:///./data/assistant.db")
        self.nlp_pipeline = NLPPipeline()
        
        # Get API key with explicit logging
        self.groq_api_key = groq_api_key or os.getenv("GROQ_API_KEY")
        if self.groq_api_key:
            logger.info("✅ Found Groq API key in environment")
            logger.info(f"API Key starts with: {self.groq_api_key[:8]}...")
        else:
            logger.warning("⚠️ No Groq API key found in environment variables")
        
        # Initialize LLM for processing
        self._setup_llm()
    
    def _setup_llm(self):
        """Setup the language model for the agent"""
        from ..utils.llm_manager import LLMManager, LLMConfig
        import requests
        import time
        
        use_local_only = os.getenv("USE_LOCAL_ONLY", "false").lower() == "true"
        
        if use_local_only:
            logger.info("🔧 Using local NLP processing only (LLM disabled)")
            self.llm = None
            return
            
        if not self.groq_api_key:
            logger.warning("No Groq API key provided - LLM processing will be disabled")
            self.llm = None
            return
            
        max_retries = 3
        retry_delay = 1  # seconds
        
        for attempt in range(max_retries):
            try:
                # Check Groq API with proper endpoint and auth
                headers = {'Authorization': f'Bearer {self.groq_api_key}'}
                response = requests.get("https://api.groq.com/openai/v1/models", headers=headers, timeout=5)
                
                # Check Groq API response
                if response.status_code != 200:
                    logger.warning(f"⚠️ Groq API health check failed with status code: {response.status_code}")
                    self.llm = None
                    return
                
                config = LLMConfig(
                    api_key=self.groq_api_key,
                    temperature=0.1,  # Model will be loaded from environment
                    max_tokens=4000  # Groq's recommended max tokens
                )
                self.llm = LLMManager(config)
                logger.info("✨ LLM (Llama 3.3 70B) initialized successfully")
                
                # Test LLM with a simple query
                test_result = self.llm.classify_card_type("Test task: call John")
                if not test_result or test_result not in ["TASK", "REMINDER", "IDEA"]:
                    raise RuntimeError("LLM test classification failed")
                
                logger.info("✅ LLM processing verified and ready")
                return
                
            except requests.exceptions.RequestException as e:
                if "SSLError" in str(e):
                    logger.error("❌ SSL verification failed. Check your internet security settings.")
                    break
                elif "ConnectTimeout" in str(e) or "ConnectionError" in str(e):
                    if attempt < max_retries - 1:
                        logger.warning(f"⚠️ Connection attempt {attempt + 1} failed. Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                    else:
                        logger.error("❌ Failed to establish connection after multiple attempts")
                else:
                    logger.error(f"❌ API request failed: {str(e)}")
                break
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize LLM: {str(e)}")
                break
        
        logger.warning("⚠️ Falling back to local NLP processing")
        self.llm = None
    
    def process_note(self, note_text: str) -> ProcessingResult:
        """
        Main method to process a raw note into a structured card.
        Tries LLM processing first, falls back to NLP if LLM fails.
        """
        start_time = datetime.now()
        logger.info(f"Processing note: {note_text[:50]}...")
        
        try:
            # Always try LLM first if available
            if self.llm and self.llm.client:
                try:
                    logger.info("Attempting LLM processing...")
                    result = self._process_with_llm(note_text)
                    logger.info("Successfully processed with LLM")
                    return result
                except Exception as llm_error:
                    logger.warning(f"LLM processing failed: {str(llm_error)}")
                    logger.info("Falling back to NLP processing...")
            
            # Fall back to NLP processing
            result = self._process_with_nlp(note_text)
            
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            result.processing_time_ms = processing_time
            return result
            
        except Exception as e:
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            logger.error(f"All processing methods failed: {e}", exc_info=True)
            
            return ProcessingResult(
                success=False,
                error_message=f"Processing failed: {str(e)}",
                processing_time_ms=processing_time
            )
    
    def _process_with_llm(self, note_text: str) -> ProcessingResult:
        """Process note using pure LLM processing"""
        session = self.db_manager.get_session()
        try:
            # Get current context for better processing
            existing_envelopes = self._get_existing_envelopes(session)
            user_context = self._get_current_context(session)
            
            # Process note with full context
            result = self.llm.process_note(
                text=note_text,
                current_envelopes=existing_envelopes,
                user_context=user_context
            )
            
            if not result:
                raise RuntimeError("LLM processing failed")
            
            # Create or get envelope
            envelope = self._handle_envelope_from_llm(
                session, 
                result["suggested_envelope"]["name"],
                result["suggested_envelope"]["type"],
                result["card"]["context_keywords"]
            )
            
            # Parse the date string if present
            due_date = None
            if result["card"]["date"] and result["card"]["date"] != "none":
                try:
                    # Try to parse the date string
                    if isinstance(result["card"]["date"], str):
                        # Parse different possible date formats
                        if ' ' in result["card"]["date"]:  # Has time component
                            due_date = datetime.strptime(result["card"]["date"], '%Y-%m-%d %H:%M:%S')
                        else:  # Date only
                            due_date = datetime.strptime(result["card"]["date"], '%Y-%m-%d')
                except ValueError as e:
                    logger.warning(f"Failed to parse date string: {result['card']['date']}, error: {e}")
                    due_date = None
            
            card = Card(
                content=note_text,
                description=result["card"]["description"],
                card_type=result["card"]["type"].lower(),  # Ensure lowercase to match CardType enum
                status=Status.PENDING.value,
                assignee=result["card"]["assignee"] if result["card"]["assignee"] != "none" else None,
                due_date=due_date,
                envelope_id=envelope.id if envelope else None
            )
            
            # Add keywords to card
            self._add_keywords_to_card(session, card, result["card"]["context_keywords"])
            
            session.add(card)
            
            # Update user context with new information
            if result["new_context"]:
                self._update_context_from_llm(session, result["new_context"], envelope)
            
            session.commit()
            
            # Create response
            return ProcessingResult(
                success=True,
                card=self._convert_card_to_response(card, envelope),
                envelope=self._convert_envelope_to_response(envelope) if envelope else None,
                extracted_entities={
                    "description": result["card"]["description"],
                    "date": result["card"]["date"] if result["card"]["date"] != "none" else None,
                    "assignee": result["card"]["assignee"] if result["card"]["assignee"] != "none" else None,
                    "keywords": result["card"]["context_keywords"]
                },
                processing_time_ms=0  # Will be set by caller
            )
            
        except Exception as e:
            session.rollback()
            raise RuntimeError(f"LLM processing failed: {str(e)}")
        finally:
            session.close()
    
    def _process_with_nlp(self, note_text: str) -> ProcessingResult:
        """Process note using local NLP pipeline"""
        
        start_time = datetime.now()
        
        # Get existing envelopes for matching
        session = self.db_manager.get_session()
        try:
            existing_envelopes = self._get_existing_envelopes(session)
            
            # Process with NLP pipeline
            nlp_result = self.nlp_pipeline.process(note_text, existing_envelopes)
            
            if not nlp_result["success"]:
                processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
                return ProcessingResult(
                    success=False,
                    error_message=nlp_result.get("error", "NLP processing failed"),
                    processing_time_ms=processing_time
                )
            
            # Extract results
            entities = nlp_result["entities"]
            classification = nlp_result["classification"]
            envelope_match = nlp_result["envelope_match"]
            
            # Create or get envelope
            envelope = self._handle_envelope(session, envelope_match, entities)
            
            # Create card
            card = self._create_card_from_nlp(session, note_text, classification, entities, envelope)
            
            # Update user context
            self._update_user_context(session, entities, envelope)
            
            session.commit()
            
            # Convert to response format
            card_response = self._convert_card_to_response(card, envelope)
            envelope_response = self._convert_envelope_to_response(envelope) if envelope else None
            
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            return ProcessingResult(
                success=True,
                card=card_response,
                envelope=envelope_response,
                extracted_entities=entities.dict(),
                processing_time_ms=processing_time
            )
            
        finally:
            session.close()
    
    def _get_existing_envelopes(self, session: Session) -> List[Dict[str, Any]]:
        """Get existing envelopes for matching"""
        from sqlalchemy import func
        envelopes = session.query(Envelope).filter(Envelope.is_active == True).all()
        
        result = []
        for envelope in envelopes:
            keywords = [kw.word for kw in envelope.keywords]
            # Count cards in this envelope
            card_count = session.query(func.count(Card.id)).filter(Card.envelope_id == envelope.id).scalar() or 0
            
            result.append({
                "id": envelope.id,
                "name": envelope.name,
                "description": envelope.description,
                "envelope_type": envelope.envelope_type,
                "keywords": keywords,
                "card_count": card_count,
                "created_at": envelope.created_at,
                "updated_at": envelope.updated_at,
                "is_active": envelope.is_active
            })
        
        return result
    
    def _get_current_context(self, session: Session) -> Dict[str, List[str]]:
        """Get current user context for LLM processing"""
        context = {
            "projects": [],
            "people": [],
            "companies": [],
            "themes": []
        }
        
        contexts = session.query(UserContext).filter(UserContext.is_active == True).all()
        for ctx in contexts:
            if ctx.context_type == "project":
                context["projects"].append(ctx.name)
            elif ctx.context_type == "person":
                context["people"].append(ctx.name)
            elif ctx.context_type == "company":
                context["companies"].append(ctx.name)
            elif ctx.context_type == "theme":
                context["themes"].append(ctx.name)
                
        return context
        
    def _handle_envelope_from_llm(self, session: Session, suggested_name: str, envelope_type: str, keywords: List[str]) -> Optional[Envelope]:
        """Handle envelope creation or selection based on LLM suggestion"""
        # Clean and improve the suggested name
        cleaned_name = self._generate_meaningful_envelope_name(suggested_name, keywords)
        
        # Try to find existing envelope by name and keywords
        existing_envelope = self._find_existing_envelope_by_similarity(
            session, cleaned_name, envelope_type, keywords
        )
        
        if existing_envelope:
            # Update existing envelope with new keywords
            self._add_keywords_to_envelope(session, existing_envelope, keywords)
            return existing_envelope
        else:
            # Create new envelope
            return self._create_new_envelope(session, cleaned_name, envelope_type, keywords)
        
    def _generate_meaningful_envelope_name(self, suggested_name: str, keywords: List[str]) -> str:
        """Generate a meaningful envelope name based on context and keywords"""
        if not suggested_name or suggested_name.lower() in ['new', 'general', 'new_1']:
            # Try to generate name from keywords
            relevant_keywords = [k for k in keywords if len(k) > 3 and k.lower() not in ['idea', 'task', 'note', 'new']]
            
            if relevant_keywords:
                # Use most relevant keywords to form clean name (no prefixes)
                if any(k.lower() in ['bill', 'bills', 'payment'] for k in relevant_keywords):
                    return 'Bills'
                elif any(k.lower() in ['grocery', 'groceries', 'shopping'] for k in relevant_keywords):
                    return 'Groceries'
                elif any(k.lower() in ['meeting', 'meetings'] for k in relevant_keywords):
                    return 'Meetings'
                elif any(k.lower() in ['design', 'logo', 'brand'] for k in relevant_keywords):
                    return 'Design'
                elif any(k.lower() in ['budget', 'finance', 'financial'] for k in relevant_keywords):
                    return 'Budget'
                else:
                    # Use the first meaningful keyword as the name
                    return relevant_keywords[0].title()
                
            return 'General'  # Fallback if no meaningful keywords
            
        return suggested_name.title()  # Use suggested name if it's meaningful
    
    def _find_existing_envelope_by_similarity(self, session: Session, name: str, envelope_type: str, keywords: List[str]) -> Optional[Envelope]:
        """Find existing envelope by name and keyword similarity"""
        # First try exact name match
        envelope = session.query(Envelope).filter(
            Envelope.name.ilike(name)
        ).first()
        
        if envelope:
            return envelope
            
        # Try matching by keywords and type
        keyword_matches = (
            session.query(Envelope)
            .filter(
                Envelope.is_active == True,
                Envelope.envelope_type == envelope_type
            )
            .all()
        )
        
        # Score each envelope based on keyword matches
        best_match = None
        best_score = 0
        lower_keywords = set(k.lower() for k in keywords)
        
        for env in keyword_matches:
            env_keywords = set(k.word.lower() for k in env.keywords)
            common_keywords = len(lower_keywords.intersection(env_keywords))
            
            # Calculate similarity score
            if common_keywords > 0:
                score = common_keywords / len(lower_keywords)
                if score > 0.5 and score > best_score:  # At least 50% keyword match
                    best_score = score
                    best_match = env
        
        return best_match
        
    def _determine_envelope_type(self, keywords: List[str]) -> str:
        """Determine envelope type based on keywords"""
        from ..models.schemas import EnvelopeType
        
        # Map keyword categories to valid EnvelopeType enum values
        keyword_types = {
            EnvelopeType.PROJECT.value: [
                'project', 'development', 'implementation', 
                'proposal', 'meeting', 'review', 'sprint'
            ],
            EnvelopeType.COMPANY.value: [
                'company', 'business', 'client', 'strategy',
                'corporate', 'vendor', 'partner'
            ],
            EnvelopeType.PERSON.value: [
                'personal', 'private', 'individual',
                'team member', 'employee'
            ],
            EnvelopeType.THEME.value: [
                'design', 'marketing', 'development', 'research',
                'brand', 'sales', 'finance', 'operations'
            ]
        }
        
        lower_keywords = [k.lower() for k in keywords]
        
        # Check keywords against each category
        for env_type, type_keywords in keyword_types.items():
            if any(k in lower_keywords for k in type_keywords):
                return env_type
                
        return EnvelopeType.GENERAL.value  # Default to "general" type
        
    def _handle_envelope(self, session: Session, envelope_match, entities) -> Optional[Envelope]:
        """Handle envelope creation or selection"""
        if envelope_match.should_create_new:
            base_name = envelope_match.suggested_name or "General"
            envelope_type = envelope_match.suggested_type.value if envelope_match.suggested_type else "general"
            return self._create_new_envelope(session, base_name, envelope_type, entities.keywords)
            
        elif envelope_match.matched_envelope:
            # Use existing envelope
            # Handle both dict and Pydantic model
            if isinstance(envelope_match.matched_envelope, dict):
                envelope_id = envelope_match.matched_envelope["id"]
            else:
                envelope_id = envelope_match.matched_envelope.id
            
            envelope = session.query(Envelope).get(envelope_id)
            
            # Update envelope keywords
            if envelope:
                self._add_keywords_to_envelope(session, envelope, entities.keywords)
            
            return envelope
        
        return None
    
    def _create_new_envelope(self, session: Session, base_name: str, envelope_type: str, keywords: List[str]) -> Envelope:
        """Create a new envelope with unique name"""
        final_name = base_name
        counter = 1

        # Check if an envelope with this name already exists
        while session.query(Envelope).filter(Envelope.name == final_name).first():
            final_name = f"{base_name}_{counter}"
            counter += 1
        
        # Create new envelope with unique name
        envelope = Envelope(
            name=final_name,
            envelope_type=envelope_type,
            description=f"Auto-created for: {base_name}"
        )
        session.add(envelope)
        session.flush()  # Get the ID
        
        # Add keywords
        self._add_keywords_to_envelope(session, envelope, keywords)
        
        logger.info(f"Created new envelope: {envelope.name}")
        return envelope
    
    def _add_keywords_to_envelope(self, session: Session, envelope: Envelope, keywords: List[str]):
        """Add keywords to envelope"""
        for keyword_text in keywords:
            keyword = self._get_or_create_keyword(session, keyword_text)
            
            # Add to envelope if not already present
            if keyword not in envelope.keywords:
                envelope.keywords.append(keyword)
                keyword.frequency += 1
    
    def _add_keywords_to_card(self, session: Session, card: Card, keywords: List[str]):
        """Add keywords to card with frequency tracking"""
        for keyword_text in keywords:
            keyword = self._get_or_create_keyword(session, keyword_text)
            
            if keyword not in card.keywords:
                card.keywords.append(keyword)
                keyword.frequency += 1
    
    def _get_or_create_keyword(self, session: Session, keyword_text: str) -> Keyword:
        """Get existing keyword or create new one"""
        keyword = session.query(Keyword).filter(Keyword.word == keyword_text.lower()).first()
        if not keyword:
            keyword = Keyword(word=keyword_text.lower())
            session.add(keyword)
            session.flush()
        return keyword
    
    def _create_card_from_nlp(
        self, 
        session: Session, 
        note_text: str, 
        classification, 
        entities, 
        envelope: Optional[Envelope]
    ) -> Card:
        """Create a card from NLP results"""
        
        # Extract the first date as due date and ensure it's in YYYY-MM-DD format
        due_date = None
        if entities.dates and len(entities.dates) > 0:
            due_date = entities.dates[0]
            # If time is not specified, set to end of day
            if due_date and due_date.hour == 0 and due_date.minute == 0:
                due_date = due_date.replace(hour=23, minute=59)
        
        # Extract assignee (first person mentioned)
        assignee = entities.people[0] if entities.people else None
        
        # Extract location
        location = entities.locations[0] if entities.locations else None
        
        # Create card
        card = Card(
            content=note_text,
            description=self._create_description(note_text, entities),
            card_type=classification.predicted_type.value,
            status=Status.PENDING.value,
            assignee=assignee,
            due_date=due_date,
            location=location,
            envelope_id=envelope.id if envelope else None
        )
        
        # Set additional entities
        additional_entities = {
            "organizations": entities.organizations,
            "keywords": entities.keywords,
            "all_entities": entities.entities
        }
        card.set_additional_entities(additional_entities)
        
        # First, update the envelope's card count if needed
        if envelope:
            current_count = session.query(func.count(Card.id)).filter(Card.envelope_id == envelope.id).scalar()
            session.execute(
                text("UPDATE envelopes SET card_count = :count WHERE id = :id"),
                {'count': current_count + 1, 'id': envelope.id}
            )
        
        # Now add the card and its keywords
        session.add(card)
        session.flush()  # Get the ID
        
        # Add keywords to card
        self._add_keywords_to_card(session, card, entities.keywords)
        
        return card
    
    def _create_description(self, text: str, entities) -> str:
        """Create a clean description from the raw text"""
        # Remove redundant phrases and clean up
        description = text.strip()
        
        # Remove common prefixes
        prefixes_to_remove = ["remember to", "remind me to", "don't forget to", "note:", "idea:"]
        for prefix in prefixes_to_remove:
            if description.lower().startswith(prefix):
                description = description[len(prefix):].strip()
        
        # Capitalize first letter
        if description:
            description = description[0].upper() + description[1:]
        
        return description
    
    def _update_user_context(self, session: Session, entities, envelope: Optional[Envelope]):
        """Update user context based on processed note"""
        # Convert NLP entities to unified format
        unified_context = {
            "person": entities.people,
            "company": entities.organizations,
            "theme": [kw for kw in entities.keywords if len(kw) > 3]  # Only significant keywords
        }
        self._update_unified_context(session, unified_context, envelope)
    
    def _update_context_from_llm(self, session: Session, new_context: Dict[str, List[str]], envelope: Optional[Envelope]):
        """Update user context based on LLM suggestions"""
        # Convert LLM context to unified format and use shared logic
        type_mapping = {
            "projects": "project",
            "people": "person", 
            "companies": "company",
            "themes": "theme"
        }
        
        unified_context = {}
        for llm_type, items in new_context.items():
            if llm_type in type_mapping and items:
                db_type = type_mapping[llm_type]
                unified_context[db_type] = items
                
        self._update_unified_context(session, unified_context, envelope)
        
    def _update_unified_context(self, session: Session, context_data: Dict[str, List[str]], envelope: Optional[Envelope]):
        """Update context using unified format"""
        for context_type, items in context_data.items():
            for item in items:
                self._update_or_create_context(session, context_type, item, envelope)
                    
    def _update_or_create_context(
        self, 
        session: Session, 
        context_type: str, 
        name: str, 
        envelope: Optional[Envelope]
    ):
        """Update or create a user context entry"""
        
        context = session.query(UserContext).filter(
            UserContext.context_type == context_type,
            UserContext.name == name.lower()
        ).first()
        
        if context:
            # Update existing context
            context.last_referenced = datetime.utcnow()
            context.relevance_score = min(context.relevance_score + 0.1, 1.0)
        else:
            # Create new context
            context = UserContext(
                context_type=context_type,
                name=name.lower(),
                description=f"Auto-extracted {context_type}: {name}",
                relevance_score=0.5,
                related_envelope_id=envelope.id if envelope else None
            )
            session.add(context)
    
    def _convert_card_to_response(self, card: Card, envelope: Optional[Envelope]) -> CardResponse:
        """Convert database card to response model"""
        return CardResponse(
            id=card.id,
            content=card.content,
            description=card.description,
            card_type=CardType(card.card_type),
            status=Status(card.status),
            assignee=card.assignee,
            due_date=card.due_date,
            location=card.location,
            envelope_id=card.envelope_id,
            envelope_name=envelope.name if envelope else None,
            keywords=[kw.word for kw in card.keywords],
            created_at=card.created_at,
            updated_at=card.updated_at,
            additional_entities=card.get_additional_entities()
        )
    
    def _convert_envelope_to_response(self, envelope: Envelope) -> EnvelopeResponse:
        """Convert database envelope to response model"""
        return EnvelopeResponse(
            id=envelope.id,
            name=envelope.name,
            description=envelope.description,
            envelope_type=envelope.envelope_type,
            card_count=envelope.card_count,
            keywords=[kw.word for kw in envelope.keywords],
            created_at=envelope.created_at,
            updated_at=envelope.updated_at,
            is_active=envelope.is_active
        )
