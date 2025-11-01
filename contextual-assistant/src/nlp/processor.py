"""
NLP processing components for entity extraction and text classification
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import dateparser
import numpy as np
from dataclasses import dataclass
import spacy

# ML imports - with try-except for optional dependencies  
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    SentenceTransformer = None
    cosine_similarity = None

# Optional fuzzy matching imports
try:
    from fuzzywuzzy import fuzz, process
    FUZZY_AVAILABLE = True
except (ImportError, ValueError):
    FUZZY_AVAILABLE = False
    fuzz = None
    process = None

from ..models.schemas import (
    EntityExtractionResult, 
    ClassificationResult, 
    CardType,
    EnvelopeMatchResult,
    EnvelopeType
)

@dataclass
class NLPConfig:
    """Configuration for NLP components"""
    spacy_model: str = "en_core_web_sm"
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    envelope_similarity_threshold: float = 0.7
    max_keywords: int = 5

class EntityExtractor:
    """
    Handles Named Entity Recognition and entity extraction from text
    """
    
    def __init__(self, config: NLPConfig = None):
        self.config = config or NLPConfig()
        self.nlp = None
        self._load_models()
    
    def _load_models(self):
        """Load spaCy model"""
        try:
            self.nlp = spacy.load(self.config.spacy_model)
        except OSError:
            raise
    
    def extract_entities(self, text: str) -> EntityExtractionResult:
        """
        Extract entities from text using spaCy NER
        """
        if not self.nlp:
            raise RuntimeError("spaCy model not loaded")
        
        doc = self.nlp(text)
        
        # Extract entities by type
        entities = {}
        people = []
        locations = []
        organizations = []
        dates = []
        
        for ent in doc.ents:
            entity_type = ent.label_
            entity_text = ent.text.strip()
            
            if entity_type not in entities:
                entities[entity_type] = []
            entities[entity_type].append(entity_text)
            
            # Categorize common entity types
            if entity_type in ["PERSON"]:
                people.append(entity_text)
            elif entity_type in ["GPE", "LOC", "FACILITY"]:  # Geopolitical, Location, Facility
                locations.append(entity_text)
            elif entity_type in ["ORG"]:
                organizations.append(entity_text)
            elif entity_type in ["DATE", "TIME"]:
                # Try to parse the date
                parsed_date = self._parse_date(entity_text, text)
                if parsed_date:
                    dates.append(parsed_date)
        
        # Extract additional dates using dateparser
        additional_dates = self._extract_dates_with_dateparser(text)
        dates.extend(additional_dates)
        
        # Remove duplicates
        dates = list(set(dates))
        people = list(set(people))
        locations = list(set(locations))
        organizations = list(set(organizations))
        
        # Extract keywords
        keywords = self._extract_keywords(doc)
        
        return EntityExtractionResult(
            entities=entities,
            dates=dates,
            people=people,
            locations=locations,
            organizations=organizations,
            keywords=keywords
        )
    
    def _parse_date(self, date_text: str, context: str) -> Optional[datetime]:
        """Parse date using dateparser with context"""
        try:
            from datetime import datetime, timedelta
            now = datetime.now()

            # Check for relative time patterns first
            relative_match = re.search(r'in\s+(?:a|one|\d+)\s+(day|days|week|weeks|month|months)', date_text, re.IGNORECASE)
            if relative_match:
                # Extract number and unit
                number_match = re.search(r'(?:a|one|\d+)', date_text, re.IGNORECASE)
                number = 1 if number_match.group().lower() in ['a', 'one'] else int(number_match.group())
                unit = relative_match.group(1).lower()
                
                # Calculate delta
                if unit.startswith('day'):
                    delta = timedelta(days=number)
                elif unit.startswith('week'):
                    delta = timedelta(weeks=number)
                elif unit.startswith('month'):
                    # Approximate months as 30 days
                    delta = timedelta(days=number * 30)
                
                return now + delta

            # Check for 'before/by/until' patterns
            before_match = re.search(r'(before|by|until|till)\s+(.+)', date_text, re.IGNORECASE)
            if before_match:
                date_part = before_match.group(2)
            else:
                date_part = date_text

            # Use dateparser for flexible date parsing
            parsed = dateparser.parse(
                date_part,
                settings={
                    'PREFER_DAY_OF_MONTH': 'first',
                    'PREFER_DATES_FROM': 'future',
                    'RETURN_AS_TIMEZONE_AWARE': False,
                    'RELATIVE_BASE': now
                }
            )
            
            # If we found a date and it was a 'before' type date, set time to end of day
            if parsed and before_match:
                parsed = parsed.replace(hour=23, minute=59, second=59)
            
            return parsed
        except Exception as e:
            return None
    
    def _extract_dates_with_dateparser(self, text: str) -> List[datetime]:
        """Extract dates using dateparser with common patterns"""
        dates = []
        
        # Common date patterns
        date_patterns = [
            r'\b(next|this)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b',
            r'\b(today|tomorrow|yesterday)\b',
            r'\b(next|this)\s+(week|month|year)\b',
            r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b',
            r'\b(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}\b',
            r'\b(before|by|until|till)\s+(january|february|march|april|may|june|july|august|september|october|november|december)\s+\d{1,2}(st|nd|rd|th)?\b',
            r'\b(before|by|until|till)\s+\d{1,2}(st|nd|rd|th)?\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b',
            r'\bin\s+(\d+)\s+(day|days|week|weeks|month|months)\b',
            r'\bafter\s+(\d+)\s+(day|days|week|weeks|month|months)\b',
            r'\b(a|one)\s+(day|week|month)\b'
        ]
        
        for pattern in date_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                date_text = match.group()
                parsed_date = self._parse_date(date_text, text)
                if parsed_date:
                    dates.append(parsed_date)
        
        return dates
    
    def _extract_keywords(self, doc) -> List[str]:
        """Extract keywords from spaCy doc"""
        keywords = []
        
        # Extract important tokens (nouns, adjectives, proper nouns)
        for token in doc:
            if (token.pos_ in ["NOUN", "PROPN", "ADJ"] and 
                not token.is_stop and 
                not token.is_punct and 
                len(token.text) > 2):
                keywords.append(token.lemma_.lower())
        
        # Remove duplicates and limit
        keywords = list(set(keywords))[:self.config.max_keywords]
        return keywords

class CardClassifier:
    """
    Classifies text into card types (Task, Reminder, Idea)
    """
    
    def __init__(self, config: NLPConfig = None):
        self.config = config or NLPConfig()
        
        # Define classification patterns
        self.task_patterns = [
            r'\b(call|email|send|contact|reach out|schedule|meet|discuss)\b',
            r'\b(do|complete|finish|submit|prepare|create|make|build)\b',
            r'\b(review|check|verify|confirm|update|fix|resolve)\b',
            r'\b(buy|purchase|order|get|pick up|deliver)\b'
        ]
        
        self.reminder_patterns = [
            r'\b(remember|remind|don\'t forget)\b',
            r'\b(pick up|grab|bring|take)\b',
            r'\b(appointment|meeting|deadline|due)\b',
            r'\b(at \d|on (monday|tuesday|wednesday|thursday|friday|saturday|sunday))\b'
        ]
        
        self.idea_patterns = [
            r'\b(idea|thought|concept|note|maybe|consider)\b',
            r'\b(should|could|might|what if|suggestion)\b',
            r'\b(design|color|style|theme|approach)\b'
        ]
    
    def classify(self, text: str, entities: EntityExtractionResult, llm=None) -> ClassificationResult:
        """
        Classify text into card type using pattern matching and optionally LLM
        
        Args:
            text: Text to classify
            entities: Extracted entities
            llm: Optional LLMManager instance for enhanced classification
        """
        text_lower = text.lower()
        
        # Calculate pattern scores
        task_score = self._calculate_pattern_score(text_lower, self.task_patterns)
        reminder_score = self._calculate_pattern_score(text_lower, self.reminder_patterns)
        idea_score = self._calculate_pattern_score(text_lower, self.idea_patterns)
        
        # Adjust scores based on entities
        if entities.dates:
            reminder_score += 0.3
            task_score += 0.2
        
        if entities.people:
            task_score += 0.2
        
        # Determine predicted type
        scores = {
            CardType.TASK: task_score,
            CardType.REMINDER: reminder_score,
            CardType.IDEA: idea_score
        }
        
        predicted_type = max(scores, key=scores.get)
        max_score = scores[predicted_type]
        
        # Generate reasoning
        reasoning = self._generate_reasoning(predicted_type, entities, max_score)
        
        return ClassificationResult(
            predicted_type=predicted_type,
            reasoning=reasoning
        )
    
    def _calculate_pattern_score(self, text: str, patterns: List[str]) -> float:
        """Calculate score based on pattern matches"""
        score = 0.0
        for pattern in patterns:
            matches = len(re.findall(pattern, text, re.IGNORECASE))
            score += matches * 0.3
        return min(score, 1.0)  # Cap at 1.0
    
    def _generate_reasoning(self, predicted_type: CardType, entities: EntityExtractionResult, score: float) -> str:
        """Generate reasoning for the classification"""
        reasons = []
        
        if predicted_type == CardType.TASK:
            reasons.append("Contains action verbs or task-related keywords")
            if entities.people:
                reasons.append(f"Mentions people: {', '.join(entities.people[:2])}")
        elif predicted_type == CardType.REMINDER:
            reasons.append("Contains reminder-related keywords")
            if entities.dates:
                reasons.append("Contains time-based information")
        else:  # IDEA
            reasons.append("Contains conceptual or ideation keywords")
        
        if entities.keywords:
            reasons.append(f"Key topics: {', '.join(entities.keywords[:3])}")
        
        return "; ".join(reasons)

class EnvelopeMatcher:
    """
    Matches cards to appropriate envelopes using semantic similarity
    """
    
    def __init__(self, config: NLPConfig = None):
        self.config = config or NLPConfig()
        self.sentence_model = None
        self._load_models()
    
    def _load_models(self):
        """Load sentence transformer model"""
        try:
            self.sentence_model = SentenceTransformer(self.config.sentence_transformer_model)
        except Exception as e:
            # Fallback to keyword-based matching
            self.sentence_model = None
    
    def find_best_envelope(
        self, 
        text: str, 
        entities: EntityExtractionResult,
        existing_envelopes: List[Dict[str, Any]]
    ) -> EnvelopeMatchResult:
        """
        Find the best matching envelope for the given text and entities
        """
        if not existing_envelopes:
            return self._suggest_new_envelope(text, entities)
        
        if self.sentence_model:
            return self._semantic_matching(text, entities, existing_envelopes)
        else:
            return self._keyword_matching(text, entities, existing_envelopes)
    
    def _semantic_matching(
        self, 
        text: str, 
        entities: EntityExtractionResult,
        existing_envelopes: List[Dict[str, Any]]
    ) -> EnvelopeMatchResult:
        """Use semantic similarity for envelope matching"""
        
        # Create embedding for the input text
        input_embedding = self.sentence_model.encode([text], show_progress_bar=False)
        
        best_match = None
        best_score = 0.0
        
        for envelope in existing_envelopes:
            # Create envelope text for comparison
            envelope_text = f"{envelope['name']} {envelope.get('description', '')}"
            envelope_embedding = self.sentence_model.encode([envelope_text], show_progress_bar=False)
            
            # Calculate similarity
            similarity = cosine_similarity(input_embedding, envelope_embedding)[0][0]
            
            if similarity > best_score:
                best_score = similarity
                best_match = envelope
        
        # Decide whether to use existing envelope or create new one
        if best_score >= self.config.envelope_similarity_threshold:
            return EnvelopeMatchResult(
                matched_envelope=best_match,
                similarity_score=best_score,
                should_create_new=False
            )
        else:
            return self._suggest_new_envelope(text, entities)
    
    def _keyword_matching(
        self, 
        text: str, 
        entities: EntityExtractionResult,
        existing_envelopes: List[Dict[str, Any]]
    ) -> EnvelopeMatchResult:
        """Fallback keyword-based matching"""
        
        # Extract keywords from text
        text_keywords = set(entities.keywords + entities.people + entities.organizations)
        
        best_match = None
        best_score = 0.0
        
        for envelope in existing_envelopes:
            # Get envelope keywords
            envelope_keywords = set(envelope.get('keywords', []))
            envelope_name_words = set(envelope['name'].lower().split())
            all_envelope_keywords = envelope_keywords.union(envelope_name_words)
            
            # Calculate keyword overlap
            if all_envelope_keywords:
                overlap = len(text_keywords.intersection(all_envelope_keywords))
                score = overlap / len(all_envelope_keywords)
                
                if score > best_score:
                    best_score = score
                    best_match = envelope
        
        # Use fuzzy string matching for envelope names if available
        if FUZZY_AVAILABLE:
            envelope_names = [env['name'] for env in existing_envelopes]
            fuzzy_matches = process.extractBests(text, envelope_names, limit=1, score_cutoff=70)
            
            if fuzzy_matches and (not best_match or fuzzy_matches[0][1] / 100 > best_score):
                matched_name = fuzzy_matches[0][0]
                best_match = next(env for env in existing_envelopes if env['name'] == matched_name)
                best_score = fuzzy_matches[0][1] / 100
        
        if best_score >= 0.5:  # Lower threshold for keyword matching
            return EnvelopeMatchResult(
                matched_envelope=best_match,
                similarity_score=best_score,
                should_create_new=False
            )
        else:
            return self._suggest_new_envelope(text, entities)
    
    def _suggest_new_envelope(self, text: str, entities: EntityExtractionResult) -> EnvelopeMatchResult:
        """Suggest creating a new envelope based on note content"""
        
        # Determine envelope type and name based on entities
        suggested_name = None
        suggested_type = EnvelopeType.GENERAL
        
        # Check for people (might be a person-based envelope)
        if entities.people:
            suggested_name = entities.people[0]
            suggested_type = EnvelopeType.PERSON
        
        # Check for organizations (might be a company-based envelope)
        elif entities.organizations:
            suggested_name = entities.organizations[0]
            suggested_type = EnvelopeType.COMPANY
        
        # Use keywords for theme-based envelopes
        elif entities.keywords and len(entities.keywords) > 0:
            # Combine top 2-3 keywords for more descriptive name
            num_keywords = min(2, len(entities.keywords))
            suggested_name = " ".join([kw.title() for kw in entities.keywords[:num_keywords]])
            suggested_type = EnvelopeType.THEME
        
        # Fallback: Extract meaningful words from text (nouns, verbs)
        else:
            words = []
            # Remove common stop words and short words
            stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'my', 'your', 'his', 'her', 'its', 'our', 'their'}
            text_words = text.lower().split()
            
            for word in text_words:
                # Clean word
                clean_word = re.sub(r'[^\w]', '', word)
                if clean_word and len(clean_word) > 2 and clean_word not in stop_words:
                    words.append(clean_word.title())
                if len(words) >= 3:
                    break
            
            if words:
                suggested_name = " ".join(words)
            else:
                # Last resort: use first 3-4 words
                suggested_name = " ".join(text.split()[:3]).title()
        
        return EnvelopeMatchResult(
            matched_envelope=None,
            similarity_score=0.0,
            should_create_new=True,
            suggested_name=suggested_name,
            suggested_type=suggested_type
        )

class NLPPipeline:
    """
    Main NLP pipeline that coordinates all components
    """
    
    def __init__(self, config: NLPConfig = None):
        self.config = config or NLPConfig()
        
        # Initialize components (spaCy is always available)
        self.entity_extractor = EntityExtractor(self.config)
        self.card_classifier = CardClassifier(self.config)
        self.envelope_matcher = EnvelopeMatcher(self.config)
        self.available = True
    
    def process(
        self, 
        text: str, 
        existing_envelopes: List[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process text through the complete NLP pipeline
        """
        start_time = datetime.now()
        
        try:
            if not self.available:
                # Fallback processing without spaCy
                return self._fallback_process(text, existing_envelopes)
            
            # Extract entities
            entities = self.entity_extractor.extract_entities(text)
            
            # Classify card type
            classification = self.card_classifier.classify(text, entities)
            
            # Find best envelope
            envelope_match = self.envelope_matcher.find_best_envelope(
                text, entities, existing_envelopes or []
            )
            
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                "success": True,
                "entities": entities,
                "classification": classification,
                "envelope_match": envelope_match,
                "processing_time_ms": int(processing_time)
            }
            
        except Exception as e:
            processing_time = (datetime.now() - start_time).total_seconds() * 1000
            
            return {
                "success": False,
                "error": str(e),
                "processing_time_ms": int(processing_time)
            }
    
    def _fallback_process(self, text: str, existing_envelopes: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Fallback processing without spaCy dependencies
        """
        from ..models.schemas import EntityExtractionResult, ClassificationResult, EnvelopeMatchResult, CardType, EnvelopeType
        
        # Simple entity extraction using regex
        # Extract meaningful keywords (remove stop words)
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'i', 'you', 'he', 'she', 'it', 'we', 'they'}
        keywords = [word.lower() for word in text.split() if len(word) > 2 and word.lower() not in stop_words][:5]
        
        entities = EntityExtractionResult(
            people=[],
            organizations=[],
            locations=[],
            dates=[],
            keywords=keywords,
            entities={}
        )
        
        # Simple classification based on keywords
        task_keywords = ['task', 'do', 'call', 'email', 'buy', 'get', 'make', 'create', 'fix']
        reminder_keywords = ['remember', 'remind', 'don\'t forget', 'note']
        
        card_type = CardType.IDEA  # Default
        if any(word in text.lower() for word in task_keywords):
            card_type = CardType.TASK
        elif any(word in text.lower() for word in reminder_keywords):
            card_type = CardType.REMINDER
        
        classification = ClassificationResult(
            predicted_type=card_type,
            reasoning="Fallback classification based on simple keyword matching"
        )
        
        # Generate meaningful envelope name from keywords
        if keywords:
            suggested_name = " ".join([kw.title() for kw in keywords[:2]])
        else:
            # Extract first meaningful words
            words = []
            for word in text.split():
                clean_word = re.sub(r'[^\w]', '', word)
                if clean_word and len(clean_word) > 2 and clean_word.lower() not in stop_words:
                    words.append(clean_word.title())
                if len(words) >= 2:
                    break
            suggested_name = " ".join(words) if words else " ".join(text.split()[:3]).title()
        
        # Simple envelope matching
        envelope_match = EnvelopeMatchResult(
            matched_envelope=None,
            similarity_score=0.5,
            should_create_new=True,
            suggested_name=suggested_name,
            suggested_type=EnvelopeType.THEME
        )
        
        return {
            "success": True,
            "entities": entities,
            "classification": classification,
            "envelope_match": envelope_match,
            "processing_time_ms": 10
        }