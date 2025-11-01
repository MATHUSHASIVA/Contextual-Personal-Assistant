"""
LLM integration module using Groq's Llama model
"""

import os
import re
import json
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from dataclasses import dataclass

# Optional LLM imports
if TYPE_CHECKING:
    from langchain.callbacks.manager import CallbackManagerForLLMRun

try:
    from langchain_groq import ChatGroq
    from langchain.llms.base import LLM
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False
    ChatGroq = None
    LLM = None

@dataclass
class LLMConfig:
    """Configuration for LLM"""
    api_key: Optional[str] = None
    model: str = "llama-3.3-70b-versatile"  # Default Groq model
    temperature: float = 0.1
    max_tokens: int = 5000  # Token limit for Groq
    provider: str = "groq"
    
    def __init__(self, **kwargs):
        # Load model from environment if available
        model_override = os.getenv("LLM_MODEL")
        if model_override:
            self.model = model_override
        # Override any attributes with provided kwargs
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)

class LLMManager:
    """Manages LLM interactions using OpenRouter's Deepseek Chat model"""
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.config.api_key = self.config.api_key or os.getenv("GROQ_API_KEY")
        self.available = LLM_AVAILABLE
        if self.available:
            self._setup_client()
        else:
            self.client = None
        
    def _setup_client(self):
        """Initialize Groq Chat client"""
        if not LLM_AVAILABLE:
            self.client = None
            return
            
        if not self.config.api_key:
            raise ValueError("API key is required for LLM functionality")
            
        try:
            self.client = ChatGroq(
                model=self.config.model,
                api_key=self.config.api_key,
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens
            )
            # Test the connection with a more comprehensive query
            test_prompt = """Convert this note to structured data: "Test task: review design"
            Output ONLY valid JSON with this structure:
            {
                "card": {
                    "type": "Task",
                    "description": "Review design",
                    "date": "none",
                    "assignee": "none",
                    "context_keywords": ["design", "review", "test"]
                },
                "suggested_envelope": "Design Reviews",
                "new_context": {
                    "projects": [],
                    "people": [],
                    "companies": [],
                    "themes": ["design"]
                }
            }"""
            
            response = self.generate_response(test_prompt, temperature=0.1)
            if not response:
                raise ConnectionError("Failed to get response from LLM")
            
            # Verify JSON parsing
            try:
                test_result = json.loads(response.strip().replace('```json', '').replace('```', ''))
                if not all(k in test_result for k in ['card', 'suggested_envelope', 'new_context']):
                    raise ValueError("Test response missing required fields")
            except Exception as e:
                raise ConnectionError(f"LLM response validation failed: {str(e)}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to initialize LLM: {str(e)}")
            
    @property
    def _llm_type(self) -> str:
        """Return identifier of llm."""
        return "groq_llama"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional['CallbackManagerForLLMRun'] = None,
        **kwargs: Any,
    ) -> str:
        """Execute the LLM call."""
        if not self.client:
            raise ValueError("LLM is not initialized - API key required")
            
        try:
            response = self.client.invoke(prompt)
            # Extract content from AIMessage object
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
        except Exception as e:
            raise
    
    def generate_response(self, 
                         prompt: str,
                         system_prompt: Optional[str] = None,
                         temperature: Optional[float] = None) -> Optional[str]:
        """
        Generate a response using the LLM
        
        Args:
            prompt: The user prompt
            system_prompt: Optional system prompt for context
            temperature: Optional temperature override
        
        Returns:
            Generated response text or None if LLM is not available
        """
        if not self.client:
            return None
            
        try:
            # Combine system prompt and user prompt if both are provided
            full_prompt = f"{system_prompt}\n\n{prompt}" if system_prompt else prompt
            
            # Update temperature if provided
            if temperature is not None:
                self.client.temperature = temperature
                
            response = self.client.invoke(full_prompt)
            # Extract content from AIMessage object
            if hasattr(response, 'content'):
                return response.content
            else:
                return str(response)
            
        except Exception as e:
            return None
    
    def process_note(self, text: str, current_envelopes: List[Dict] = None, user_context: Dict = None) -> Dict[str, Any]:
        """
        Process a note into a structured card with comprehensive context
        
        Args:
            text: The note to process
            current_envelopes: List of current envelopes
            user_context: Current user context
            
        Returns:
            Dictionary containing card details, envelope suggestion, and context updates
        """
        if not self.client:
            # Improved fallback processing without LLM
            from datetime import datetime
            import re
            
            # Basic date extraction
            due_date = "none"
            date_patterns = [
                (r'by\s+(\w+\s+\d+)', lambda m: self._parse_fallback_date(m.group(1))),
                (r'on\s+(\w+\s+\d+)', lambda m: self._parse_fallback_date(m.group(1))),
                (r'(\w+\s+\d+)', lambda m: self._parse_fallback_date(m.group(1)))
            ]
            
            for pattern, parser in date_patterns:
                match = re.search(pattern, text.lower())
                if match:
                    due_date = parser(match)
                    break
            
            # Basic card type detection
            card_type = "task"
            if any(word in text.lower() for word in ["idea", "think", "consider"]):
                card_type = "idea"
            elif any(word in text.lower() for word in ["remind", "remember", "don't forget"]):
                card_type = "reminder"
                
            # Basic envelope detection
            envelope_name = "General"
            envelope_type = "general"
            
            if any(word in text.lower() for word in ["bill", "payment", "pay"]):
                envelope_name = "Bills"
                envelope_type = "theme"
            elif any(word in text.lower() for word in ["grocery", "groceries", "shopping"]):
                envelope_name = "Groceries"
                envelope_type = "theme"
            elif any(word in text.lower() for word in ["meeting", "call"]):
                envelope_name = "Meetings"
                envelope_type = "theme"
            
            # Extract basic keywords
            keywords = []
            words = re.findall(r'\b\w+\b', text.lower())
            skip_words = {'the', 'and', 'or', 'but', 'to', 'for', 'by', 'on', 'at', 'in', 'a', 'an'}
            for word in words:
                if len(word) > 2 and word not in skip_words:
                    keywords.append(word)
                if len(keywords) >= 5:
                    break
                    
            return {
                "card": {
                    "type": card_type,
                    "description": text,
                    "assignee": "none",
                    "date": due_date,
                    "context_keywords": keywords
                },
                "suggested_envelope": {
                    "name": envelope_name,
                    "type": envelope_type
                },
                "new_context": {
                    "projects": [],
                    "people": [],
                    "companies": [],
                    "themes": []
                }
            }
        system_prompt = """You are a JSON data processor that converts notes to structured data. Return ONLY a single-line JSON object.

RULES FOR ENVELOPE NAMES AND TYPES:
Categorize envelopes into these exact types:
- "project": Named projects, specific initiatives (e.g., "Alpha", "Website Redesign")
- "company": Business entities, clients, organizations (e.g., "Acme Corp", "Microsoft") 
- "person": Individual people mentioned (e.g., "Sarah", "John")
- "theme": Functional categories, recurring activities (e.g., "Bills", "Groceries", "Meetings", "Budget Review")
- "general": Default for uncategorized items

NAMING CONVENTION - NO PREFIXES:
- Projects: Just the project name (e.g., "Alpha", "Marketing Campaign", "Website Redesign")
- Company: Just the company name (e.g., "Acme Corp", "Logo Design") 
- Personal: Just the person's name (e.g., "Sarah", "John")
- Theme: Just the category name (e.g., "Bills", "Groceries", "Meetings", "Budget Review")
- General: Descriptive category (e.g., "Tasks", "Ideas")

IMPORTANT: Do NOT use prefixes like "Project:", "Person:", "Company:" in envelope names. Use clean names only.

ENVELOPE TYPE EXAMPLES:
- "Pay credit card bill" -> theme: "Bills"
- "Call Sarah" -> person: "Sarah" 
- "Project Alpha meeting" -> project: "Alpha"
- "Acme Corp proposal" -> company: "Acme Corp"

Note to process: "{note}"
Current Envelopes: {envelopes}
User Context: {context}

DATE HANDLING RULES:
- For deadlines (before/by/until dates), set the time to 23:59:59
- Always return dates in YYYY-MM-DD HH:MM:SS format for deadlines
- Convert ALL dates to absolute dates using current date as base:
  * "October 30th", "Oct 30", "30th October" -> "2025-10-30 23:59:59"
  * "in X days/weeks/months" -> calculate from current date
  * "next Monday", "tomorrow" -> calculate from current date
  * "after X days" -> calculate from current date
  * For relative expressions, set time to 23:59:59
- For "before/by/until" dates, use the specified date as the deadline
- Current date is: {current_date}
- Examples:
  * "October 30th" -> "2025-10-30 23:59:59" 
  * "by November 5" -> "2025-11-05 23:59:59"
  * "in 2 days" on 2025-10-30 -> "2025-11-01 23:59:59"
  * "next week" on 2025-10-30 -> "2025-11-06 23:59:59"

KEYWORD EXTRACTION RULES:
- Extract 4-6 relevant keywords from the note
- Include ACTION words (book, call, buy, schedule, draft, research, etc.)
- Include OBJECT/TOPIC words (flight, tickets, meeting, budget, etc.)
- Include LOCATION/PEOPLE words when mentioned (seattle, sarah, office, etc.)
- Use lowercase, single words only
- Skip common words like "the", "and", "for", "about"
- Examples:
  * "Book flight tickets for Seattle" -> ["book", "flight", "tickets", "seattle"]
  * "Call Sarah about budget review" -> ["call", "sarah", "budget", "review"]
 
Return EXACTLY this structure on a SINGLE LINE (no line breaks, no spaces after colons):
{{"card":{{"type":"task|reminder|idea","description":"text","date":"YYYY-MM-DD HH:MM:SS|none","assignee":"name|none","context_keywords":["4-6","keywords"]}},"suggested_envelope":{{"name":"Clean Name","type":"project|company|person|theme|general"}},"new_context":{{"projects":[],"people":[],"companies":[],"themes":[]}}}}"""
        
        from datetime import datetime
        
        # Format the prompt with current context and date
        prompt = system_prompt.format(
            envelopes=json.dumps(current_envelopes or [], indent=2),
            context=json.dumps(user_context or {}, indent=2),
            note=text,
            current_date=datetime.now().strftime('%Y-%m-%d')
        )
        
        response = self.generate_response(prompt, temperature=0.1)
        if not response:
            raise RuntimeError("Failed to get LLM response")
            
        try:
            # Aggressively clean the response
            response = response.strip()
            
            # Remove any markdown formatting
            if '```' in response:
                response = response.split('```')[1] if response.count('```') >= 2 else response.replace('```', '')
            
            # Remove any JSON code block identifier
            if response.startswith('json'):
                response = response[4:].lstrip()
                
            # Clean all whitespace and formatting
            response = re.sub(r'\s+', ' ', response)  # Replace all whitespace sequences with single space
            response = response.replace(' "', '"').replace('{ ', '{').replace(' }', '}')  # Remove spaces around brackets
            response = response.strip('` \t\n\r')  # Remove any remaining backticks and whitespace
            
            # Attempt to parse JSON with error handling
            try:
                result = json.loads(response)
            except json.JSONDecodeError as e:
                # Try one more time with an even more aggressive cleaning
                try:
                    # Remove all whitespace between JSON elements
                    response = re.sub(r'"\s*:\s*"', '":"', response)
                    response = re.sub(r'"\s*,\s*"', '","', response)
                    response = re.sub(r'"\s*}', '"}', response)
                    response = re.sub(r'{\s*"', '{"', response)
                    result = json.loads(response)
                except json.JSONDecodeError as e2:
                    raise ValueError("Failed to parse LLM response as JSON")
            
            # Process dates in the response
            if result.get("card", {}).get("date") and result["card"]["date"] != "none":
                try:
                    # Parse the date from the response
                    from datetime import datetime
                    date_str = result["card"]["date"]
                    
                    # Check if time is included
                    if ' ' in date_str:
                        parsed_date = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
                    else:
                        # Add end of day time for deadline-style dates
                        parsed_date = datetime.strptime(date_str, '%Y-%m-%d').replace(hour=23, minute=59, second=59)
                    
                    # Store the datetime object directly
                    result["card"]["date"] = parsed_date
                except ValueError as e:
                    result["card"]["date"] = "none"
            
            # Validate the response structure
            required_fields = {
                "card": ["type", "description", "date", "assignee", "context_keywords"],
                "suggested_envelope": None,
                "new_context": ["projects", "people", "companies", "themes"]
            }
            
            for field, subfields in required_fields.items():
                if field not in result:
                    raise ValueError(f"Missing required field: {field}")
                if subfields and field in result:
                    for subfield in subfields:
                        if subfield not in result[field]:
                            raise ValueError(f"Missing required subfield: {field}.{subfield}")
            
            return result
            
        except Exception as e:
            raise RuntimeError(f"Failed to process note: {str(e)}")
    
    def _parse_fallback_date(self, date_text: str) -> str:
        """Parse date text for fallback processing"""
        from datetime import datetime
        import re
        
        try:
            current_date = datetime.now()
            
            # Handle "October 30th", "Oct 30", etc.
            month_names = {
                'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
                'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 
                'july': 7, 'jul': 7, 'august': 8, 'aug': 8, 'september': 9, 'sep': 9,
                'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
            }
            
            # Extract month and day
            date_text = date_text.lower().strip()
            
            for month_name, month_num in month_names.items():
                if month_name in date_text:
                    # Extract day number
                    day_match = re.search(r'(\d+)', date_text)
                    if day_match:
                        day = int(day_match.group(1))
                        year = current_date.year
                        
                        # If the date has passed this year, assume next year
                        try_date = datetime(year, month_num, day)
                        if try_date < current_date:
                            year += 1
                        
                        return f"{year:04d}-{month_num:02d}-{day:02d} 23:59:59"
            
            return "none"
        except Exception:
            return "none"
            
    def classify_card_type(self, text: str) -> str:
        """
        Classify text into card types
        
        Args:
            text: Text to classify
        
        Returns:
            Card type string (TASK, REMINDER, or IDEA)
        """
        try:
            # Use the comprehensive processor
            result = self.process_note(text)
            return result["card"]["type"].upper()
            
        except:
            return "TASK"

    def extract_entities(self, text: str) -> Dict[str, Any]:
        """
        Extract entities from text using LLM
        
        Args:
            text: Text to extract entities from
        
        Returns:
            Dictionary of extracted entities
        """
        try:
            # Use the comprehensive processor
            result = self.process_note(text)
            return {
                "description": result["card"]["description"],
                "date": result["card"]["date"] if result["card"]["date"] != "none" else None,
                "assignee": result["card"]["assignee"] if result["card"]["assignee"] != "none" else None,
                "context_keywords": result["card"]["context_keywords"]
            }
        except:
            return {
                "description": text,
                "date": None,
                "assignee": None,
                "context_keywords": []
            }