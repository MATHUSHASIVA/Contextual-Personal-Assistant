"""
Pydantic models for data validation and serialization
"""

from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum

# Enums for constrained values
class CardType(str, Enum):
    TASK = "task"
    REMINDER = "reminder"
    IDEA = "idea"

class Status(str, Enum):
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    CANCELLED = "cancelled"

class EnvelopeType(str, Enum):
    PROJECT = "project"
    COMPANY = "company"
    PERSON = "person"
    THEME = "theme"
    GENERAL = "general"

class ContextType(str, Enum):
    PROJECT = "project"
    COMPANY = "company"
    PERSON = "person"
    THEME = "theme"
    WORK = "work"
    PERSONAL = "personal"
    TRAVEL = "travel"
    PREFERENCE = "preference"

class InsightType(str, Enum):
    NEXT_STEP = "next_step"
    RECOMMENDATION = "recommendation"
    CONFLICT = "conflict"
    OPTIMIZATION = "optimization"

# Core data models
class CardEntity(BaseModel):
    """
    Extracted entities from a card
    """
    description: str = Field(..., description="Core description of the card")
    assignee: Optional[str] = Field(None, description="Person or team assigned")
    due_date: Optional[datetime] = Field(None, description="Extracted or inferred due date")
    location: Optional[str] = Field(None, description="Extracted location")
    context_keywords: List[str] = Field(default_factory=list, description="Auto-extracted keywords")
    additional_entities: Dict[str, Any] = Field(default_factory=dict, description="Other extracted entities")

class CardCreate(BaseModel):
    """
    Model for creating a new card
    """
    content: str = Field(..., min_length=1, description="Original raw text")
    description: str = Field(..., min_length=1, description="Processed description")
    card_type: CardType = Field(..., description="Type of card")
    status: Status = Field(Status.PENDING, description="Current status")
    entities: CardEntity = Field(..., description="Extracted entities")
    envelope_id: Optional[int] = Field(None, description="Associated envelope ID")

class CardUpdate(BaseModel):
    """
    Model for updating an existing card
    """
    description: Optional[str] = Field(None, min_length=1)
    card_type: Optional[CardType] = None
    status: Optional[Status] = None
    assignee: Optional[str] = None
    due_date: Optional[datetime] = None
    location: Optional[str] = None
    envelope_id: Optional[int] = None

class CardResponse(BaseModel):
    """
    Model for card responses
    """
    id: int
    content: str
    description: str
    card_type: CardType
    status: Status
    assignee: Optional[str]
    due_date: Optional[datetime]
    location: Optional[str]
    envelope_id: Optional[int]
    envelope_name: Optional[str]
    keywords: List[str]
    created_at: datetime
    updated_at: datetime
    additional_entities: Dict[str, Any]

    class Config:
        from_attributes = True

class EnvelopeCreate(BaseModel):
    """
    Model for creating a new envelope
    """
    name: str = Field(..., min_length=1, max_length=200, description="Envelope name")
    description: Optional[str] = Field(None, description="Envelope description")
    envelope_type: EnvelopeType = Field(EnvelopeType.GENERAL, description="Type of envelope")
    keywords: List[str] = Field(default_factory=list, description="Associated keywords")

class EnvelopeUpdate(BaseModel):
    """
    Model for updating an existing envelope
    """
    name: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    envelope_type: Optional[EnvelopeType] = None
    is_active: Optional[bool] = None

class EnvelopeResponse(BaseModel):
    """
    Model for envelope responses
    """
    id: int
    name: str
    description: Optional[str]
    envelope_type: EnvelopeType
    card_count: int
    keywords: List[str]
    created_at: datetime
    updated_at: datetime
    is_active: bool

    class Config:
        from_attributes = True

class UserContextCreate(BaseModel):
    """
    Model for creating user context
    """
    context_type: ContextType = Field(..., description="Type of context")
    name: str = Field(..., min_length=1, max_length=200, description="Context name")
    description: Optional[str] = Field(None, description="Context description")
    relevance_score: float = Field(1.0, ge=0.0, le=1.0, description="Relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context data")
    related_envelope_id: Optional[int] = Field(None, description="Related envelope ID")

class UserContextResponse(BaseModel):
    """
    Model for user context responses
    """
    id: int
    context_type: ContextType
    name: str
    description: Optional[str]
    relevance_score: float
    metadata: Dict[str, Any]
    related_envelope_id: Optional[int]
    created_at: datetime
    updated_at: datetime
    last_referenced: datetime
    is_active: bool

    class Config:
        from_attributes = True

class ProcessingResult(BaseModel):
    """
    Result of processing a raw note
    """
    success: bool = Field(..., description="Whether processing was successful")
    card: Optional[CardResponse] = Field(None, description="Created card if successful")
    envelope: Optional[EnvelopeResponse] = Field(None, description="Assigned or created envelope")
    processing_time_ms: int = Field(..., description="Processing time in milliseconds")
    error_message: Optional[str] = Field(None, description="Error message if failed")
    extracted_entities: Dict[str, Any] = Field(default_factory=dict, description="All extracted entities")

class ThinkingInsightCreate(BaseModel):
    """
    Model for creating thinking insights
    """
    insight_type: InsightType = Field(..., description="Type of insight")
    title: str = Field(..., min_length=1, max_length=200, description="Insight title")
    description: str = Field(..., min_length=1, description="Detailed description")
    related_card_ids: List[int] = Field(default_factory=list, description="Related card IDs")
    related_envelope_ids: List[int] = Field(default_factory=list, description="Related envelope IDs")
    is_actionable: bool = Field(False, description="Whether this insight is actionable")
    suggested_action: Optional[str] = Field(None, description="Suggested action if actionable")

    expires_at: Optional[datetime] = Field(None, description="When this insight expires")

class ThinkingInsightResponse(BaseModel):
    """
    Model for thinking insight responses
    """
    id: int
    insight_type: InsightType
    title: str
    description: str
    related_card_ids: List[int]
    related_envelope_ids: List[int]
    is_actionable: bool
    suggested_action: Optional[str]
    is_dismissed: bool
    is_acted_upon: bool
    created_at: datetime
    expires_at: Optional[datetime]

    class Config:
        from_attributes = True

class NLPProcessingRequest(BaseModel):
    """
    Request for NLP processing
    """
    text: str = Field(..., min_length=1, description="Text to process")
    user_context: Optional[List[UserContextResponse]] = Field(None, description="Current user context")
    existing_envelopes: Optional[List[EnvelopeResponse]] = Field(None, description="Existing envelopes for matching")

class EntityExtractionResult(BaseModel):
    """
    Result of entity extraction
    """
    entities: Dict[str, List[str]] = Field(default_factory=dict, description="Extracted entities by type")
    dates: List[datetime] = Field(default_factory=list, description="Extracted dates")
    people: List[str] = Field(default_factory=list, description="Extracted person names")
    locations: List[str] = Field(default_factory=list, description="Extracted locations")
    organizations: List[str] = Field(default_factory=list, description="Extracted organizations")
    keywords: List[str] = Field(default_factory=list, description="Extracted keywords")

class ClassificationResult(BaseModel):
    """
    Result of card type classification
    """
    predicted_type: CardType = Field(..., description="Predicted card type")
    reasoning: Optional[str] = Field(None, description="Reasoning for the classification")

class EnvelopeMatchResult(BaseModel):
    """
    Result of envelope matching
    """
    matched_envelope: Optional[EnvelopeResponse] = Field(None, description="Best matching envelope")
    similarity_score: float = Field(0.0, ge=0.0, le=1.0, description="Similarity score")
    should_create_new: bool = Field(False, description="Whether to create a new envelope")
    suggested_name: Optional[str] = Field(None, description="Suggested name for new envelope")
    suggested_type: Optional[EnvelopeType] = Field(None, description="Suggested type for new envelope")

# Utility models
class SystemStats(BaseModel):
    """
    System statistics
    """
    total_cards: int
    total_envelopes: int
    active_contexts: int
    recent_insights: int
    processing_accuracy: float
    average_processing_time_ms: float

class BulkProcessingRequest(BaseModel):
    """
    Request for processing multiple notes at once
    """
    notes: List[str] = Field(..., min_items=1, description="List of notes to process")
    batch_size: int = Field(10, ge=1, le=100, description="Batch size for processing")

class BulkProcessingResponse(BaseModel):
    """
    Response for bulk processing
    """
    total_processed: int
    successful: int
    failed: int
    results: List[ProcessingResult]
    total_time_ms: int