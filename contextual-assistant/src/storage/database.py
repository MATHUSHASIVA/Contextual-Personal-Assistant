"""
Database schema and models for the Contextual Personal Assistant
"""

# Standard library imports
import json
from datetime import datetime

# Third-party imports
from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Table,
    Text,
    create_engine,
    event,
    inspect
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import object_session, relationship, sessionmaker

Base = declarative_base()


# Association tables for many-to-many relationships
card_keywords = Table(
    'card_keywords',
    Base.metadata,
    Column('card_id', Integer, ForeignKey('cards.id')),
    Column('keyword_id', Integer, ForeignKey('keywords.id'))
)

envelope_keywords = Table(
    'envelope_keywords',
    Base.metadata,
    Column('envelope_id', Integer, ForeignKey('envelopes.id')),
    Column('keyword_id', Integer, ForeignKey('keywords.id'))
)


class Card(Base):
    """
    Core data structure representing a processed note
    """
    __tablename__ = 'cards'
    
    id = Column(Integer, primary_key=True)
    content = Column(Text, nullable=False)  # Original raw text
    description = Column(Text, nullable=False)  # Processed description
    card_type = Column(String(20), nullable=False)  # Task, Reminder, Idea
    status = Column(String(20), default='pending')  # pending, in_progress, completed, cancelled
    
    # Extracted entities
    assignee = Column(String(100))  # Person or team
    due_date = Column(DateTime)  # Extracted or inferred date
    location = Column(String(200))  # Extracted location if any
    
    # Relationships
    envelope_id = Column(Integer, ForeignKey('envelopes.id'))
    envelope = relationship("Envelope", back_populates="cards")
    keywords = relationship("Keyword", secondary=card_keywords, back_populates="cards")
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # JSON field for additional extracted entities
    additional_entities = Column(Text)  # JSON string for flexible entity storage
    
    def get_additional_entities(self):
        """Parse additional entities from JSON"""
        return json.loads(self.additional_entities) if self.additional_entities else {}
    
    def set_additional_entities(self, entities_dict):
        """Store additional entities as JSON"""
        self.additional_entities = json.dumps(entities_dict)


class Envelope(Base):
    """
    High-level groupings of related Cards (projects, companies, people, themes)
    """
    __tablename__ = 'envelopes'
    
    id = Column(Integer, primary_key=True)
    name = Column(String(200), nullable=False, unique=True)
    description = Column(Text)
    envelope_type = Column(String(50))  # project, company, person, theme, general
    
    # Relationships
    cards = relationship("Card", back_populates="envelope")
    keywords = relationship("Keyword", secondary="envelope_keywords", back_populates="envelopes")
    
    # Metadata
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    card_count = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    
    # Semantic vector for envelope matching (stored as JSON)
    semantic_vector = Column(Text)  # JSON array of vector values
    
    def get_semantic_vector(self):
        """Parse semantic vector from JSON"""
        return json.loads(self.semantic_vector) if self.semantic_vector else []
    
    def set_semantic_vector(self, vector):
        """Store semantic vector as JSON"""
        self.semantic_vector = json.dumps(vector)

    def update_card_count(self):
        """Update the card count for this envelope"""
        session = object_session(self)
        if session:
            self.card_count = session.query(Card).filter(Card.envelope_id == self.id).count()


class Keyword(Base):
    """
    Keywords extracted from cards for search and organization
    """
    __tablename__ = 'keywords'
    
    id = Column(Integer, primary_key=True)
    word = Column(String(100), nullable=False, unique=True)
    frequency = Column(Integer, default=1)
    importance_score = Column(Float, default=0.0)  # TF-IDF or similar
    
    # Relationships
    cards = relationship("Card", secondary=card_keywords, back_populates="keywords")
    envelopes = relationship("Envelope", secondary=envelope_keywords, back_populates="keywords")
    created_at = Column(DateTime, default=datetime.utcnow)


class UserContext(Base):
    """
    Dynamic user context including current projects, companies, people, themes
    """
    __tablename__ = 'user_context'
    
    id = Column(Integer, primary_key=True)
    context_type = Column(String(50), nullable=False)  # project, company, person, theme, preference
    name = Column(String(200), nullable=False)
    description = Column(Text)
    relevance_score = Column(Float, default=1.0)  # How relevant/active this context is
    
    # Relationships and references
    related_envelope_id = Column(Integer, ForeignKey('envelopes.id'))
    related_envelope = relationship("Envelope")
    
    # Context-specific data (JSON)
    context_data = Column(Text)  # JSON for flexible context storage
    
    # Temporal information
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    last_referenced = Column(DateTime, default=datetime.utcnow)
    is_active = Column(Boolean, default=True)
    
    def get_context_data(self):
        """Parse context data from JSON"""
        return json.loads(self.context_data) if self.context_data else {}
    
    def set_context_data(self, data_dict):
        """Store context data as JSON"""
        self.context_data = json.dumps(data_dict)


class ProcessingLog(Base):
    """
    Log of processing activities for debugging and analytics
    """
    __tablename__ = 'processing_logs'
    
    id = Column(Integer, primary_key=True)
    input_text = Column(Text, nullable=False)
    processing_stage = Column(String(50))  # classification, ner, envelope_assignment, etc.
    result = Column(Text)  # JSON of processing result
    processing_time_ms = Column(Integer)
    
    # Related card if processing was successful
    card_id = Column(Integer, ForeignKey('cards.id'))
    card = relationship("Card")
    
    created_at = Column(DateTime, default=datetime.utcnow)
    success = Column(Boolean, default=True)
    error_message = Column(Text)


class ThinkingInsight(Base):
    """
    Insights and suggestions generated by the Thinking Agent
    """
    __tablename__ = 'thinking_insights'
    
    id = Column(Integer, primary_key=True)
    insight_type = Column(String(50), nullable=False)  # next_step, recommendation, conflict, optimization
    title = Column(String(200), nullable=False)
    description = Column(Text, nullable=False)
    
    # References to relevant data
    related_card_ids = Column(Text)  # JSON array of card IDs
    related_envelope_ids = Column(Text)  # JSON array of envelope IDs
    
    # Action information
    is_actionable = Column(Boolean, default=False)
    suggested_action = Column(Text)
    
    # Status tracking
    is_dismissed = Column(Boolean, default=False)
    is_acted_upon = Column(Boolean, default=False)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime)  # Some insights may be time-sensitive
    
    def get_related_card_ids(self):
        """Parse related card IDs from JSON"""
        return json.loads(self.related_card_ids) if self.related_card_ids else []
    
    def set_related_card_ids(self, card_ids):
        """Store related card IDs as JSON"""
        self.related_card_ids = json.dumps(card_ids)
    
    def get_related_envelope_ids(self):
        """Parse related envelope IDs from JSON"""
        return json.loads(self.related_envelope_ids) if self.related_envelope_ids else []
    
    def set_related_envelope_ids(self, envelope_ids):
        """Store related envelope IDs as JSON"""
        self.related_envelope_ids = json.dumps(envelope_ids)


class DatabaseManager:
    """
    Manages database connections and operations
    """
    
    def __init__(self, database_url="sqlite:///./data/assistant.db"):
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=self.engine)
        self.create_tables()
        
        # Set up event listeners for card count updates
        event.listen(Card, 'after_insert', self._update_envelope_card_count)
        event.listen(Card, 'after_delete', self._update_envelope_card_count)
        event.listen(Card, 'after_update', self._update_envelope_card_count_on_move)
    
    def create_tables(self):
        """Create all database tables"""
        Base.metadata.create_all(bind=self.engine)
    
    def get_session(self):
        """Get a database session"""
        return self.SessionLocal()
    
    def close_session(self, session):
        """Close a database session"""
        session.close()
    
    def fix_card_counts(self):
        """Fix all envelope card counts"""
        session = self.get_session()
        try:
            envelopes = session.query(Envelope).all()
            for envelope in envelopes:
                envelope.update_card_count()
            session.commit()
        finally:
            session.close()
    
    @staticmethod
    def _update_envelope_card_count(mapper, connection, target):
        """Update envelope card count after card insert/delete"""
        if target.envelope_id:
            # Use raw SQL to avoid triggering additional SQLAlchemy events
            from sqlalchemy import text
            count_query = text("""
                UPDATE envelopes 
                SET card_count = (
                    SELECT COUNT(*) 
                    FROM cards 
                    WHERE envelope_id = :envelope_id
                ) 
                WHERE id = :envelope_id
            """)
            connection.execute(count_query, {"envelope_id": target.envelope_id})
    
    @staticmethod
    def _update_envelope_card_count_on_move(mapper, connection, target):
        """Update envelope card counts when a card moves between envelopes"""
        # Get the state change history
        state = inspect(target)
        # Check if envelope_id has changed
        if hasattr(state.attrs, 'envelope_id') and state.attrs.envelope_id.history.has_changes():
            # Get old and new envelope IDs
            old_envelope_id = state.attrs.envelope_id.history.deleted[0] if state.attrs.envelope_id.history.deleted else None
            new_envelope_id = target.envelope_id
            
            # Use raw SQL to avoid triggering additional SQLAlchemy events
            from sqlalchemy import text
            count_query = text("""
                UPDATE envelopes 
                SET card_count = (
                    SELECT COUNT(*) 
                    FROM cards 
                    WHERE envelope_id = :envelope_id
                ) 
                WHERE id = :envelope_id
            """)
            
            # Update both old and new envelopes if they exist
            if old_envelope_id:
                connection.execute(count_query, {"envelope_id": old_envelope_id})
            
            if new_envelope_id:
                connection.execute(count_query, {"envelope_id": new_envelope_id})