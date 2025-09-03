from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, String, Integer, DateTime, Boolean, Numeric, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum

db = SQLAlchemy()

class EventType(str, Enum):
    LIFE_EVENT = "life_event"
    MARKET_EVENT = "market_event"
    NEWS_EVENT = "news_event"
    WEATHER_EVENT = "weather_event"
    ECONOMIC_EVENT = "economic_event"
    REGULATORY_EVENT = "regulatory_event"

class EventSource(str, Enum):
    CRM = "CRM"
    NEWS_API = "NewsAPI"
    SOCIAL_MEDIA = "Social Media"
    WEATHER_API = "Weather API"
    SATELLITE_DATA = "Satellite Data"
    ECONOMIC_DATA = "Economic Data"
    MANUAL = "Manual"
    ORACLE = "Oracle"

class EventStatus(str, Enum):
    NEW = "NEW"
    ENRICHED = "ENRICHED"
    PROPOSED = "PROPOSED"
    CHECKED = "CHECKED"
    APPROVED = "APPROVED"
    EXECUTED = "EXECUTED"
    NARRATED = "NARRATED"
    CLOSED = "CLOSED"
    FAILED = "FAILED"
    ESCALATED = "ESCALATED"

class EventSeverity(str, Enum):
    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"

class EventCategory(str, Enum):
    # Life Events
    MARRIAGE = "Marriage"
    DIVORCE = "Divorce"
    BIRTH = "Birth"
    DEATH = "Death"
    RETIREMENT = "Retirement"
    JOB_CHANGE = "Job Change"
    INHERITANCE = "Inheritance"
    MAJOR_PURCHASE = "Major Purchase"
    
    # Market Events
    EARNINGS_ANNOUNCEMENT = "Earnings Announcement"
    DIVIDEND_ANNOUNCEMENT = "Dividend Announcement"
    MERGER_ACQUISITION = "Merger & Acquisition"
    STOCK_SPLIT = "Stock Split"
    IPO = "IPO"
    
    # Economic Events
    INTEREST_RATE_CHANGE = "Interest Rate Change"
    INFLATION_DATA = "Inflation Data"
    GDP_RELEASE = "GDP Release"
    EMPLOYMENT_DATA = "Employment Data"
    
    # Weather/Environmental Events
    NATURAL_DISASTER = "Natural Disaster"
    CLIMATE_CHANGE = "Climate Change"
    CROP_CONDITIONS = "Crop Conditions"
    ENERGY_DISRUPTION = "Energy Disruption"

# SQLAlchemy Models
class Event(db.Model):
    __tablename__ = 'events'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    client_id = db.Column(db.String(36), db.ForeignKey('clients.id'), nullable=True)  # Can be null for market events
    event_type = db.Column(db.Enum(EventType), nullable=False)
    event_source = db.Column(db.Enum(EventSource), nullable=False)
    event_category = db.Column(db.Enum(EventCategory))
    
    # Event Details
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text)
    severity = db.Column(db.Enum(EventSeverity), default=EventSeverity.MEDIUM)
    confidence = db.Column(db.Numeric(3, 2))  # 0.00 to 1.00
    
    # Location Information (for weather/geographic events)
    latitude = db.Column(db.Numeric(10, 8))
    longitude = db.Column(db.Numeric(11, 8))
    location_name = db.Column(db.String(100))
    
    # Related Financial Instruments
    related_symbols = db.Column(JSON)  # Array of stock symbols
    affected_sectors = db.Column(JSON)  # Array of sector names
    
    # Sentiment Analysis
    sentiment_score = db.Column(db.Numeric(3, 2))  # -1.00 to 1.00
    sentiment_label = db.Column(db.String(20))  # "positive", "negative", "neutral"
    
    # Event Data (flexible JSON storage)
    event_data = db.Column(JSON)
    
    # Processing Status
    status = db.Column(db.Enum(EventStatus), default=EventStatus.NEW)
    processing_results = db.Column(JSON)  # Results from each processing stage
    
    # Metadata
    event_timestamp = db.Column(db.DateTime)  # When the event actually occurred
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    processed_at = db.Column(db.DateTime)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    proposals = db.relationship('Proposal', backref='event', lazy=True)
    
    def __repr__(self):
        return f'<Event {self.title}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'client_id': self.client_id,
            'event_type': self.event_type.value if self.event_type else None,
            'event_source': self.event_source.value if self.event_source else None,
            'event_category': self.event_category.value if self.event_category else None,
            'title': self.title,
            'description': self.description,
            'severity': self.severity.value if self.severity else None,
            'confidence': float(self.confidence) if self.confidence else None,
            'location': {
                'latitude': float(self.latitude) if self.latitude else None,
                'longitude': float(self.longitude) if self.longitude else None,
                'name': self.location_name
            } if self.latitude and self.longitude else None,
            'related_symbols': self.related_symbols,
            'affected_sectors': self.affected_sectors,
            'sentiment': {
                'score': float(self.sentiment_score) if self.sentiment_score else None,
                'label': self.sentiment_label
            },
            'event_data': self.event_data,
            'status': self.status.value if self.status else None,
            'processing_results': self.processing_results,
            'event_timestamp': self.event_timestamp.isoformat() if self.event_timestamp else None,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'processed_at': self.processed_at.isoformat() if self.processed_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class Proposal(db.Model):
    __tablename__ = 'proposals'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    event_id = db.Column(db.String(36), db.ForeignKey('events.id'), nullable=False)
    portfolio_id = db.Column(db.String(36), db.ForeignKey('portfolios.id'), nullable=False)
    
    # Proposal Details
    proposal_type = db.Column(db.String(50))  # "rebalance", "new_investment", "divestment"
    objective = db.Column(db.String(100))  # "SharpeMax", "DrawdownMin", "ESG", "TaxAware"
    
    # Proposed Changes
    proposed_trades = db.Column(JSON)  # Array of trade objects
    expected_impact = db.Column(JSON)  # Expected performance impact
    
    # Risk Assessment
    risk_assessment = db.Column(JSON)
    compliance_status = db.Column(db.String(20), default='Pending')
    
    # Approval Workflow
    approval_required = db.Column(db.Boolean, default=True)
    approved_by = db.Column(db.String(100))
    approval_timestamp = db.Column(db.DateTime)
    rejection_reason = db.Column(db.Text)
    
    # Execution Status
    execution_status = db.Column(db.String(20), default='Pending')
    executed_at = db.Column(db.DateTime)
    execution_results = db.Column(JSON)
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    def __repr__(self):
        return f'<Proposal {self.proposal_type} for Event {self.event_id}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'event_id': self.event_id,
            'portfolio_id': self.portfolio_id,
            'proposal_type': self.proposal_type,
            'objective': self.objective,
            'proposed_trades': self.proposed_trades,
            'expected_impact': self.expected_impact,
            'risk_assessment': self.risk_assessment,
            'compliance_status': self.compliance_status,
            'approval': {
                'required': self.approval_required,
                'approved_by': self.approved_by,
                'timestamp': self.approval_timestamp.isoformat() if self.approval_timestamp else None,
                'rejection_reason': self.rejection_reason
            },
            'execution': {
                'status': self.execution_status,
                'executed_at': self.executed_at.isoformat() if self.executed_at else None,
                'results': self.execution_results
            },
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

# Pydantic Models
class LocationData(BaseModel):
    latitude: Optional[float] = Field(None, ge=-90, le=90)
    longitude: Optional[float] = Field(None, ge=-180, le=180)
    name: Optional[str] = Field(None, max_length=100)

class SentimentData(BaseModel):
    score: Optional[float] = Field(None, ge=-1, le=1)
    label: Optional[str] = Field(None, pattern="^(positive|negative|neutral)$")

class ProcessingResults(BaseModel):
    proposal_generated: bool = False
    compliance_checked: bool = False
    client_notified: bool = False
    execution_completed: bool = False
    narrative_generated: bool = False

class EventBase(BaseModel):
    event_type: EventType
    event_source: EventSource
    event_category: Optional[EventCategory] = None
    title: str = Field(..., min_length=1, max_length=200)
    description: Optional[str] = None
    severity: Optional[EventSeverity] = EventSeverity.MEDIUM
    confidence: Optional[float] = Field(None, ge=0, le=1)
    location: Optional[LocationData] = None
    related_symbols: Optional[List[str]] = []
    affected_sectors: Optional[List[str]] = []
    sentiment: Optional[SentimentData] = None
    event_data: Optional[Dict[str, Any]] = {}
    event_timestamp: Optional[str] = None  # ISO datetime string

class EventCreate(EventBase):
    client_id: Optional[str] = None

class EventUpdate(BaseModel):
    event_category: Optional[EventCategory] = None
    title: Optional[str] = Field(None, min_length=1, max_length=200)
    description: Optional[str] = None
    severity: Optional[EventSeverity] = None
    confidence: Optional[float] = Field(None, ge=0, le=1)
    location: Optional[LocationData] = None
    related_symbols: Optional[List[str]] = None
    affected_sectors: Optional[List[str]] = None
    sentiment: Optional[SentimentData] = None
    event_data: Optional[Dict[str, Any]] = None
    status: Optional[EventStatus] = None
    processing_results: Optional[ProcessingResults] = None

class EventResponse(EventBase):
    id: str
    client_id: Optional[str] = None
    status: EventStatus
    processing_results: Optional[ProcessingResults] = None
    created_at: str
    processed_at: Optional[str] = None
    updated_at: str
    
    class Config:
        from_attributes = True

class TradeInstruction(BaseModel):
    symbol: str
    action: str  # "buy", "sell"
    quantity: float
    order_type: str  # "market", "limit"
    limit_price: Optional[float] = None
    time_in_force: str = "DAY"

class ExpectedImpact(BaseModel):
    expected_return: Optional[float] = None
    risk_change: Optional[float] = None
    sharpe_improvement: Optional[float] = None
    tax_impact: Optional[float] = None

class ApprovalInfo(BaseModel):
    required: bool = True
    approved_by: Optional[str] = None
    timestamp: Optional[str] = None
    rejection_reason: Optional[str] = None

class ExecutionInfo(BaseModel):
    status: str = "Pending"
    executed_at: Optional[str] = None
    results: Optional[Dict[str, Any]] = None

class ProposalBase(BaseModel):
    proposal_type: str = Field(..., max_length=50)
    objective: str = Field(..., max_length=100)
    proposed_trades: List[TradeInstruction]
    expected_impact: Optional[ExpectedImpact] = None
    risk_assessment: Optional[Dict[str, Any]] = None

class ProposalCreate(ProposalBase):
    event_id: str
    portfolio_id: str

class ProposalUpdate(BaseModel):
    proposal_type: Optional[str] = Field(None, max_length=50)
    objective: Optional[str] = Field(None, max_length=100)
    proposed_trades: Optional[List[TradeInstruction]] = None
    expected_impact: Optional[ExpectedImpact] = None
    risk_assessment: Optional[Dict[str, Any]] = None
    compliance_status: Optional[str] = None
    execution_status: Optional[str] = None

class ProposalResponse(ProposalBase):
    id: str
    event_id: str
    portfolio_id: str
    compliance_status: str
    approval: ApprovalInfo
    execution: ExecutionInfo
    created_at: str
    updated_at: str
    
    class Config:
        from_attributes = True

class EventListResponse(BaseModel):
    events: List[EventResponse]
    total: int
    page: int
    per_page: int
    pages: int

class ProposalListResponse(BaseModel):
    proposals: List[ProposalResponse]
    total: int
    page: int
    per_page: int
    pages: int

class ProposalStatus(str, Enum):
    PENDING = "PENDING"
    APPROVED = "APPROVED"
    EXECUTED = "EXECUTED"
    PARTIALLY_EXECUTED = "PARTIALLY_EXECUTED"
    EXECUTION_FAILED = "EXECUTION_FAILED"

