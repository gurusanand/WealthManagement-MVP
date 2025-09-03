from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, String, Integer, DateTime, Boolean, Numeric, Text, JSON
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.sql import func
from datetime import datetime
import uuid
from pydantic import BaseModel, Field, EmailStr
from typing import Optional, List, Dict, Any
from enum import Enum

db = SQLAlchemy()

class RiskTolerance(str, Enum):
    CONSERVATIVE = "Conservative"
    MODERATE = "Moderate"
    AGGRESSIVE = "Aggressive"

class ClientStatus(str, Enum):
    ACTIVE = "Active"
    INACTIVE = "Inactive"
    SUSPENDED = "Suspended"
    PENDING = "Pending"

# SQLAlchemy Model
class Client(db.Model):
    __tablename__ = 'clients'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    first_name = db.Column(db.String(100), nullable=False)
    last_name = db.Column(db.String(100), nullable=False)
    email = db.Column(db.String(255), unique=True, nullable=False)
    phone = db.Column(db.String(20))
    date_of_birth = db.Column(db.Date)
    risk_tolerance = db.Column(db.Enum(RiskTolerance), default=RiskTolerance.MODERATE)
    net_worth = db.Column(db.Numeric(18, 2))
    annual_income = db.Column(db.Numeric(18, 2))
    investment_objectives = db.Column(db.Text)
    kyc_status = db.Column(db.String(20), default='Pending')
    aml_status = db.Column(db.String(20), default='Pending')
    status = db.Column(db.Enum(ClientStatus), default=ClientStatus.ACTIVE)
    
    # ESG Preferences
    esg_environmental = db.Column(db.Boolean, default=False)
    esg_social = db.Column(db.Boolean, default=False)
    esg_governance = db.Column(db.Boolean, default=False)
    
    # Communication Preferences
    preferred_language = db.Column(db.String(10), default='en')
    preferred_currency = db.Column(db.String(3), default='USD')
    communication_channels = db.Column(JSON)  # ['email', 'sms', 'phone']
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)
    
    # Relationships
    portfolios = db.relationship('Portfolio', backref='client', lazy=True, cascade='all, delete-orphan')
    events = db.relationship('Event', backref='client', lazy=True)
    
    def __repr__(self):
        return f'<Client {self.first_name} {self.last_name}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'email': self.email,
            'phone': self.phone,
            'date_of_birth': self.date_of_birth.isoformat() if self.date_of_birth else None,
            'risk_tolerance': self.risk_tolerance.value if self.risk_tolerance else None,
            'net_worth': float(self.net_worth) if self.net_worth else None,
            'annual_income': float(self.annual_income) if self.annual_income else None,
            'investment_objectives': self.investment_objectives,
            'kyc_status': self.kyc_status,
            'aml_status': self.aml_status,
            'status': self.status.value if self.status else None,
            'esg_preferences': {
                'environmental': self.esg_environmental,
                'social': self.esg_social,
                'governance': self.esg_governance
            },
            'preferences': {
                'language': self.preferred_language,
                'currency': self.preferred_currency,
                'communication_channels': self.communication_channels
            },
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'is_active': self.is_active
        }

# Pydantic Models for API validation
class ESGPreferences(BaseModel):
    environmental: bool = False
    social: bool = False
    governance: bool = False

class ClientPreferences(BaseModel):
    language: str = "en"
    currency: str = "USD"
    communication_channels: List[str] = ["email"]

class ClientBase(BaseModel):
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    email: EmailStr
    phone: Optional[str] = Field(None, max_length=20)
    date_of_birth: Optional[str] = None  # ISO date string
    risk_tolerance: Optional[RiskTolerance] = RiskTolerance.MODERATE
    net_worth: Optional[float] = Field(None, ge=0)
    annual_income: Optional[float] = Field(None, ge=0)
    investment_objectives: Optional[str] = None
    esg_preferences: Optional[ESGPreferences] = ESGPreferences()
    preferences: Optional[ClientPreferences] = ClientPreferences()

class ClientCreate(ClientBase):
    pass

class ClientUpdate(BaseModel):
    first_name: Optional[str] = Field(None, min_length=1, max_length=100)
    last_name: Optional[str] = Field(None, min_length=1, max_length=100)
    email: Optional[EmailStr] = None
    phone: Optional[str] = Field(None, max_length=20)
    date_of_birth: Optional[str] = None
    risk_tolerance: Optional[RiskTolerance] = None
    net_worth: Optional[float] = Field(None, ge=0)
    annual_income: Optional[float] = Field(None, ge=0)
    investment_objectives: Optional[str] = None
    esg_preferences: Optional[ESGPreferences] = None
    preferences: Optional[ClientPreferences] = None

class ClientResponse(ClientBase):
    id: str
    kyc_status: str
    aml_status: str
    status: ClientStatus
    created_at: str
    updated_at: str
    is_active: bool
    
    class Config:
        from_attributes = True

class ClientListResponse(BaseModel):
    clients: List[ClientResponse]
    total: int
    page: int
    per_page: int
    pages: int

