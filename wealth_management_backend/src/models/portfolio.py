from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, String, Integer, DateTime, Boolean, Numeric, Text, JSON, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
import uuid
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum

db = SQLAlchemy()

class PortfolioStatus(str, Enum):
    ACTIVE = "Active"
    INACTIVE = "Inactive"
    CLOSED = "Closed"
    SUSPENDED = "Suspended"

class AssetClass(str, Enum):
    EQUITY = "Equity"
    FIXED_INCOME = "Fixed Income"
    COMMODITY = "Commodity"
    REAL_ESTATE = "Real Estate"
    CRYPTOCURRENCY = "Cryptocurrency"
    CASH = "Cash"
    ALTERNATIVE = "Alternative"

# SQLAlchemy Models
class Portfolio(db.Model):
    __tablename__ = 'portfolios'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    client_id = db.Column(db.String(36), db.ForeignKey('clients.id'), nullable=False)
    portfolio_name = db.Column(db.String(100), nullable=False)
    total_value = db.Column(db.Numeric(18, 2), default=0)
    cash_balance = db.Column(db.Numeric(18, 2), default=0)
    status = db.Column(db.Enum(PortfolioStatus), default=PortfolioStatus.ACTIVE)
    
    # Performance Metrics
    total_return = db.Column(db.Numeric(10, 4))  # Percentage
    annualized_return = db.Column(db.Numeric(10, 4))  # Percentage
    sharpe_ratio = db.Column(db.Numeric(10, 4))
    max_drawdown = db.Column(db.Numeric(10, 4))  # Percentage
    volatility = db.Column(db.Numeric(10, 4))  # Percentage
    
    # Risk Metrics
    beta = db.Column(db.Numeric(10, 4))
    var_95 = db.Column(db.Numeric(18, 2))  # Value at Risk 95%
    expected_shortfall = db.Column(db.Numeric(18, 2))
    concentration_risk = db.Column(db.Numeric(10, 4))  # Percentage
    
    # IPS (Investment Policy Statement) Constraints
    ips_constraints = db.Column(JSON)  # JSON object with constraints
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    holdings = db.relationship('Holding', backref='portfolio', lazy=True, cascade='all, delete-orphan')
    proposals = db.relationship('Proposal', backref='portfolio', lazy=True)
    
    def __repr__(self):
        return f'<Portfolio {self.portfolio_name}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'client_id': self.client_id,
            'portfolio_name': self.portfolio_name,
            'total_value': float(self.total_value) if self.total_value else 0,
            'cash_balance': float(self.cash_balance) if self.cash_balance else 0,
            'status': self.status.value if self.status else None,
            'performance': {
                'total_return': float(self.total_return) if self.total_return else None,
                'annualized_return': float(self.annualized_return) if self.annualized_return else None,
                'sharpe_ratio': float(self.sharpe_ratio) if self.sharpe_ratio else None,
                'max_drawdown': float(self.max_drawdown) if self.max_drawdown else None,
                'volatility': float(self.volatility) if self.volatility else None
            },
            'risk_metrics': {
                'beta': float(self.beta) if self.beta else None,
                'var_95': float(self.var_95) if self.var_95 else None,
                'expected_shortfall': float(self.expected_shortfall) if self.expected_shortfall else None,
                'concentration_risk': float(self.concentration_risk) if self.concentration_risk else None
            },
            'ips_constraints': self.ips_constraints,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None
        }

class Holding(db.Model):
    __tablename__ = 'holdings'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    portfolio_id = db.Column(db.String(36), db.ForeignKey('portfolios.id'), nullable=False)
    symbol = db.Column(db.String(10), nullable=False)
    company_name = db.Column(db.String(200))
    quantity = db.Column(db.Numeric(18, 6), nullable=False)
    average_cost = db.Column(db.Numeric(18, 4), nullable=False)
    current_price = db.Column(db.Numeric(18, 4))
    market_value = db.Column(db.Numeric(18, 2))
    unrealized_gain_loss = db.Column(db.Numeric(18, 2))
    
    # Classification
    asset_class = db.Column(db.Enum(AssetClass))
    sector = db.Column(db.String(50))
    industry = db.Column(db.String(100))
    country = db.Column(db.String(50))
    
    # ESG Scores
    esg_score = db.Column(db.Numeric(5, 2))  # 0-100 scale
    environmental_score = db.Column(db.Numeric(5, 2))
    social_score = db.Column(db.Numeric(5, 2))
    governance_score = db.Column(db.Numeric(5, 2))
    
    # Tax Information
    tax_lots = db.Column(JSON)  # Array of tax lot information
    
    # Metadata
    last_updated = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<Holding {self.symbol}: {self.quantity}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'portfolio_id': self.portfolio_id,
            'symbol': self.symbol,
            'company_name': self.company_name,
            'quantity': float(self.quantity) if self.quantity else 0,
            'average_cost': float(self.average_cost) if self.average_cost else 0,
            'current_price': float(self.current_price) if self.current_price else 0,
            'market_value': float(self.market_value) if self.market_value else 0,
            'unrealized_gain_loss': float(self.unrealized_gain_loss) if self.unrealized_gain_loss else 0,
            'asset_class': self.asset_class.value if self.asset_class else None,
            'sector': self.sector,
            'industry': self.industry,
            'country': self.country,
            'esg_scores': {
                'overall': float(self.esg_score) if self.esg_score else None,
                'environmental': float(self.environmental_score) if self.environmental_score else None,
                'social': float(self.social_score) if self.social_score else None,
                'governance': float(self.governance_score) if self.governance_score else None
            },
            'tax_lots': self.tax_lots,
            'last_updated': self.last_updated.isoformat() if self.last_updated else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

# Pydantic Models
class PerformanceMetrics(BaseModel):
    total_return: Optional[float] = None
    annualized_return: Optional[float] = None
    sharpe_ratio: Optional[float] = None
    max_drawdown: Optional[float] = None
    volatility: Optional[float] = None

class RiskMetrics(BaseModel):
    beta: Optional[float] = None
    var_95: Optional[float] = None
    expected_shortfall: Optional[float] = None
    concentration_risk: Optional[float] = None

class IPSConstraints(BaseModel):
    max_equity_allocation: Optional[float] = 100.0
    max_single_position: Optional[float] = 10.0
    min_cash_allocation: Optional[float] = 5.0
    max_sector_allocation: Optional[float] = 25.0
    restricted_securities: Optional[List[str]] = []
    esg_requirements: Optional[Dict[str, Any]] = {}

class ESGScores(BaseModel):
    overall: Optional[float] = None
    environmental: Optional[float] = None
    social: Optional[float] = None
    governance: Optional[float] = None

class TaxLot(BaseModel):
    purchase_date: str
    quantity: float
    cost_basis: float
    holding_period: str  # "short" or "long"

class HoldingBase(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=10)
    company_name: Optional[str] = Field(None, max_length=200)
    quantity: float = Field(..., gt=0)
    average_cost: float = Field(..., gt=0)
    current_price: Optional[float] = Field(None, gt=0)
    asset_class: Optional[AssetClass] = None
    sector: Optional[str] = Field(None, max_length=50)
    industry: Optional[str] = Field(None, max_length=100)
    country: Optional[str] = Field(None, max_length=50)
    esg_scores: Optional[ESGScores] = ESGScores()
    tax_lots: Optional[List[TaxLot]] = []

class HoldingCreate(HoldingBase):
    portfolio_id: str

class HoldingUpdate(BaseModel):
    quantity: Optional[float] = Field(None, gt=0)
    average_cost: Optional[float] = Field(None, gt=0)
    current_price: Optional[float] = Field(None, gt=0)
    asset_class: Optional[AssetClass] = None
    sector: Optional[str] = Field(None, max_length=50)
    industry: Optional[str] = Field(None, max_length=100)
    country: Optional[str] = Field(None, max_length=50)
    esg_scores: Optional[ESGScores] = None
    tax_lots: Optional[List[TaxLot]] = None

class HoldingResponse(HoldingBase):
    id: str
    portfolio_id: str
    market_value: float
    unrealized_gain_loss: float
    last_updated: str
    created_at: str
    
    class Config:
        from_attributes = True

class PortfolioBase(BaseModel):
    portfolio_name: str = Field(..., min_length=1, max_length=100)
    cash_balance: Optional[float] = Field(0, ge=0)
    ips_constraints: Optional[IPSConstraints] = IPSConstraints()

class PortfolioCreate(PortfolioBase):
    client_id: str

class PortfolioUpdate(BaseModel):
    portfolio_name: Optional[str] = Field(None, min_length=1, max_length=100)
    cash_balance: Optional[float] = Field(None, ge=0)
    status: Optional[PortfolioStatus] = None
    ips_constraints: Optional[IPSConstraints] = None

class PortfolioResponse(PortfolioBase):
    id: str
    client_id: str
    total_value: float
    status: PortfolioStatus
    performance: PerformanceMetrics
    risk_metrics: RiskMetrics
    holdings: List[HoldingResponse] = []
    created_at: str
    updated_at: str
    
    class Config:
        from_attributes = True

class PortfolioListResponse(BaseModel):
    portfolios: List[PortfolioResponse]
    total: int
    page: int
    per_page: int
    pages: int

