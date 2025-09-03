import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from decimal import Decimal

from models.client import Client
from models.portfolio import Portfolio, Holding
from models.event import Event, Proposal
from models.external_data import MarketData, WeatherData, NewsData


class TestClientModel:
    """Test cases for Client model"""
    
    def test_client_creation(self, sample_client_data):
        """Test client creation with valid data"""
        client = Client(**sample_client_data)
        
        assert client.client_id == sample_client_data['client_id']
        assert client.first_name == sample_client_data['first_name']
        assert client.last_name == sample_client_data['last_name']
        assert client.email == sample_client_data['email']
        assert client.net_worth == sample_client_data['net_worth']
        assert client.risk_tolerance == sample_client_data['risk_tolerance']
    
    def test_client_full_name(self, sample_client_data):
        """Test client full name property"""
        client = Client(**sample_client_data)
        expected_name = f"{sample_client_data['first_name']} {sample_client_data['last_name']}"
        assert client.full_name == expected_name
    
    def test_client_age_calculation(self, sample_client_data):
        """Test client age calculation"""
        client = Client(**sample_client_data)
        expected_age = datetime.now().year - sample_client_data['date_of_birth'].year
        
        # Adjust for birthday not yet occurred this year
        if datetime.now().month < sample_client_data['date_of_birth'].month:
            expected_age -= 1
        elif (datetime.now().month == sample_client_data['date_of_birth'].month and 
              datetime.now().day < sample_client_data['date_of_birth'].day):
            expected_age -= 1
            
        assert client.age == expected_age
    
    def test_client_risk_score_calculation(self, sample_client_data):
        """Test client risk score calculation"""
        client = Client(**sample_client_data)
        risk_score = client.calculate_risk_score()
        
        assert isinstance(risk_score, float)
        assert 0 <= risk_score <= 1
    
    def test_client_wealth_tier(self, sample_client_data):
        """Test client wealth tier classification"""
        client = Client(**sample_client_data)
        wealth_tier = client.get_wealth_tier()
        
        assert wealth_tier in ['mass_market', 'affluent', 'high_net_worth', 'ultra_high_net_worth']
        
        # Test specific thresholds
        if client.net_worth >= 30000000:
            assert wealth_tier == 'ultra_high_net_worth'
        elif client.net_worth >= 1000000:
            assert wealth_tier == 'high_net_worth'
        elif client.net_worth >= 250000:
            assert wealth_tier == 'affluent'
        else:
            assert wealth_tier == 'mass_market'
    
    def test_client_compliance_status(self, sample_client_data):
        """Test client compliance status validation"""
        client = Client(**sample_client_data)
        
        assert client.kyc_status in ['pending', 'in_progress', 'completed', 'expired', 'rejected']
        assert client.aml_status in ['pending', 'in_progress', 'cleared', 'flagged', 'rejected']
        assert client.compliance_status in ['compliant', 'non_compliant', 'pending_review', 'requires_documentation', 'escalated', 'approved_with_conditions']
    
    def test_client_to_dict(self, sample_client_data):
        """Test client to_dict method"""
        client = Client(**sample_client_data)
        client_dict = client.to_dict()
        
        assert isinstance(client_dict, dict)
        assert 'client_id' in client_dict
        assert 'full_name' in client_dict
        assert 'age' in client_dict
        assert 'wealth_tier' in client_dict


class TestPortfolioModel:
    """Test cases for Portfolio model"""
    
    def test_portfolio_creation(self, sample_portfolio_data):
        """Test portfolio creation with valid data"""
        portfolio = Portfolio(**sample_portfolio_data)
        
        assert portfolio.portfolio_id == sample_portfolio_data['portfolio_id']
        assert portfolio.client_id == sample_portfolio_data['client_id']
        assert portfolio.portfolio_name == sample_portfolio_data['portfolio_name']
        assert portfolio.total_value == sample_portfolio_data['total_value']
        assert portfolio.base_currency == sample_portfolio_data['base_currency']
    
    def test_portfolio_allocation_validation(self, sample_portfolio_data):
        """Test portfolio allocation validation"""
        portfolio = Portfolio(**sample_portfolio_data)
        
        # Test that allocation sums to approximately 1.0
        allocation_sum = sum(portfolio.target_allocation.values())
        assert abs(allocation_sum - 1.0) < 0.01
    
    def test_portfolio_performance_metrics(self, sample_portfolio_data):
        """Test portfolio performance metrics"""
        portfolio = Portfolio(**sample_portfolio_data)
        
        assert 'ytd_return' in portfolio.performance_metrics
        assert 'total_return' in portfolio.performance_metrics
        assert 'volatility' in portfolio.performance_metrics
        assert 'sharpe_ratio' in portfolio.performance_metrics
        assert 'max_drawdown' in portfolio.performance_metrics
        
        # Validate metric ranges
        assert -1 <= portfolio.performance_metrics['max_drawdown'] <= 0
        assert portfolio.performance_metrics['volatility'] >= 0
    
    def test_portfolio_risk_assessment(self, sample_portfolio_data):
        """Test portfolio risk assessment"""
        portfolio = Portfolio(**sample_portfolio_data)
        risk_level = portfolio.assess_risk_level()
        
        assert risk_level in ['very_low', 'low', 'moderate', 'high', 'very_high']
    
    def test_portfolio_rebalancing_needed(self, sample_portfolio_data, sample_holdings_data):
        """Test portfolio rebalancing detection"""
        portfolio = Portfolio(**sample_portfolio_data)
        
        # Mock current allocation that differs from target
        current_allocation = {'equities': 0.7, 'bonds': 0.2, 'alternatives': 0.1}
        needs_rebalancing = portfolio.needs_rebalancing(current_allocation, threshold=0.05)
        
        assert isinstance(needs_rebalancing, bool)
    
    def test_portfolio_to_dict(self, sample_portfolio_data):
        """Test portfolio to_dict method"""
        portfolio = Portfolio(**sample_portfolio_data)
        portfolio_dict = portfolio.to_dict()
        
        assert isinstance(portfolio_dict, dict)
        assert 'portfolio_id' in portfolio_dict
        assert 'total_value' in portfolio_dict
        assert 'performance_metrics' in portfolio_dict


class TestHoldingModel:
    """Test cases for Holding model"""
    
    def test_holding_creation(self, sample_holdings_data):
        """Test holding creation with valid data"""
        holding_data = sample_holdings_data[0]
        holding = Holding(**holding_data)
        
        assert holding.holding_id == holding_data['holding_id']
        assert holding.symbol == holding_data['symbol']
        assert holding.quantity == holding_data['quantity']
        assert holding.unit_price == holding_data['unit_price']
        assert holding.market_value == holding_data['market_value']
    
    def test_holding_pnl_calculation(self, sample_holdings_data):
        """Test holding P&L calculation"""
        holding_data = sample_holdings_data[0]
        holding = Holding(**holding_data)
        
        expected_pnl = holding.market_value - holding.cost_basis
        assert abs(holding.unrealized_pnl - expected_pnl) < 0.01
    
    def test_holding_return_calculation(self, sample_holdings_data):
        """Test holding return calculation"""
        holding_data = sample_holdings_data[0]
        holding = Holding(**holding_data)
        
        expected_return = (holding.market_value - holding.cost_basis) / holding.cost_basis
        calculated_return = holding.calculate_return()
        
        assert abs(calculated_return - expected_return) < 0.001
    
    def test_holding_weight_validation(self, sample_holdings_data):
        """Test holding weight validation"""
        holding_data = sample_holdings_data[0]
        holding = Holding(**holding_data)
        
        assert 0 <= holding.weight <= 1
    
    def test_holding_to_dict(self, sample_holdings_data):
        """Test holding to_dict method"""
        holding_data = sample_holdings_data[0]
        holding = Holding(**holding_data)
        holding_dict = holding.to_dict()
        
        assert isinstance(holding_dict, dict)
        assert 'holding_id' in holding_dict
        assert 'symbol' in holding_dict
        assert 'market_value' in holding_dict


class TestEventModel:
    """Test cases for Event model"""
    
    def test_event_creation(self, sample_event_data):
        """Test event creation with valid data"""
        event = Event(**sample_event_data)
        
        assert event.event_id == sample_event_data['event_id']
        assert event.event_type == sample_event_data['event_type']
        assert event.severity == sample_event_data['severity']
        assert event.confidence_score == sample_event_data['confidence_score']
        assert event.impact_score == sample_event_data['impact_score']
    
    def test_event_severity_validation(self, sample_event_data):
        """Test event severity validation"""
        event = Event(**sample_event_data)
        
        assert event.severity in ['low', 'medium', 'high', 'critical', 'emergency']
    
    def test_event_score_validation(self, sample_event_data):
        """Test event score validation"""
        event = Event(**sample_event_data)
        
        assert 0 <= event.confidence_score <= 1
        assert 0 <= event.impact_score <= 1
    
    def test_event_type_validation(self, sample_event_data):
        """Test event type validation"""
        event = Event(**sample_event_data)
        
        valid_types = ['market_event', 'news_event', 'economic_event', 'corporate_event', 
                      'life_event', 'portfolio_event', 'compliance_event', 'system_event',
                      'weather_event', 'satellite_event']
        assert event.event_type in valid_types
    
    def test_event_priority_calculation(self, sample_event_data):
        """Test event priority calculation"""
        event = Event(**sample_event_data)
        priority = event.calculate_priority()
        
        assert isinstance(priority, float)
        assert priority >= 0
    
    def test_event_to_dict(self, sample_event_data):
        """Test event to_dict method"""
        event = Event(**sample_event_data)
        event_dict = event.to_dict()
        
        assert isinstance(event_dict, dict)
        assert 'event_id' in event_dict
        assert 'event_type' in event_dict
        assert 'severity' in event_dict


class TestProposalModel:
    """Test cases for Proposal model"""
    
    def test_proposal_creation(self):
        """Test proposal creation with valid data"""
        proposal_data = {
            'proposal_id': 'prop_001',
            'event_id': 'event_001',
            'client_id': 'client_001',
            'portfolio_id': 'portfolio_001',
            'proposal_type': 'rebalancing',
            'status': 'pending',
            'priority': 'medium',
            'recommendations': [
                {
                    'action': 'sell',
                    'symbol': 'AAPL',
                    'quantity': 100,
                    'rationale': 'Reduce technology exposure'
                }
            ],
            'expected_impact': {
                'risk_reduction': 0.05,
                'return_impact': 0.02
            },
            'confidence_score': 0.8,
            'created_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(days=7)
        }
        
        proposal = Proposal(**proposal_data)
        
        assert proposal.proposal_id == proposal_data['proposal_id']
        assert proposal.proposal_type == proposal_data['proposal_type']
        assert proposal.status == proposal_data['status']
        assert proposal.confidence_score == proposal_data['confidence_score']
    
    def test_proposal_status_validation(self):
        """Test proposal status validation"""
        proposal_data = {
            'proposal_id': 'prop_001',
            'event_id': 'event_001',
            'client_id': 'client_001',
            'proposal_type': 'rebalancing',
            'status': 'pending',
            'recommendations': [],
            'confidence_score': 0.8
        }
        
        proposal = Proposal(**proposal_data)
        valid_statuses = ['pending', 'approved', 'rejected', 'executed', 'expired', 'cancelled']
        assert proposal.status in valid_statuses
    
    def test_proposal_expiry_check(self):
        """Test proposal expiry check"""
        proposal_data = {
            'proposal_id': 'prop_001',
            'event_id': 'event_001',
            'client_id': 'client_001',
            'proposal_type': 'rebalancing',
            'status': 'pending',
            'recommendations': [],
            'confidence_score': 0.8,
            'expires_at': datetime.now() - timedelta(days=1)  # Expired
        }
        
        proposal = Proposal(**proposal_data)
        assert proposal.is_expired()


class TestExternalDataModels:
    """Test cases for External Data models"""
    
    def test_market_data_creation(self, sample_market_data):
        """Test market data creation"""
        market_data = MarketData(
            symbol='AAPL',
            timestamp=datetime.now(),
            price=150.0,
            volume=1000000,
            high=152.0,
            low=148.0,
            open=149.0,
            close=150.0,
            source='alpha_vantage'
        )
        
        assert market_data.symbol == 'AAPL'
        assert market_data.price == 150.0
        assert market_data.volume == 1000000
    
    def test_weather_data_creation(self, sample_weather_data):
        """Test weather data creation"""
        weather_data = WeatherData(
            location=sample_weather_data['location'],
            timestamp=sample_weather_data['timestamp'],
            temperature=sample_weather_data['temperature']['current'],
            precipitation=sample_weather_data['precipitation']['current_24h'],
            wind_speed=sample_weather_data['wind']['speed'],
            conditions='partly_cloudy',
            source='openweathermap'
        )
        
        assert weather_data.location == sample_weather_data['location']
        assert weather_data.temperature == sample_weather_data['temperature']['current']
    
    def test_news_data_creation(self, sample_news_data):
        """Test news data creation"""
        news_item = sample_news_data[0]
        news_data = NewsData(
            title=news_item['title'],
            description=news_item['description'],
            source=news_item['source'],
            published_at=datetime.fromisoformat(news_item['published_at'].replace('Z', '+00:00')),
            url=news_item['url'],
            sentiment=news_item['sentiment'],
            relevance=news_item['relevance'],
            category=news_item['category']
        )
        
        assert news_data.title == news_item['title']
        assert news_data.sentiment == news_item['sentiment']
        assert news_data.relevance == news_item['relevance']
    
    def test_news_sentiment_validation(self, sample_news_data):
        """Test news sentiment validation"""
        news_item = sample_news_data[0]
        news_data = NewsData(
            title=news_item['title'],
            description=news_item['description'],
            source=news_item['source'],
            published_at=datetime.fromisoformat(news_item['published_at'].replace('Z', '+00:00')),
            url=news_item['url'],
            sentiment=news_item['sentiment'],
            relevance=news_item['relevance'],
            category=news_item['category']
        )
        
        assert -1 <= news_data.sentiment <= 1
        assert 0 <= news_data.relevance <= 1


class TestModelIntegration:
    """Test cases for model integration"""
    
    def test_client_portfolio_relationship(self, sample_client_data, sample_portfolio_data):
        """Test client-portfolio relationship"""
        client = Client(**sample_client_data)
        portfolio = Portfolio(**sample_portfolio_data)
        
        assert portfolio.client_id == client.client_id
    
    def test_portfolio_holdings_relationship(self, sample_portfolio_data, sample_holdings_data):
        """Test portfolio-holdings relationship"""
        portfolio = Portfolio(**sample_portfolio_data)
        holdings = [Holding(**holding_data) for holding_data in sample_holdings_data]
        
        for holding in holdings:
            assert holding.portfolio_id == portfolio.portfolio_id
        
        # Test total value consistency
        total_holdings_value = sum(holding.market_value for holding in holdings)
        expected_total = portfolio.total_value - portfolio.cash_balance
        
        # Allow for some difference due to other holdings not in sample
        assert total_holdings_value <= portfolio.total_value
    
    def test_event_proposal_relationship(self, sample_event_data):
        """Test event-proposal relationship"""
        event = Event(**sample_event_data)
        
        proposal_data = {
            'proposal_id': 'prop_001',
            'event_id': event.event_id,
            'client_id': 'client_001',
            'portfolio_id': 'portfolio_001',
            'proposal_type': 'rebalancing',
            'status': 'pending',
            'recommendations': [],
            'confidence_score': 0.8
        }
        
        proposal = Proposal(**proposal_data)
        assert proposal.event_id == event.event_id

