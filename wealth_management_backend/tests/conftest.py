import pytest
import os
import sys
import tempfile
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from unittest.mock import Mock, MagicMock

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from main import create_app
from models.client import Client
from models.portfolio import Portfolio, Holding
from models.event import Event, Proposal
from models.external_data import MarketData, WeatherData, NewsData

@pytest.fixture
def app():
    """Create and configure a test Flask application"""
    app = create_app()
    app.config.update({
        "TESTING": True,
        "SQLALCHEMY_DATABASE_URI": "sqlite:///:memory:",
        "WTF_CSRF_ENABLED": False
    })
    
    with app.app_context():
        from models.user import db
        db.create_all()
        yield app
        db.drop_all()

@pytest.fixture
def client(app):
    """Create a test client for the Flask application"""
    return app.test_client()

@pytest.fixture
def runner(app):
    """Create a test runner for the Flask application"""
    return app.test_cli_runner()

@pytest.fixture
def sample_client_data():
    """Sample client data for testing"""
    return {
        'client_id': 'test_client_001',
        'first_name': 'John',
        'last_name': 'Doe',
        'email': 'john.doe@example.com',
        'phone': '+1-555-0123',
        'date_of_birth': datetime(1980, 5, 15),
        'nationality': 'US',
        'tax_residency': 'US',
        'net_worth': 1500000.0,
        'liquid_net_worth': 800000.0,
        'annual_income': 250000.0,
        'risk_tolerance': 'moderate',
        'investment_experience': 'moderate',
        'investment_objectives': ['capital_appreciation', 'income_generation'],
        'time_horizon': 15,
        'liquidity_needs': 'medium',
        'esg_preferences': {
            'importance': 'high',
            'exclusions': ['tobacco', 'weapons'],
            'preferences': ['renewable_energy', 'sustainable_agriculture']
        },
        'kyc_status': 'completed',
        'aml_status': 'cleared',
        'compliance_status': 'compliant'
    }

@pytest.fixture
def sample_portfolio_data():
    """Sample portfolio data for testing"""
    return {
        'portfolio_id': 'test_portfolio_001',
        'client_id': 'test_client_001',
        'portfolio_name': 'Conservative Growth Portfolio',
        'portfolio_type': 'discretionary',
        'base_currency': 'USD',
        'total_value': 1000000.0,
        'cash_balance': 50000.0,
        'target_allocation': {
            'equities': 0.6,
            'bonds': 0.3,
            'alternatives': 0.1
        },
        'risk_budget': 0.15,
        'benchmark': 'S&P 500',
        'inception_date': datetime(2020, 1, 1),
        'last_rebalance_date': datetime(2024, 1, 1),
        'performance_metrics': {
            'ytd_return': 0.085,
            'total_return': 0.124,
            'volatility': 0.142,
            'sharpe_ratio': 0.596,
            'max_drawdown': -0.089
        }
    }

@pytest.fixture
def sample_holdings_data():
    """Sample holdings data for testing"""
    return [
        {
            'holding_id': 'holding_001',
            'portfolio_id': 'test_portfolio_001',
            'symbol': 'AAPL',
            'asset_name': 'Apple Inc.',
            'asset_type': 'equity',
            'sector': 'technology',
            'quantity': 1000,
            'unit_price': 150.0,
            'market_value': 150000.0,
            'weight': 0.15,
            'cost_basis': 140000.0,
            'unrealized_pnl': 10000.0,
            'currency': 'USD'
        },
        {
            'holding_id': 'holding_002',
            'portfolio_id': 'test_portfolio_001',
            'symbol': 'MSFT',
            'asset_name': 'Microsoft Corporation',
            'asset_type': 'equity',
            'sector': 'technology',
            'quantity': 500,
            'unit_price': 300.0,
            'market_value': 150000.0,
            'weight': 0.15,
            'cost_basis': 145000.0,
            'unrealized_pnl': 5000.0,
            'currency': 'USD'
        }
    ]

@pytest.fixture
def sample_market_data():
    """Sample market data for testing"""
    dates = pd.date_range(start='2023-01-01', end='2024-01-01', freq='D')
    np.random.seed(42)
    
    # Generate realistic price data
    initial_price = 100
    returns = np.random.normal(0.0008, 0.02, len(dates))  # Daily returns
    prices = [initial_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    return pd.DataFrame({
        'date': dates,
        'price': prices,
        'volume': np.random.randint(1000000, 10000000, len(dates)),
        'high': [p * (1 + abs(np.random.normal(0, 0.01))) for p in prices],
        'low': [p * (1 - abs(np.random.normal(0, 0.01))) for p in prices],
        'open': [p * (1 + np.random.normal(0, 0.005)) for p in prices],
        'close': prices
    }).set_index('date')

@pytest.fixture
def sample_news_data():
    """Sample news data for testing"""
    return [
        {
            'id': 'news_001',
            'title': 'Federal Reserve Raises Interest Rates by 0.25%',
            'description': 'The Federal Reserve announced a quarter-point increase in interest rates to combat inflation.',
            'source': 'Reuters',
            'published_at': '2024-01-15T14:30:00Z',
            'url': 'https://example.com/news/fed-rates',
            'sentiment': -0.2,
            'relevance': 0.9,
            'category': 'monetary_policy'
        },
        {
            'id': 'news_002',
            'title': 'Tech Stocks Rally on Strong Earnings Reports',
            'description': 'Major technology companies reported better-than-expected quarterly earnings.',
            'source': 'Bloomberg',
            'published_at': '2024-01-16T09:15:00Z',
            'url': 'https://example.com/news/tech-rally',
            'sentiment': 0.7,
            'relevance': 0.8,
            'category': 'earnings'
        }
    ]

@pytest.fixture
def sample_weather_data():
    """Sample weather data for testing"""
    return {
        'location': 'New York, NY',
        'timestamp': datetime.now(),
        'temperature': {
            'current': 75.0,
            'high': 82.0,
            'low': 68.0,
            'historical_average': 72.0
        },
        'precipitation': {
            'current_24h': 0.5,
            'historical_average_24h': 0.1,
            'probability': 0.3
        },
        'wind': {
            'speed': 10.0,
            'direction': 'NW',
            'gusts': 15.0
        },
        'alerts': [
            {
                'title': 'Heat Advisory',
                'description': 'Temperatures expected to reach 95°F',
                'severity': 'moderate',
                'duration': '24 hours'
            }
        ]
    }

@pytest.fixture
def mock_openai_client():
    """Mock OpenAI client for testing"""
    mock_client = Mock()
    mock_response = Mock()
    mock_response.choices = [Mock()]
    mock_response.choices[0].message = Mock()
    mock_response.choices[0].message.content = "This is a mock AI response for testing purposes."
    mock_client.chat.completions.create.return_value = mock_response
    return mock_client

@pytest.fixture
def sample_event_data():
    """Sample event data for testing"""
    return {
        'event_id': 'event_001',
        'event_type': 'market_event',
        'severity': 'medium',
        'timestamp': datetime.now(),
        'source': 'market_data',
        'title': 'High Volatility Detected',
        'description': 'Market volatility exceeded normal thresholds',
        'affected_entities': ['AAPL', 'MSFT', 'GOOGL'],
        'confidence_score': 0.85,
        'impact_score': 0.6,
        'metadata': {
            'volatility_level': 0.25,
            'threshold': 0.20,
            'duration': '2 hours'
        }
    }

@pytest.fixture
def sample_ml_training_data():
    """Sample ML training data for testing"""
    np.random.seed(42)
    n_samples = 1000
    
    # Generate synthetic client behavior data
    data = {
        'age': np.random.randint(25, 75, n_samples),
        'net_worth': np.random.lognormal(13, 1, n_samples),  # Log-normal distribution
        'portfolio_return': np.random.normal(0.08, 0.15, n_samples),
        'portfolio_volatility': np.random.uniform(0.05, 0.30, n_samples),
        'login_frequency': np.random.poisson(5, n_samples),
        'session_duration': np.random.exponential(30, n_samples),
        'account_age_days': np.random.randint(30, 3650, n_samples),
        'goals_count': np.random.randint(1, 6, n_samples),
        'risk_tolerance': np.random.choice([1, 2, 3, 4, 5], n_samples),
        'market_volatility': np.random.uniform(0.10, 0.40, n_samples)
    }
    
    # Generate target variables
    data['churn_probability'] = np.random.beta(2, 8, n_samples)  # Skewed towards low churn
    data['satisfaction_level'] = np.random.beta(8, 2, n_samples)  # Skewed towards high satisfaction
    data['risk_tolerance_change'] = np.random.choice([0, 1], n_samples, p=[0.8, 0.2])
    
    return pd.DataFrame(data)

@pytest.fixture
def temp_directory():
    """Create a temporary directory for testing"""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir

@pytest.fixture
def mock_database():
    """Mock database for testing"""
    mock_db = Mock()
    mock_session = Mock()
    mock_db.session = mock_session
    return mock_db

# Test data generators
def generate_price_series(start_date, end_date, initial_price=100, volatility=0.02):
    """Generate realistic price series for testing"""
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    np.random.seed(42)
    
    returns = np.random.normal(0.0008, volatility, len(dates))
    prices = [initial_price]
    
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    return pd.Series(prices, index=dates)

def generate_portfolio_returns(start_date, end_date, assets, correlations=None):
    """Generate correlated portfolio returns for testing"""
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    n_assets = len(assets)
    n_periods = len(dates)
    
    if correlations is None:
        correlations = np.eye(n_assets) * 0.6 + np.ones((n_assets, n_assets)) * 0.2
        np.fill_diagonal(correlations, 1.0)
    
    # Generate correlated returns
    np.random.seed(42)
    returns = np.random.multivariate_normal(
        mean=[0.0008] * n_assets,
        cov=correlations * 0.02**2,
        size=n_periods
    )
    
    return pd.DataFrame(returns, index=dates, columns=assets)

# Utility functions for tests
def assert_portfolio_valid(portfolio_data):
    """Assert that portfolio data is valid"""
    assert 'portfolio_id' in portfolio_data
    assert 'client_id' in portfolio_data
    assert 'total_value' in portfolio_data
    assert portfolio_data['total_value'] > 0
    
    if 'target_allocation' in portfolio_data:
        allocation_sum = sum(portfolio_data['target_allocation'].values())
        assert abs(allocation_sum - 1.0) < 0.01  # Allow small rounding errors

def assert_client_valid(client_data):
    """Assert that client data is valid"""
    assert 'client_id' in client_data
    assert 'first_name' in client_data
    assert 'last_name' in client_data
    assert 'email' in client_data
    assert '@' in client_data['email']
    
    if 'net_worth' in client_data:
        assert client_data['net_worth'] >= 0

def assert_event_valid(event_data):
    """Assert that event data is valid"""
    assert 'event_id' in event_data
    assert 'event_type' in event_data
    assert 'severity' in event_data
    assert 'timestamp' in event_data
    assert 'confidence_score' in event_data
    assert 0 <= event_data['confidence_score'] <= 1
    
    if 'impact_score' in event_data:
        assert 0 <= event_data['impact_score'] <= 1

