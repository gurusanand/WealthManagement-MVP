from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, String, Integer, DateTime, Boolean, Numeric, Text, JSON, Date, Index
from datetime import datetime, date
import uuid
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from enum import Enum

db = SQLAlchemy()

class DataSource(str, Enum):
    ALPHA_VANTAGE = "Alpha Vantage"
    YAHOO_FINANCE = "Yahoo Finance"
    FMP = "Financial Modeling Prep"
    NEWSAPI = "NewsAPI"
    OPENWEATHER = "OpenWeatherMap"
    NASA_EARTH = "NASA Earth Data"
    WORLD_BANK = "World Bank"
    FRED = "FRED"

class WeatherCondition(str, Enum):
    CLEAR = "Clear"
    CLOUDY = "Cloudy"
    RAINY = "Rainy"
    STORMY = "Stormy"
    SNOWY = "Snowy"
    FOGGY = "Foggy"

class SatelliteDataType(str, Enum):
    NDVI = "NDVI"  # Normalized Difference Vegetation Index
    LAND_COVER = "Land Cover"
    TEMPERATURE = "Temperature"
    PRECIPITATION = "Precipitation"
    FIRE_DETECTION = "Fire Detection"
    FLOOD_MONITORING = "Flood Monitoring"

# SQLAlchemy Models
class MarketData(db.Model):
    __tablename__ = 'market_data'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    symbol = db.Column(db.String(10), nullable=False, index=True)
    data_date = db.Column(db.Date, nullable=False, index=True)
    
    # OHLCV Data
    open_price = db.Column(db.Numeric(18, 4))
    high_price = db.Column(db.Numeric(18, 4))
    low_price = db.Column(db.Numeric(18, 4))
    close_price = db.Column(db.Numeric(18, 4))
    volume = db.Column(db.BigInteger)
    adjusted_close = db.Column(db.Numeric(18, 4))
    
    # Technical Indicators
    sma_20 = db.Column(db.Numeric(18, 4))  # 20-day Simple Moving Average
    sma_50 = db.Column(db.Numeric(18, 4))  # 50-day Simple Moving Average
    rsi = db.Column(db.Numeric(5, 2))      # Relative Strength Index
    macd = db.Column(db.Numeric(10, 4))    # MACD Line
    macd_signal = db.Column(db.Numeric(10, 4))  # MACD Signal Line
    macd_histogram = db.Column(db.Numeric(10, 4))  # MACD Histogram
    
    # Fundamental Data
    pe_ratio = db.Column(db.Numeric(10, 2))
    pb_ratio = db.Column(db.Numeric(10, 2))
    dividend_yield = db.Column(db.Numeric(5, 4))
    market_cap = db.Column(db.BigInteger)
    
    # Data Source
    data_source = db.Column(db.Enum(DataSource), nullable=False)
    
    # Metadata
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('ix_market_data_symbol_date', 'symbol', 'data_date'),
    )
    
    def __repr__(self):
        return f'<MarketData {self.symbol} {self.data_date}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'symbol': self.symbol,
            'data_date': self.data_date.isoformat() if self.data_date else None,
            'ohlcv': {
                'open': float(self.open_price) if self.open_price else None,
                'high': float(self.high_price) if self.high_price else None,
                'low': float(self.low_price) if self.low_price else None,
                'close': float(self.close_price) if self.close_price else None,
                'volume': int(self.volume) if self.volume else None,
                'adjusted_close': float(self.adjusted_close) if self.adjusted_close else None
            },
            'technical_indicators': {
                'sma_20': float(self.sma_20) if self.sma_20 else None,
                'sma_50': float(self.sma_50) if self.sma_50 else None,
                'rsi': float(self.rsi) if self.rsi else None,
                'macd': {
                    'macd': float(self.macd) if self.macd else None,
                    'signal': float(self.macd_signal) if self.macd_signal else None,
                    'histogram': float(self.macd_histogram) if self.macd_histogram else None
                }
            },
            'fundamentals': {
                'pe_ratio': float(self.pe_ratio) if self.pe_ratio else None,
                'pb_ratio': float(self.pb_ratio) if self.pb_ratio else None,
                'dividend_yield': float(self.dividend_yield) if self.dividend_yield else None,
                'market_cap': int(self.market_cap) if self.market_cap else None
            },
            'data_source': self.data_source.value if self.data_source else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class WeatherData(db.Model):
    __tablename__ = 'weather_data'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Location
    latitude = db.Column(db.Numeric(10, 8), nullable=False)
    longitude = db.Column(db.Numeric(11, 8), nullable=False)
    location_name = db.Column(db.String(100))
    
    # Weather Metrics
    temperature = db.Column(db.Numeric(5, 2))  # Celsius
    humidity = db.Column(db.Numeric(5, 2))     # Percentage
    pressure = db.Column(db.Numeric(7, 2))     # hPa
    wind_speed = db.Column(db.Numeric(5, 2))   # m/s
    wind_direction = db.Column(db.Integer)     # Degrees
    visibility = db.Column(db.Numeric(5, 2))   # km
    uv_index = db.Column(db.Numeric(4, 2))
    
    # Weather Condition
    weather_condition = db.Column(db.Enum(WeatherCondition))
    condition_description = db.Column(db.String(100))
    
    # Environmental Data
    air_quality_index = db.Column(db.Integer)
    precipitation = db.Column(db.Numeric(6, 2))  # mm
    cloud_cover = db.Column(db.Integer)          # Percentage
    
    # Timestamps
    recorded_at = db.Column(db.DateTime, nullable=False)
    data_source = db.Column(db.Enum(DataSource), default=DataSource.OPENWEATHER)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<WeatherData {self.location_name} {self.recorded_at}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'location': {
                'latitude': float(self.latitude) if self.latitude else None,
                'longitude': float(self.longitude) if self.longitude else None,
                'name': self.location_name
            },
            'weather': {
                'temperature': float(self.temperature) if self.temperature else None,
                'humidity': float(self.humidity) if self.humidity else None,
                'pressure': float(self.pressure) if self.pressure else None,
                'wind_speed': float(self.wind_speed) if self.wind_speed else None,
                'wind_direction': self.wind_direction,
                'visibility': float(self.visibility) if self.visibility else None,
                'uv_index': float(self.uv_index) if self.uv_index else None,
                'condition': self.weather_condition.value if self.weather_condition else None,
                'description': self.condition_description
            },
            'environmental': {
                'air_quality_index': self.air_quality_index,
                'precipitation': float(self.precipitation) if self.precipitation else None,
                'cloud_cover': self.cloud_cover
            },
            'recorded_at': self.recorded_at.isoformat() if self.recorded_at else None,
            'data_source': self.data_source.value if self.data_source else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class SatelliteData(db.Model):
    __tablename__ = 'satellite_data'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Location (can be point or bounding box)
    latitude = db.Column(db.Numeric(10, 8))
    longitude = db.Column(db.Numeric(11, 8))
    bounding_box = db.Column(JSON)  # GeoJSON polygon for area coverage
    location_name = db.Column(db.String(100))
    
    # Image Metadata
    image_url = db.Column(db.String(500))
    image_date = db.Column(db.Date, nullable=False)
    satellite_source = db.Column(db.String(100))  # "Landsat", "Sentinel", "MODIS"
    resolution = db.Column(db.Numeric(10, 2))     # meters per pixel
    cloud_cover = db.Column(db.Numeric(5, 2))     # percentage
    
    # Analysis Results
    data_type = db.Column(db.Enum(SatelliteDataType), nullable=False)
    data_value = db.Column(db.Numeric(10, 4))
    
    # Change Detection
    has_change = db.Column(db.Boolean, default=False)
    change_type = db.Column(db.String(50))
    change_confidence = db.Column(db.Numeric(3, 2))  # 0.00 to 1.00
    
    # Related Assets
    related_commodities = db.Column(JSON)  # Array of commodity symbols
    affected_companies = db.Column(JSON)   # Array of company symbols
    
    # Additional Metadata
    additional_metadata = db.Column(JSON)  # Flexible storage for additional satellite metadata
    data_source = db.Column(db.Enum(DataSource), default=DataSource.NASA_EARTH)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<SatelliteData {self.data_type} {self.image_date}>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'location': {
                'latitude': float(self.latitude) if self.latitude else None,
                'longitude': float(self.longitude) if self.longitude else None,
                'bounding_box': self.bounding_box,
                'name': self.location_name
            },
            'image_metadata': {
                'url': self.image_url,
                'date': self.image_date.isoformat() if self.image_date else None,
                'satellite': self.satellite_source,
                'resolution': float(self.resolution) if self.resolution else None,
                'cloud_cover': float(self.cloud_cover) if self.cloud_cover else None
            },
            'analysis': {
                'data_type': self.data_type.value if self.data_type else None,
                'value': float(self.data_value) if self.data_value else None,
                'change_detection': {
                    'has_change': self.has_change,
                    'type': self.change_type,
                    'confidence': float(self.change_confidence) if self.change_confidence else None
                }
            },
            'related_assets': {
                'commodities': self.related_commodities,
                'companies': self.affected_companies
            },
            'metadata': self.additional_metadata,
            'data_source': self.data_source.value if self.data_source else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

class NewsData(db.Model):
    __tablename__ = 'news_data'
    
    id = db.Column(db.String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    
    # Article Information
    title = db.Column(db.String(500), nullable=False)
    description = db.Column(db.Text)
    content = db.Column(db.Text)
    url = db.Column(db.String(1000))
    image_url = db.Column(db.String(1000))
    
    # Source Information
    source_name = db.Column(db.String(100))
    author = db.Column(db.String(200))
    published_at = db.Column(db.DateTime)
    
    # Classification
    category = db.Column(db.String(50))  # "business", "technology", "health", etc.
    language = db.Column(db.String(10), default='en')
    country = db.Column(db.String(2))    # ISO country code
    
    # Financial Relevance
    related_symbols = db.Column(JSON)    # Array of stock symbols mentioned
    related_sectors = db.Column(JSON)    # Array of sectors mentioned
    financial_relevance_score = db.Column(db.Numeric(3, 2))  # 0.00 to 1.00
    
    # Sentiment Analysis
    sentiment_score = db.Column(db.Numeric(3, 2))  # -1.00 to 1.00
    sentiment_label = db.Column(db.String(20))     # "positive", "negative", "neutral"
    sentiment_confidence = db.Column(db.Numeric(3, 2))  # 0.00 to 1.00
    
    # Keywords and Entities
    keywords = db.Column(JSON)           # Array of extracted keywords
    entities = db.Column(JSON)           # Named entities (companies, people, locations)
    
    # Data Source
    data_source = db.Column(db.Enum(DataSource), default=DataSource.NEWSAPI)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    def __repr__(self):
        return f'<NewsData {self.title[:50]}...>'
    
    def to_dict(self):
        return {
            'id': self.id,
            'article': {
                'title': self.title,
                'description': self.description,
                'content': self.content,
                'url': self.url,
                'image_url': self.image_url
            },
            'source': {
                'name': self.source_name,
                'author': self.author,
                'published_at': self.published_at.isoformat() if self.published_at else None
            },
            'classification': {
                'category': self.category,
                'language': self.language,
                'country': self.country
            },
            'financial_relevance': {
                'symbols': self.related_symbols,
                'sectors': self.related_sectors,
                'score': float(self.financial_relevance_score) if self.financial_relevance_score else None
            },
            'sentiment': {
                'score': float(self.sentiment_score) if self.sentiment_score else None,
                'label': self.sentiment_label,
                'confidence': float(self.sentiment_confidence) if self.sentiment_confidence else None
            },
            'analysis': {
                'keywords': self.keywords,
                'entities': self.entities
            },
            'data_source': self.data_source.value if self.data_source else None,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

# Pydantic Models
class OHLCVData(BaseModel):
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    close: Optional[float] = None
    volume: Optional[int] = None
    adjusted_close: Optional[float] = None

class TechnicalIndicators(BaseModel):
    sma_20: Optional[float] = None
    sma_50: Optional[float] = None
    rsi: Optional[float] = None
    macd: Optional[Dict[str, float]] = None

class FundamentalData(BaseModel):
    pe_ratio: Optional[float] = None
    pb_ratio: Optional[float] = None
    dividend_yield: Optional[float] = None
    market_cap: Optional[int] = None

class MarketDataBase(BaseModel):
    symbol: str = Field(..., min_length=1, max_length=10)
    data_date: str  # ISO date string
    ohlcv: Optional[OHLCVData] = None
    technical_indicators: Optional[TechnicalIndicators] = None
    fundamentals: Optional[FundamentalData] = None
    data_source: DataSource

class MarketDataCreate(MarketDataBase):
    pass

class MarketDataResponse(MarketDataBase):
    id: str
    created_at: str
    
    class Config:
        from_attributes = True

class WeatherMetrics(BaseModel):
    temperature: Optional[float] = None
    humidity: Optional[float] = None
    pressure: Optional[float] = None
    wind_speed: Optional[float] = None
    wind_direction: Optional[int] = None
    visibility: Optional[float] = None
    uv_index: Optional[float] = None
    condition: Optional[WeatherCondition] = None
    description: Optional[str] = None

class EnvironmentalData(BaseModel):
    air_quality_index: Optional[int] = None
    precipitation: Optional[float] = None
    cloud_cover: Optional[int] = None

class LocationInfo(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    name: Optional[str] = None

class WeatherDataBase(BaseModel):
    location: LocationInfo
    weather: WeatherMetrics
    environmental: Optional[EnvironmentalData] = None
    recorded_at: str  # ISO datetime string
    data_source: Optional[DataSource] = DataSource.OPENWEATHER

class WeatherDataCreate(WeatherDataBase):
    pass

class WeatherDataResponse(WeatherDataBase):
    id: str
    created_at: str
    
    class Config:
        from_attributes = True

class ImageMetadata(BaseModel):
    url: Optional[str] = None
    date: str  # ISO date string
    satellite: Optional[str] = None
    resolution: Optional[float] = None
    cloud_cover: Optional[float] = None

class ChangeDetection(BaseModel):
    has_change: bool = False
    type: Optional[str] = None
    confidence: Optional[float] = Field(None, ge=0, le=1)

class SatelliteAnalysis(BaseModel):
    data_type: SatelliteDataType
    value: Optional[float] = None
    change_detection: Optional[ChangeDetection] = None

class RelatedAssets(BaseModel):
    commodities: Optional[List[str]] = []
    companies: Optional[List[str]] = []

class SatelliteDataBase(BaseModel):
    location: LocationInfo
    image_metadata: ImageMetadata
    analysis: SatelliteAnalysis
    related_assets: Optional[RelatedAssets] = None
    metadata: Optional[Dict[str, Any]] = None
    data_source: Optional[DataSource] = DataSource.NASA_EARTH

class SatelliteDataCreate(SatelliteDataBase):
    pass

class SatelliteDataResponse(SatelliteDataBase):
    id: str
    created_at: str
    
    class Config:
        from_attributes = True

