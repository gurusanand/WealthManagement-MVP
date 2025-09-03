import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from enum import Enum
import json
import asyncio
from abc import ABC, abstractmethod
import re
from collections import defaultdict, deque
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)

class EventType(Enum):
    """Types of events that can be processed"""
    MARKET_EVENT = "market_event"
    NEWS_EVENT = "news_event"
    ECONOMIC_EVENT = "economic_event"
    CORPORATE_EVENT = "corporate_event"
    LIFE_EVENT = "life_event"
    PORTFOLIO_EVENT = "portfolio_event"
    COMPLIANCE_EVENT = "compliance_event"
    SYSTEM_EVENT = "system_event"
    WEATHER_EVENT = "weather_event"
    SATELLITE_EVENT = "satellite_event"

class EventSeverity(Enum):
    """Event severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class EventStatus(Enum):
    """Event processing status"""
    DETECTED = "detected"
    PROCESSING = "processing"
    ANALYZED = "analyzed"
    ACTION_REQUIRED = "action_required"
    COMPLETED = "completed"
    IGNORED = "ignored"
    ERROR = "error"

@dataclass
class Event:
    """Event data structure"""
    event_id: str
    event_type: EventType
    severity: EventSeverity
    status: EventStatus
    timestamp: datetime
    source: str
    title: str
    description: str
    raw_data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)
    affected_entities: List[str] = field(default_factory=list)
    confidence_score: float = 0.0
    impact_score: float = 0.0
    processing_history: List[Dict[str, Any]] = field(default_factory=list)

@dataclass
class EventAnalysis:
    """Event analysis results"""
    event_id: str
    analysis_timestamp: datetime
    sentiment_score: float
    impact_assessment: Dict[str, float]
    affected_portfolios: List[str]
    affected_clients: List[str]
    market_implications: Dict[str, Any]
    recommended_actions: List[str]
    risk_factors: Dict[str, float]
    correlation_analysis: Dict[str, float]
    historical_precedents: List[Dict[str, Any]]
    confidence_level: float

@dataclass
class EventResponse:
    """Event response and actions"""
    event_id: str
    response_timestamp: datetime
    actions_taken: List[str]
    notifications_sent: List[str]
    portfolio_adjustments: List[Dict[str, Any]]
    client_communications: List[Dict[str, Any]]
    compliance_checks: List[str]
    follow_up_required: bool
    response_effectiveness: Optional[float] = None

class EventDetector(ABC):
    """Abstract base class for event detectors"""
    
    @abstractmethod
    def detect_events(self, data: Dict[str, Any]) -> List[Event]:
        """Detect events from data"""
        pass
    
    @abstractmethod
    def get_detector_name(self) -> str:
        """Get detector name"""
        pass

class NewsEventDetector(EventDetector):
    """Detector for news events"""
    
    def __init__(self):
        self.keywords = {
            'market_crash': ['crash', 'plunge', 'collapse', 'panic', 'selloff'],
            'earnings': ['earnings', 'quarterly', 'revenue', 'profit', 'loss'],
            'merger': ['merger', 'acquisition', 'takeover', 'buyout'],
            'regulatory': ['regulation', 'policy', 'fed', 'central bank', 'interest rate'],
            'geopolitical': ['war', 'conflict', 'sanctions', 'trade war', 'election'],
            'economic': ['gdp', 'inflation', 'unemployment', 'recession', 'growth']
        }
        
        self.severity_keywords = {
            EventSeverity.CRITICAL: ['crisis', 'emergency', 'collapse', 'crash'],
            EventSeverity.HIGH: ['significant', 'major', 'substantial', 'dramatic'],
            EventSeverity.MEDIUM: ['moderate', 'notable', 'considerable'],
            EventSeverity.LOW: ['minor', 'slight', 'small']
        }
    
    def detect_events(self, data: Dict[str, Any]) -> List[Event]:
        """Detect news events from news data"""
        events = []
        
        try:
            news_items = data.get('news', [])
            
            for item in news_items:
                title = item.get('title', '').lower()
                description = item.get('description', '').lower()
                content = f"{title} {description}"
                
                # Detect event type
                event_type = self._classify_news_event(content)
                if event_type:
                    # Determine severity
                    severity = self._assess_news_severity(content)
                    
                    # Calculate confidence and impact
                    confidence = self._calculate_confidence(content, event_type)
                    impact = self._assess_impact(content, event_type)
                    
                    # Extract affected entities
                    affected_entities = self._extract_entities(content)
                    
                    event = Event(
                        event_id=f"news_{item.get('id', int(time.time()))}",
                        event_type=EventType.NEWS_EVENT,
                        severity=severity,
                        status=EventStatus.DETECTED,
                        timestamp=datetime.fromisoformat(item.get('published_at', datetime.now().isoformat())),
                        source=item.get('source', 'unknown'),
                        title=item.get('title', ''),
                        description=item.get('description', ''),
                        raw_data=item,
                        metadata={
                            'url': item.get('url'),
                            'author': item.get('author'),
                            'category': event_type
                        },
                        affected_entities=affected_entities,
                        confidence_score=confidence,
                        impact_score=impact
                    )
                    
                    events.append(event)
                    
        except Exception as e:
            logger.error(f"Error detecting news events: {str(e)}")
        
        return events
    
    def _classify_news_event(self, content: str) -> Optional[str]:
        """Classify news event type"""
        for category, keywords in self.keywords.items():
            if any(keyword in content for keyword in keywords):
                return category
        return None
    
    def _assess_news_severity(self, content: str) -> EventSeverity:
        """Assess news event severity"""
        for severity, keywords in self.severity_keywords.items():
            if any(keyword in content for keyword in keywords):
                return severity
        return EventSeverity.MEDIUM
    
    def _calculate_confidence(self, content: str, event_type: str) -> float:
        """Calculate confidence score for event detection"""
        keyword_matches = sum(1 for keyword in self.keywords.get(event_type, []) if keyword in content)
        max_matches = len(self.keywords.get(event_type, []))
        return min(keyword_matches / max_matches if max_matches > 0 else 0.5, 1.0)
    
    def _assess_impact(self, content: str, event_type: str) -> float:
        """Assess potential impact of news event"""
        impact_keywords = {
            'high_impact': ['global', 'worldwide', 'major', 'significant', 'substantial'],
            'medium_impact': ['regional', 'sector', 'industry', 'moderate'],
            'low_impact': ['local', 'minor', 'slight', 'limited']
        }
        
        if any(keyword in content for keyword in impact_keywords['high_impact']):
            return 0.8
        elif any(keyword in content for keyword in impact_keywords['medium_impact']):
            return 0.5
        elif any(keyword in content for keyword in impact_keywords['low_impact']):
            return 0.2
        else:
            return 0.4  # Default medium-low impact
    
    def _extract_entities(self, content: str) -> List[str]:
        """Extract affected entities (companies, sectors, etc.)"""
        entities = []
        
        # Simple entity extraction (in practice, would use NLP libraries)
        company_patterns = [
            r'\b[A-Z][a-z]+ (?:Inc|Corp|Ltd|LLC|Co)\b',
            r'\b[A-Z]{2,5}\b'  # Stock tickers
        ]
        
        for pattern in company_patterns:
            matches = re.findall(pattern, content)
            entities.extend(matches)
        
        return list(set(entities))
    
    def get_detector_name(self) -> str:
        return "NewsEventDetector"

class MarketEventDetector(EventDetector):
    """Detector for market events"""
    
    def __init__(self):
        self.volatility_threshold = 0.02  # 2% daily move
        self.volume_threshold = 2.0  # 2x average volume
        self.correlation_threshold = 0.8
    
    def detect_events(self, data: Dict[str, Any]) -> List[Event]:
        """Detect market events from market data"""
        events = []
        
        try:
            market_data = data.get('market_data', {})
            
            # Detect volatility events
            volatility_events = self._detect_volatility_events(market_data)
            events.extend(volatility_events)
            
            # Detect volume events
            volume_events = self._detect_volume_events(market_data)
            events.extend(volume_events)
            
            # Detect correlation events
            correlation_events = self._detect_correlation_events(market_data)
            events.extend(correlation_events)
            
            # Detect trend events
            trend_events = self._detect_trend_events(market_data)
            events.extend(trend_events)
            
        except Exception as e:
            logger.error(f"Error detecting market events: {str(e)}")
        
        return events
    
    def _detect_volatility_events(self, market_data: Dict[str, Any]) -> List[Event]:
        """Detect high volatility events"""
        events = []
        
        try:
            prices = market_data.get('prices', pd.DataFrame())
            if prices.empty:
                return events
            
            # Calculate daily returns
            returns = prices.pct_change().dropna()
            
            for column in returns.columns:
                latest_return = abs(returns[column].iloc[-1])
                
                if latest_return > self.volatility_threshold:
                    severity = self._assess_volatility_severity(latest_return)
                    
                    event = Event(
                        event_id=f"volatility_{column}_{int(time.time())}",
                        event_type=EventType.MARKET_EVENT,
                        severity=severity,
                        status=EventStatus.DETECTED,
                        timestamp=datetime.now(),
                        source="market_data",
                        title=f"High Volatility in {column}",
                        description=f"Daily return of {latest_return:.2%} exceeds threshold",
                        raw_data={'return': latest_return, 'asset': column},
                        affected_entities=[column],
                        confidence_score=0.9,
                        impact_score=min(latest_return / self.volatility_threshold, 1.0)
                    )
                    
                    events.append(event)
                    
        except Exception as e:
            logger.error(f"Error detecting volatility events: {str(e)}")
        
        return events
    
    def _detect_volume_events(self, market_data: Dict[str, Any]) -> List[Event]:
        """Detect unusual volume events"""
        events = []
        
        try:
            volume = market_data.get('volume', pd.DataFrame())
            if volume.empty:
                return events
            
            for column in volume.columns:
                if len(volume[column]) < 20:
                    continue
                
                avg_volume = volume[column].rolling(20).mean().iloc[-1]
                current_volume = volume[column].iloc[-1]
                
                if current_volume > avg_volume * self.volume_threshold:
                    volume_ratio = current_volume / avg_volume
                    severity = self._assess_volume_severity(volume_ratio)
                    
                    event = Event(
                        event_id=f"volume_{column}_{int(time.time())}",
                        event_type=EventType.MARKET_EVENT,
                        severity=severity,
                        status=EventStatus.DETECTED,
                        timestamp=datetime.now(),
                        source="market_data",
                        title=f"Unusual Volume in {column}",
                        description=f"Volume {volume_ratio:.1f}x above average",
                        raw_data={'volume_ratio': volume_ratio, 'asset': column},
                        affected_entities=[column],
                        confidence_score=0.8,
                        impact_score=min(volume_ratio / self.volume_threshold / 2, 1.0)
                    )
                    
                    events.append(event)
                    
        except Exception as e:
            logger.error(f"Error detecting volume events: {str(e)}")
        
        return events
    
    def _detect_correlation_events(self, market_data: Dict[str, Any]) -> List[Event]:
        """Detect correlation breakdown events"""
        events = []
        
        try:
            prices = market_data.get('prices', pd.DataFrame())
            if prices.empty or len(prices.columns) < 2:
                return events
            
            returns = prices.pct_change().dropna()
            
            # Calculate rolling correlation
            if len(returns) >= 60:
                recent_corr = returns.tail(20).corr()
                historical_corr = returns.tail(60).corr()
                
                # Find significant correlation changes
                for i in range(len(recent_corr.columns)):
                    for j in range(i+1, len(recent_corr.columns)):
                        asset1 = recent_corr.columns[i]
                        asset2 = recent_corr.columns[j]
                        
                        recent_corr_val = recent_corr.iloc[i, j]
                        historical_corr_val = historical_corr.iloc[i, j]
                        
                        corr_change = abs(recent_corr_val - historical_corr_val)
                        
                        if corr_change > 0.3:  # Significant correlation change
                            event = Event(
                                event_id=f"correlation_{asset1}_{asset2}_{int(time.time())}",
                                event_type=EventType.MARKET_EVENT,
                                severity=EventSeverity.MEDIUM,
                                status=EventStatus.DETECTED,
                                timestamp=datetime.now(),
                                source="market_data",
                                title=f"Correlation Change: {asset1} - {asset2}",
                                description=f"Correlation changed from {historical_corr_val:.2f} to {recent_corr_val:.2f}",
                                raw_data={
                                    'recent_correlation': recent_corr_val,
                                    'historical_correlation': historical_corr_val,
                                    'change': corr_change
                                },
                                affected_entities=[asset1, asset2],
                                confidence_score=0.7,
                                impact_score=corr_change
                            )
                            
                            events.append(event)
                            
        except Exception as e:
            logger.error(f"Error detecting correlation events: {str(e)}")
        
        return events
    
    def _detect_trend_events(self, market_data: Dict[str, Any]) -> List[Event]:
        """Detect trend reversal events"""
        events = []
        
        try:
            prices = market_data.get('prices', pd.DataFrame())
            if prices.empty:
                return events
            
            for column in prices.columns:
                if len(prices[column]) < 50:
                    continue
                
                price_series = prices[column]
                
                # Simple trend detection using moving averages
                sma_20 = price_series.rolling(20).mean()
                sma_50 = price_series.rolling(50).mean()
                
                if len(sma_20) < 2 or len(sma_50) < 2:
                    continue
                
                # Detect crossover
                current_cross = sma_20.iloc[-1] > sma_50.iloc[-1]
                previous_cross = sma_20.iloc[-2] > sma_50.iloc[-2]
                
                if current_cross != previous_cross:
                    trend_direction = "bullish" if current_cross else "bearish"
                    
                    event = Event(
                        event_id=f"trend_{column}_{int(time.time())}",
                        event_type=EventType.MARKET_EVENT,
                        severity=EventSeverity.MEDIUM,
                        status=EventStatus.DETECTED,
                        timestamp=datetime.now(),
                        source="market_data",
                        title=f"Trend Reversal in {column}",
                        description=f"Moving average crossover indicates {trend_direction} trend",
                        raw_data={
                            'trend_direction': trend_direction,
                            'sma_20': sma_20.iloc[-1],
                            'sma_50': sma_50.iloc[-1]
                        },
                        affected_entities=[column],
                        confidence_score=0.6,
                        impact_score=0.5
                    )
                    
                    events.append(event)
                    
        except Exception as e:
            logger.error(f"Error detecting trend events: {str(e)}")
        
        return events
    
    def _assess_volatility_severity(self, volatility: float) -> EventSeverity:
        """Assess volatility event severity"""
        if volatility > 0.1:  # 10%+ move
            return EventSeverity.CRITICAL
        elif volatility > 0.05:  # 5%+ move
            return EventSeverity.HIGH
        elif volatility > 0.03:  # 3%+ move
            return EventSeverity.MEDIUM
        else:
            return EventSeverity.LOW
    
    def _assess_volume_severity(self, volume_ratio: float) -> EventSeverity:
        """Assess volume event severity"""
        if volume_ratio > 5:
            return EventSeverity.HIGH
        elif volume_ratio > 3:
            return EventSeverity.MEDIUM
        else:
            return EventSeverity.LOW
    
    def get_detector_name(self) -> str:
        return "MarketEventDetector"

class WeatherEventDetector(EventDetector):
    """Detector for weather events that might impact investments"""
    
    def __init__(self):
        self.severe_weather_keywords = [
            'hurricane', 'tornado', 'flood', 'drought', 'wildfire',
            'blizzard', 'heatwave', 'storm', 'cyclone', 'typhoon'
        ]
        
        self.affected_sectors = {
            'hurricane': ['insurance', 'utilities', 'oil', 'agriculture'],
            'drought': ['agriculture', 'utilities', 'food'],
            'flood': ['insurance', 'agriculture', 'infrastructure'],
            'wildfire': ['insurance', 'utilities', 'forestry'],
            'extreme_temperature': ['utilities', 'agriculture', 'energy']
        }
    
    def detect_events(self, data: Dict[str, Any]) -> List[Event]:
        """Detect weather events from weather data"""
        events = []
        
        try:
            weather_data = data.get('weather', {})
            
            # Detect severe weather alerts
            alerts = weather_data.get('alerts', [])
            for alert in alerts:
                event = self._process_weather_alert(alert)
                if event:
                    events.append(event)
            
            # Detect extreme temperature events
            temperature_events = self._detect_temperature_events(weather_data)
            events.extend(temperature_events)
            
            # Detect precipitation events
            precipitation_events = self._detect_precipitation_events(weather_data)
            events.extend(precipitation_events)
            
        except Exception as e:
            logger.error(f"Error detecting weather events: {str(e)}")
        
        return events
    
    def _process_weather_alert(self, alert: Dict[str, Any]) -> Optional[Event]:
        """Process weather alert into event"""
        try:
            title = alert.get('title', '').lower()
            description = alert.get('description', '').lower()
            
            # Determine event type
            weather_type = None
            for keyword in self.severe_weather_keywords:
                if keyword in title or keyword in description:
                    weather_type = keyword
                    break
            
            if not weather_type:
                return None
            
            # Assess severity
            severity = self._assess_weather_severity(alert)
            
            # Determine affected sectors
            affected_sectors = self.affected_sectors.get(weather_type, [])
            
            event = Event(
                event_id=f"weather_{weather_type}_{int(time.time())}",
                event_type=EventType.WEATHER_EVENT,
                severity=severity,
                status=EventStatus.DETECTED,
                timestamp=datetime.now(),
                source="weather_service",
                title=alert.get('title', ''),
                description=alert.get('description', ''),
                raw_data=alert,
                metadata={
                    'weather_type': weather_type,
                    'location': alert.get('location'),
                    'duration': alert.get('duration')
                },
                affected_entities=affected_sectors,
                confidence_score=0.8,
                impact_score=self._calculate_weather_impact(weather_type, alert)
            )
            
            return event
            
        except Exception as e:
            logger.error(f"Error processing weather alert: {str(e)}")
            return None
    
    def _detect_temperature_events(self, weather_data: Dict[str, Any]) -> List[Event]:
        """Detect extreme temperature events"""
        events = []
        
        try:
            temperature = weather_data.get('temperature', {})
            current_temp = temperature.get('current')
            historical_avg = temperature.get('historical_average')
            
            if current_temp is not None and historical_avg is not None:
                temp_deviation = abs(current_temp - historical_avg)
                
                if temp_deviation > 10:  # 10+ degree deviation
                    severity = EventSeverity.HIGH if temp_deviation > 20 else EventSeverity.MEDIUM
                    
                    event = Event(
                        event_id=f"temperature_{int(time.time())}",
                        event_type=EventType.WEATHER_EVENT,
                        severity=severity,
                        status=EventStatus.DETECTED,
                        timestamp=datetime.now(),
                        source="weather_service",
                        title="Extreme Temperature Event",
                        description=f"Temperature {temp_deviation:.1f}°F from historical average",
                        raw_data=temperature,
                        affected_entities=['utilities', 'agriculture', 'energy'],
                        confidence_score=0.7,
                        impact_score=min(temp_deviation / 20, 1.0)
                    )
                    
                    events.append(event)
                    
        except Exception as e:
            logger.error(f"Error detecting temperature events: {str(e)}")
        
        return events
    
    def _detect_precipitation_events(self, weather_data: Dict[str, Any]) -> List[Event]:
        """Detect extreme precipitation events"""
        events = []
        
        try:
            precipitation = weather_data.get('precipitation', {})
            current_precip = precipitation.get('current_24h')
            historical_avg = precipitation.get('historical_average_24h')
            
            if current_precip is not None and historical_avg is not None:
                if current_precip > historical_avg * 3:  # 3x normal precipitation
                    severity = EventSeverity.HIGH if current_precip > historical_avg * 5 else EventSeverity.MEDIUM
                    
                    event = Event(
                        event_id=f"precipitation_{int(time.time())}",
                        event_type=EventType.WEATHER_EVENT,
                        severity=severity,
                        status=EventStatus.DETECTED,
                        timestamp=datetime.now(),
                        source="weather_service",
                        title="Extreme Precipitation Event",
                        description=f"Precipitation {current_precip / historical_avg:.1f}x above average",
                        raw_data=precipitation,
                        affected_entities=['agriculture', 'insurance', 'infrastructure'],
                        confidence_score=0.8,
                        impact_score=min(current_precip / (historical_avg * 5), 1.0)
                    )
                    
                    events.append(event)
                    
        except Exception as e:
            logger.error(f"Error detecting precipitation events: {str(e)}")
        
        return events
    
    def _assess_weather_severity(self, alert: Dict[str, Any]) -> EventSeverity:
        """Assess weather event severity"""
        severity_indicators = alert.get('severity', '').lower()
        
        if 'extreme' in severity_indicators or 'emergency' in severity_indicators:
            return EventSeverity.CRITICAL
        elif 'severe' in severity_indicators or 'major' in severity_indicators:
            return EventSeverity.HIGH
        elif 'moderate' in severity_indicators:
            return EventSeverity.MEDIUM
        else:
            return EventSeverity.LOW
    
    def _calculate_weather_impact(self, weather_type: str, alert: Dict[str, Any]) -> float:
        """Calculate weather event impact score"""
        base_impact = {
            'hurricane': 0.8,
            'tornado': 0.6,
            'flood': 0.7,
            'drought': 0.6,
            'wildfire': 0.5,
            'blizzard': 0.4,
            'heatwave': 0.3
        }
        
        return base_impact.get(weather_type, 0.3)
    
    def get_detector_name(self) -> str:
        return "WeatherEventDetector"

class EventAnalyzer:
    """
    Advanced Event Analyzer
    
    Analyzes detected events to assess their impact, sentiment, and implications
    for portfolios and clients.
    """
    
    def __init__(self):
        self.sentiment_analyzer = None  # Would integrate with NLP library
        self.impact_models = {}
        self.historical_events = []
        
    def analyze_event(self, event: Event, context_data: Dict[str, Any]) -> EventAnalysis:
        """
        Comprehensive event analysis
        
        Args:
            event: Event to analyze
            context_data: Additional context data (portfolios, market data, etc.)
            
        Returns:
            Event analysis results
        """
        try:
            # Sentiment analysis
            sentiment_score = self._analyze_sentiment(event)
            
            # Impact assessment
            impact_assessment = self._assess_impact(event, context_data)
            
            # Identify affected portfolios and clients
            affected_portfolios = self._identify_affected_portfolios(event, context_data)
            affected_clients = self._identify_affected_clients(event, context_data)
            
            # Market implications
            market_implications = self._analyze_market_implications(event, context_data)
            
            # Generate recommendations
            recommended_actions = self._generate_recommendations(event, impact_assessment)
            
            # Risk factor analysis
            risk_factors = self._analyze_risk_factors(event, context_data)
            
            # Correlation analysis
            correlation_analysis = self._analyze_correlations(event, context_data)
            
            # Historical precedents
            historical_precedents = self._find_historical_precedents(event)
            
            # Calculate confidence level
            confidence_level = self._calculate_analysis_confidence(event, impact_assessment)
            
            analysis = EventAnalysis(
                event_id=event.event_id,
                analysis_timestamp=datetime.now(),
                sentiment_score=sentiment_score,
                impact_assessment=impact_assessment,
                affected_portfolios=affected_portfolios,
                affected_clients=affected_clients,
                market_implications=market_implications,
                recommended_actions=recommended_actions,
                risk_factors=risk_factors,
                correlation_analysis=correlation_analysis,
                historical_precedents=historical_precedents,
                confidence_level=confidence_level
            )
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing event {event.event_id}: {str(e)}")
            return self._default_analysis(event.event_id)
    
    def _analyze_sentiment(self, event: Event) -> float:
        """Analyze event sentiment"""
        try:
            # Simple sentiment analysis based on keywords
            positive_keywords = ['growth', 'increase', 'positive', 'gain', 'success', 'improvement']
            negative_keywords = ['decline', 'decrease', 'negative', 'loss', 'failure', 'crisis', 'crash']
            
            text = f"{event.title} {event.description}".lower()
            
            positive_count = sum(1 for keyword in positive_keywords if keyword in text)
            negative_count = sum(1 for keyword in negative_keywords if keyword in text)
            
            if positive_count + negative_count == 0:
                return 0.0  # Neutral
            
            sentiment = (positive_count - negative_count) / (positive_count + negative_count)
            return sentiment
            
        except Exception as e:
            logger.error(f"Error analyzing sentiment: {str(e)}")
            return 0.0
    
    def _assess_impact(self, event: Event, context_data: Dict[str, Any]) -> Dict[str, float]:
        """Assess event impact on different dimensions"""
        
        impact_assessment = {
            'market_impact': 0.0,
            'portfolio_impact': 0.0,
            'sector_impact': 0.0,
            'geographic_impact': 0.0,
            'liquidity_impact': 0.0,
            'volatility_impact': 0.0
        }
        
        try:
            # Base impact from event properties
            base_impact = event.impact_score
            
            # Market impact
            if event.event_type in [EventType.MARKET_EVENT, EventType.ECONOMIC_EVENT]:
                impact_assessment['market_impact'] = base_impact * 0.8
            elif event.event_type == EventType.NEWS_EVENT:
                impact_assessment['market_impact'] = base_impact * 0.6
            else:
                impact_assessment['market_impact'] = base_impact * 0.3
            
            # Portfolio impact based on affected entities
            portfolio_exposure = self._calculate_portfolio_exposure(event, context_data)
            impact_assessment['portfolio_impact'] = base_impact * portfolio_exposure
            
            # Sector impact
            if event.affected_entities:
                impact_assessment['sector_impact'] = base_impact * 0.7
            else:
                impact_assessment['sector_impact'] = base_impact * 0.2
            
            # Volatility impact
            if event.event_type == EventType.MARKET_EVENT:
                impact_assessment['volatility_impact'] = base_impact * 0.9
            else:
                impact_assessment['volatility_impact'] = base_impact * 0.4
            
            # Liquidity impact
            if event.severity in [EventSeverity.CRITICAL, EventSeverity.EMERGENCY]:
                impact_assessment['liquidity_impact'] = base_impact * 0.8
            else:
                impact_assessment['liquidity_impact'] = base_impact * 0.3
            
        except Exception as e:
            logger.error(f"Error assessing impact: {str(e)}")
        
        return impact_assessment
    
    def _identify_affected_portfolios(self, event: Event, context_data: Dict[str, Any]) -> List[str]:
        """Identify portfolios affected by the event"""
        affected_portfolios = []
        
        try:
            portfolios = context_data.get('portfolios', {})
            
            for portfolio_id, portfolio_data in portfolios.items():
                holdings = portfolio_data.get('holdings', {})
                
                # Check if portfolio has exposure to affected entities
                for entity in event.affected_entities:
                    if entity.lower() in [holding.lower() for holding in holdings.keys()]:
                        affected_portfolios.append(portfolio_id)
                        break
                
                # Check sector exposure for weather/economic events
                if event.event_type in [EventType.WEATHER_EVENT, EventType.ECONOMIC_EVENT]:
                    sector_exposure = portfolio_data.get('sector_exposure', {})
                    for entity in event.affected_entities:
                        if entity in sector_exposure and sector_exposure[entity] > 0.05:  # 5%+ exposure
                            affected_portfolios.append(portfolio_id)
                            break
                            
        except Exception as e:
            logger.error(f"Error identifying affected portfolios: {str(e)}")
        
        return list(set(affected_portfolios))
    
    def _identify_affected_clients(self, event: Event, context_data: Dict[str, Any]) -> List[str]:
        """Identify clients affected by the event"""
        affected_clients = []
        
        try:
            clients = context_data.get('clients', {})
            affected_portfolios = self._identify_affected_portfolios(event, context_data)
            
            for client_id, client_data in clients.items():
                client_portfolios = client_data.get('portfolios', [])
                
                # Check if client has affected portfolios
                if any(portfolio in affected_portfolios for portfolio in client_portfolios):
                    affected_clients.append(client_id)
                
                # Check for life events correlation
                if event.event_type == EventType.LIFE_EVENT:
                    if client_id in event.affected_entities:
                        affected_clients.append(client_id)
                        
        except Exception as e:
            logger.error(f"Error identifying affected clients: {str(e)}")
        
        return list(set(affected_clients))
    
    def _analyze_market_implications(self, event: Event, context_data: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze market implications of the event"""
        
        implications = {
            'expected_volatility_change': 0.0,
            'expected_correlation_change': 0.0,
            'sector_rotation_probability': 0.0,
            'flight_to_quality_probability': 0.0,
            'liquidity_impact': 0.0
        }
        
        try:
            # Volatility implications
            if event.event_type == EventType.MARKET_EVENT:
                implications['expected_volatility_change'] = event.impact_score * 0.5
            elif event.severity in [EventSeverity.CRITICAL, EventSeverity.EMERGENCY]:
                implications['expected_volatility_change'] = event.impact_score * 0.3
            
            # Correlation implications
            if event.event_type in [EventType.MARKET_EVENT, EventType.ECONOMIC_EVENT]:
                implications['expected_correlation_change'] = event.impact_score * 0.4
            
            # Sector rotation
            if event.affected_entities and event.event_type != EventType.MARKET_EVENT:
                implications['sector_rotation_probability'] = event.impact_score * 0.6
            
            # Flight to quality
            if event.severity in [EventSeverity.HIGH, EventSeverity.CRITICAL, EventSeverity.EMERGENCY]:
                implications['flight_to_quality_probability'] = event.impact_score * 0.7
            
            # Liquidity impact
            if event.event_type == EventType.MARKET_EVENT and event.severity >= EventSeverity.HIGH:
                implications['liquidity_impact'] = event.impact_score * 0.8
                
        except Exception as e:
            logger.error(f"Error analyzing market implications: {str(e)}")
        
        return implications
    
    def _generate_recommendations(self, event: Event, impact_assessment: Dict[str, float]) -> List[str]:
        """Generate recommended actions based on event analysis"""
        
        recommendations = []
        
        try:
            # High impact recommendations
            if max(impact_assessment.values()) > 0.7:
                recommendations.append("Consider immediate portfolio review and risk assessment")
                recommendations.append("Evaluate hedging strategies for affected positions")
                recommendations.append("Prepare client communications regarding potential impact")
            
            # Market event recommendations
            if event.event_type == EventType.MARKET_EVENT:
                if event.severity >= EventSeverity.HIGH:
                    recommendations.append("Monitor portfolio volatility and consider rebalancing")
                    recommendations.append("Review stop-loss orders and risk management rules")
                
                if impact_assessment.get('liquidity_impact', 0) > 0.5:
                    recommendations.append("Assess portfolio liquidity and cash positions")
            
            # News event recommendations
            if event.event_type == EventType.NEWS_EVENT:
                recommendations.append("Monitor news developments for additional information")
                recommendations.append("Assess sentiment impact on affected securities")
            
            # Weather event recommendations
            if event.event_type == EventType.WEATHER_EVENT:
                recommendations.append("Review exposure to weather-sensitive sectors")
                recommendations.append("Consider supply chain impact on holdings")
            
            # Sector-specific recommendations
            if event.affected_entities:
                recommendations.append(f"Review exposure to affected entities: {', '.join(event.affected_entities[:3])}")
            
            # Severity-based recommendations
            if event.severity >= EventSeverity.CRITICAL:
                recommendations.append("Activate crisis management protocols")
                recommendations.append("Consider emergency client communications")
            
        except Exception as e:
            logger.error(f"Error generating recommendations: {str(e)}")
        
        return recommendations
    
    def _analyze_risk_factors(self, event: Event, context_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze risk factors associated with the event"""
        
        risk_factors = {
            'systematic_risk': 0.0,
            'idiosyncratic_risk': 0.0,
            'liquidity_risk': 0.0,
            'credit_risk': 0.0,
            'operational_risk': 0.0,
            'reputational_risk': 0.0
        }
        
        try:
            base_risk = event.impact_score
            
            # Systematic risk
            if event.event_type in [EventType.MARKET_EVENT, EventType.ECONOMIC_EVENT]:
                risk_factors['systematic_risk'] = base_risk * 0.8
            else:
                risk_factors['systematic_risk'] = base_risk * 0.3
            
            # Idiosyncratic risk
            if event.affected_entities:
                risk_factors['idiosyncratic_risk'] = base_risk * 0.6
            else:
                risk_factors['idiosyncratic_risk'] = base_risk * 0.2
            
            # Liquidity risk
            if event.severity >= EventSeverity.HIGH:
                risk_factors['liquidity_risk'] = base_risk * 0.7
            else:
                risk_factors['liquidity_risk'] = base_risk * 0.2
            
            # Credit risk
            if 'credit' in event.description.lower() or 'debt' in event.description.lower():
                risk_factors['credit_risk'] = base_risk * 0.8
            else:
                risk_factors['credit_risk'] = base_risk * 0.1
            
            # Operational risk
            if event.event_type == EventType.SYSTEM_EVENT:
                risk_factors['operational_risk'] = base_risk * 0.9
            else:
                risk_factors['operational_risk'] = base_risk * 0.1
            
            # Reputational risk
            if event.event_type == EventType.NEWS_EVENT and event.severity >= EventSeverity.MEDIUM:
                risk_factors['reputational_risk'] = base_risk * 0.5
            else:
                risk_factors['reputational_risk'] = base_risk * 0.1
                
        except Exception as e:
            logger.error(f"Error analyzing risk factors: {str(e)}")
        
        return risk_factors
    
    def _analyze_correlations(self, event: Event, context_data: Dict[str, Any]) -> Dict[str, float]:
        """Analyze correlations with other events or market factors"""
        
        correlations = {
            'market_correlation': 0.0,
            'sector_correlation': 0.0,
            'economic_correlation': 0.0,
            'seasonal_correlation': 0.0
        }
        
        try:
            # Market correlation
            if event.event_type == EventType.MARKET_EVENT:
                correlations['market_correlation'] = 0.9
            elif event.event_type in [EventType.NEWS_EVENT, EventType.ECONOMIC_EVENT]:
                correlations['market_correlation'] = 0.6
            else:
                correlations['market_correlation'] = 0.3
            
            # Sector correlation
            if event.affected_entities:
                correlations['sector_correlation'] = 0.8
            else:
                correlations['sector_correlation'] = 0.2
            
            # Economic correlation
            if event.event_type == EventType.ECONOMIC_EVENT:
                correlations['economic_correlation'] = 0.9
            elif event.event_type == EventType.WEATHER_EVENT:
                correlations['economic_correlation'] = 0.4
            else:
                correlations['economic_correlation'] = 0.2
            
            # Seasonal correlation (simplified)
            current_month = datetime.now().month
            if event.event_type == EventType.WEATHER_EVENT:
                if current_month in [6, 7, 8, 9]:  # Hurricane season
                    correlations['seasonal_correlation'] = 0.7
                else:
                    correlations['seasonal_correlation'] = 0.3
            else:
                correlations['seasonal_correlation'] = 0.1
                
        except Exception as e:
            logger.error(f"Error analyzing correlations: {str(e)}")
        
        return correlations
    
    def _find_historical_precedents(self, event: Event) -> List[Dict[str, Any]]:
        """Find historical precedents for similar events"""
        
        precedents = []
        
        try:
            # Simplified historical precedent matching
            # In practice, this would query a historical events database
            
            if event.event_type == EventType.MARKET_EVENT and event.severity >= EventSeverity.HIGH:
                precedents.append({
                    'date': '2020-03-16',
                    'description': 'COVID-19 market crash',
                    'impact': -0.12,
                    'recovery_time_days': 150
                })
                precedents.append({
                    'date': '2008-09-15',
                    'description': 'Lehman Brothers collapse',
                    'impact': -0.45,
                    'recovery_time_days': 500
                })
            
            elif event.event_type == EventType.WEATHER_EVENT:
                precedents.append({
                    'date': '2005-08-29',
                    'description': 'Hurricane Katrina',
                    'impact': -0.08,
                    'recovery_time_days': 90
                })
            
            elif event.event_type == EventType.NEWS_EVENT and 'earnings' in event.description.lower():
                precedents.append({
                    'date': '2022-07-26',
                    'description': 'Tech earnings disappointment',
                    'impact': -0.05,
                    'recovery_time_days': 30
                })
                
        except Exception as e:
            logger.error(f"Error finding historical precedents: {str(e)}")
        
        return precedents
    
    def _calculate_portfolio_exposure(self, event: Event, context_data: Dict[str, Any]) -> float:
        """Calculate average portfolio exposure to event"""
        
        try:
            portfolios = context_data.get('portfolios', {})
            if not portfolios:
                return 0.0
            
            total_exposure = 0.0
            portfolio_count = 0
            
            for portfolio_data in portfolios.values():
                holdings = portfolio_data.get('holdings', {})
                portfolio_exposure = 0.0
                
                for entity in event.affected_entities:
                    if entity in holdings:
                        portfolio_exposure += holdings[entity]
                
                total_exposure += portfolio_exposure
                portfolio_count += 1
            
            return total_exposure / portfolio_count if portfolio_count > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculating portfolio exposure: {str(e)}")
            return 0.0
    
    def _calculate_analysis_confidence(self, event: Event, impact_assessment: Dict[str, float]) -> float:
        """Calculate confidence level for the analysis"""
        
        try:
            # Base confidence from event confidence
            confidence = event.confidence_score
            
            # Adjust based on data availability
            if event.affected_entities:
                confidence += 0.1
            
            if event.raw_data:
                confidence += 0.1
            
            # Adjust based on event type
            if event.event_type in [EventType.MARKET_EVENT, EventType.ECONOMIC_EVENT]:
                confidence += 0.1
            
            # Adjust based on impact consistency
            impact_values = list(impact_assessment.values())
            if impact_values:
                impact_std = np.std(impact_values)
                if impact_std < 0.2:  # Consistent impact across dimensions
                    confidence += 0.1
            
            return min(confidence, 1.0)
            
        except Exception as e:
            logger.error(f"Error calculating analysis confidence: {str(e)}")
            return 0.5
    
    def _default_analysis(self, event_id: str) -> EventAnalysis:
        """Return default analysis when analysis fails"""
        return EventAnalysis(
            event_id=event_id,
            analysis_timestamp=datetime.now(),
            sentiment_score=0.0,
            impact_assessment={'overall_impact': 0.5},
            affected_portfolios=[],
            affected_clients=[],
            market_implications={},
            recommended_actions=['Monitor situation for developments'],
            risk_factors={'general_risk': 0.5},
            correlation_analysis={},
            historical_precedents=[],
            confidence_level=0.3
        )

class IntelligentEventProcessor:
    """
    Intelligent Event Processing System
    
    Orchestrates the entire event processing pipeline from detection through
    analysis to response generation and execution.
    """
    
    def __init__(self):
        self.detectors = []
        self.analyzer = EventAnalyzer()
        self.event_queue = deque()
        self.processed_events = {}
        self.active_events = {}
        self.event_history = []
        
        # Processing configuration
        self.max_queue_size = 1000
        self.processing_threads = 4
        self.batch_size = 10
        
        # Initialize detectors
        self._initialize_detectors()
        
        # Start processing thread
        self.processing_active = False
        self.processing_thread = None
    
    def _initialize_detectors(self):
        """Initialize event detectors"""
        self.detectors = [
            NewsEventDetector(),
            MarketEventDetector(),
            WeatherEventDetector()
        ]
        
        logger.info(f"Initialized {len(self.detectors)} event detectors")
    
    def start_processing(self):
        """Start the event processing system"""
        if not self.processing_active:
            self.processing_active = True
            self.processing_thread = threading.Thread(target=self._processing_loop)
            self.processing_thread.daemon = True
            self.processing_thread.start()
            logger.info("Event processing system started")
    
    def stop_processing(self):
        """Stop the event processing system"""
        self.processing_active = False
        if self.processing_thread:
            self.processing_thread.join(timeout=5)
        logger.info("Event processing system stopped")
    
    def process_data_batch(self, data_batch: Dict[str, Any]) -> List[Event]:
        """
        Process a batch of data for event detection
        
        Args:
            data_batch: Dictionary containing various data sources
            
        Returns:
            List of detected events
        """
        detected_events = []
        
        try:
            # Run all detectors on the data
            for detector in self.detectors:
                try:
                    events = detector.detect_events(data_batch)
                    detected_events.extend(events)
                    logger.debug(f"{detector.get_detector_name()} detected {len(events)} events")
                except Exception as e:
                    logger.error(f"Error in {detector.get_detector_name()}: {str(e)}")
            
            # Add events to processing queue
            for event in detected_events:
                if len(self.event_queue) < self.max_queue_size:
                    self.event_queue.append(event)
                    logger.debug(f"Added event {event.event_id} to processing queue")
                else:
                    logger.warning("Event queue full, dropping event")
            
            return detected_events
            
        except Exception as e:
            logger.error(f"Error processing data batch: {str(e)}")
            return []
    
    def process_single_event(
        self,
        event: Event,
        context_data: Dict[str, Any]
    ) -> Tuple[EventAnalysis, EventResponse]:
        """
        Process a single event through the complete pipeline
        
        Args:
            event: Event to process
            context_data: Context data for analysis
            
        Returns:
            Tuple of event analysis and response
        """
        try:
            # Update event status
            event.status = EventStatus.PROCESSING
            event.processing_history.append({
                'timestamp': datetime.now(),
                'status': EventStatus.PROCESSING,
                'processor': 'IntelligentEventProcessor'
            })
            
            # Analyze event
            analysis = self.analyzer.analyze_event(event, context_data)
            
            # Generate response
            response = self._generate_response(event, analysis, context_data)
            
            # Update event status
            event.status = EventStatus.COMPLETED
            event.processing_history.append({
                'timestamp': datetime.now(),
                'status': EventStatus.COMPLETED,
                'processor': 'IntelligentEventProcessor'
            })
            
            # Store processed event
            self.processed_events[event.event_id] = {
                'event': event,
                'analysis': analysis,
                'response': response
            }
            
            # Add to history
            self.event_history.append({
                'event_id': event.event_id,
                'timestamp': event.timestamp,
                'type': event.event_type.value,
                'severity': event.severity.value,
                'impact_score': event.impact_score
            })
            
            logger.info(f"Successfully processed event {event.event_id}")
            return analysis, response
            
        except Exception as e:
            logger.error(f"Error processing event {event.event_id}: {str(e)}")
            
            # Update event status to error
            event.status = EventStatus.ERROR
            event.processing_history.append({
                'timestamp': datetime.now(),
                'status': EventStatus.ERROR,
                'processor': 'IntelligentEventProcessor',
                'error': str(e)
            })
            
            # Return default analysis and response
            return self.analyzer._default_analysis(event.event_id), self._default_response(event.event_id)
    
    def _processing_loop(self):
        """Main processing loop for events"""
        while self.processing_active:
            try:
                if self.event_queue:
                    # Process events in batches
                    batch = []
                    for _ in range(min(self.batch_size, len(self.event_queue))):
                        if self.event_queue:
                            batch.append(self.event_queue.popleft())
                    
                    if batch:
                        self._process_event_batch(batch)
                else:
                    # No events to process, sleep briefly
                    time.sleep(1)
                    
            except Exception as e:
                logger.error(f"Error in processing loop: {str(e)}")
                time.sleep(5)  # Wait before retrying
    
    def _process_event_batch(self, events: List[Event]):
        """Process a batch of events"""
        try:
            # Use thread pool for parallel processing
            with ThreadPoolExecutor(max_workers=self.processing_threads) as executor:
                # Submit processing tasks
                future_to_event = {
                    executor.submit(self._process_event_with_context, event): event
                    for event in events
                }
                
                # Collect results
                for future in as_completed(future_to_event):
                    event = future_to_event[future]
                    try:
                        analysis, response = future.result()
                        logger.debug(f"Processed event {event.event_id} in batch")
                    except Exception as e:
                        logger.error(f"Error processing event {event.event_id} in batch: {str(e)}")
                        
        except Exception as e:
            logger.error(f"Error processing event batch: {str(e)}")
    
    def _process_event_with_context(self, event: Event) -> Tuple[EventAnalysis, EventResponse]:
        """Process event with default context"""
        # In practice, this would fetch real context data
        default_context = {
            'portfolios': {},
            'clients': {},
            'market_data': {},
            'current_positions': {}
        }
        
        return self.process_single_event(event, default_context)
    
    def _generate_response(
        self,
        event: Event,
        analysis: EventAnalysis,
        context_data: Dict[str, Any]
    ) -> EventResponse:
        """Generate response actions based on event analysis"""
        
        try:
            actions_taken = []
            notifications_sent = []
            portfolio_adjustments = []
            client_communications = []
            compliance_checks = []
            follow_up_required = False
            
            # Determine actions based on severity and impact
            max_impact = max(analysis.impact_assessment.values()) if analysis.impact_assessment else 0
            
            # High impact actions
            if max_impact > 0.7 or event.severity >= EventSeverity.HIGH:
                actions_taken.append("Triggered high-impact event protocol")
                notifications_sent.append("Alert sent to risk management team")
                follow_up_required = True
                
                # Portfolio adjustments for high impact events
                if analysis.affected_portfolios:
                    for portfolio_id in analysis.affected_portfolios[:5]:  # Limit to top 5
                        portfolio_adjustments.append({
                            'portfolio_id': portfolio_id,
                            'action': 'risk_assessment',
                            'priority': 'high',
                            'timestamp': datetime.now()
                        })
            
            # Medium impact actions
            elif max_impact > 0.4 or event.severity >= EventSeverity.MEDIUM:
                actions_taken.append("Triggered medium-impact event protocol")
                notifications_sent.append("Alert sent to portfolio managers")
                
                # Portfolio monitoring for medium impact events
                if analysis.affected_portfolios:
                    for portfolio_id in analysis.affected_portfolios[:3]:
                        portfolio_adjustments.append({
                            'portfolio_id': portfolio_id,
                            'action': 'enhanced_monitoring',
                            'priority': 'medium',
                            'timestamp': datetime.now()
                        })
            
            # Client communications
            if analysis.affected_clients and max_impact > 0.5:
                for client_id in analysis.affected_clients[:10]:  # Limit communications
                    communication_type = 'urgent' if max_impact > 0.7 else 'informational'
                    client_communications.append({
                        'client_id': client_id,
                        'type': communication_type,
                        'subject': f"Market Event Update: {event.title}",
                        'priority': 'high' if max_impact > 0.7 else 'medium',
                        'timestamp': datetime.now()
                    })
            
            # Compliance checks
            if event.event_type == EventType.COMPLIANCE_EVENT:
                compliance_checks.append("Regulatory compliance review initiated")
                follow_up_required = True
            
            if max_impact > 0.6:
                compliance_checks.append("Position limit compliance check")
                compliance_checks.append("Risk limit compliance check")
            
            # Market event specific actions
            if event.event_type == EventType.MARKET_EVENT:
                actions_taken.append("Market volatility monitoring activated")
                if max_impact > 0.5:
                    actions_taken.append("Liquidity assessment initiated")
            
            # News event specific actions
            if event.event_type == EventType.NEWS_EVENT:
                actions_taken.append("News sentiment monitoring activated")
                if analysis.sentiment_score < -0.5:
                    actions_taken.append("Negative sentiment alert generated")
            
            response = EventResponse(
                event_id=event.event_id,
                response_timestamp=datetime.now(),
                actions_taken=actions_taken,
                notifications_sent=notifications_sent,
                portfolio_adjustments=portfolio_adjustments,
                client_communications=client_communications,
                compliance_checks=compliance_checks,
                follow_up_required=follow_up_required
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Error generating response for event {event.event_id}: {str(e)}")
            return self._default_response(event.event_id)
    
    def _default_response(self, event_id: str) -> EventResponse:
        """Generate default response when response generation fails"""
        return EventResponse(
            event_id=event_id,
            response_timestamp=datetime.now(),
            actions_taken=['Default monitoring activated'],
            notifications_sent=[],
            portfolio_adjustments=[],
            client_communications=[],
            compliance_checks=[],
            follow_up_required=False
        )
    
    def get_processing_status(self) -> Dict[str, Any]:
        """Get current processing status"""
        return {
            'processing_active': self.processing_active,
            'queue_size': len(self.event_queue),
            'processed_events_count': len(self.processed_events),
            'active_events_count': len(self.active_events),
            'detectors_count': len(self.detectors),
            'processing_threads': self.processing_threads
        }
    
    def get_event_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent event history"""
        return self.event_history[-limit:] if self.event_history else []
    
    def get_processed_event(self, event_id: str) -> Optional[Dict[str, Any]]:
        """Get processed event details"""
        return self.processed_events.get(event_id)

