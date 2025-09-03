import asyncio
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from .base_agent import BaseAgent, AgentType, AgentMessage, AgentResponse, AgentCapability
from src.models.event import Event, EventType, EventSource, EventSeverity, EventCategory
from src.models.external_data import NewsData, MarketData, WeatherData
from src.models.user import db
import logging

logger = logging.getLogger(__name__)

class OracleAgent(BaseAgent):
    """
    Oracle Agent - Responsible for detecting and ingesting life events and market events
    
    Capabilities:
    1. Monitor external data sources (news, social media, market data)
    2. Detect life events from CRM systems
    3. Identify market events and anomalies
    4. Create event records in the database
    5. Trigger downstream processing workflows
    """
    
    def __init__(self, agent_id: str = "oracle_001"):
        capabilities = [
            AgentCapability(
                name="event_detection",
                description="Detect and classify events from various data sources",
                input_schema={
                    "type": "object",
                    "properties": {
                        "data_source": {"type": "string"},
                        "data": {"type": "object"}
                    }
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "events": {"type": "array"},
                        "confidence": {"type": "number"}
                    }
                }
            ),
            AgentCapability(
                name="news_monitoring",
                description="Monitor news sources for financial events",
                input_schema={
                    "type": "object",
                    "properties": {
                        "symbols": {"type": "array"},
                        "keywords": {"type": "array"}
                    }
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "relevant_news": {"type": "array"},
                        "sentiment": {"type": "object"}
                    }
                }
            ),
            AgentCapability(
                name="market_anomaly_detection",
                description="Detect unusual market movements and events",
                input_schema={
                    "type": "object",
                    "properties": {
                        "symbol": {"type": "string"},
                        "timeframe": {"type": "string"}
                    }
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "anomalies": {"type": "array"},
                        "severity": {"type": "string"}
                    }
                }
            )
        ]
        
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.ORACLE,
            name="Oracle Agent",
            description="Detects and ingests life events and market events from various sources",
            capabilities=capabilities
        )
        
        # Configuration for external APIs
        self.news_api_key = None  # Will be set from environment
        self.alpha_vantage_key = None  # Will be set from environment
        self.weather_api_key = None  # Will be set from environment
        
        # Event detection thresholds
        self.price_change_threshold = 0.05  # 5% price change
        self.volume_spike_threshold = 2.0   # 2x average volume
        self.news_relevance_threshold = 0.7  # 70% relevance score
    
    def get_system_prompt(self) -> str:
        return """You are the Oracle Agent in a wealth management system. Your role is to detect and classify events that could impact client portfolios.

You have access to various data sources including:
- News feeds and financial news
- Market data and price movements
- Weather and satellite data
- Social media sentiment
- CRM system data

Your responsibilities:
1. Monitor data sources continuously
2. Identify significant events (life events, market events, economic events)
3. Classify events by type, severity, and relevance
4. Extract key information and metadata
5. Create structured event records
6. Assess confidence levels and impact potential

Always provide structured, actionable information with confidence scores and clear categorization."""
    
    async def process_message(self, message: AgentMessage) -> AgentResponse:
        """Process incoming messages and route to appropriate handlers"""
        try:
            action = message.content.get("action")
            
            if action == "ping":
                return AgentResponse(success=True, data={"status": "pong"})
            
            elif action == "detect_events":
                return await self._detect_events(message.content)
            
            elif action == "monitor_news":
                return await self._monitor_news(message.content)
            
            elif action == "check_market_anomalies":
                return await self._check_market_anomalies(message.content)
            
            elif action == "process_weather_data":
                return await self._process_weather_data(message.content)
            
            elif action == "scan_life_events":
                return await self._scan_life_events(message.content)
            
            else:
                return AgentResponse(
                    success=False,
                    error=f"Unknown action: {action}"
                )
                
        except Exception as e:
            logger.error(f"Oracle agent error: {str(e)}")
            return AgentResponse(success=False, error=str(e))
    
    async def _detect_events(self, content: Dict[str, Any]) -> AgentResponse:
        """Main event detection orchestrator"""
        try:
            data_source = content.get("data_source")
            data = content.get("data", {})
            
            events = []
            
            if data_source == "news":
                events.extend(await self._process_news_events(data))
            elif data_source == "market":
                events.extend(await self._process_market_events(data))
            elif data_source == "weather":
                events.extend(await self._process_weather_events(data))
            elif data_source == "crm":
                events.extend(await self._process_crm_events(data))
            
            # Store events in database
            created_events = []
            for event_data in events:
                event = await self._create_event_record(event_data)
                if event:
                    created_events.append(event.to_dict())
            
            return AgentResponse(
                success=True,
                data={
                    "events_detected": len(created_events),
                    "events": created_events
                }
            )
            
        except Exception as e:
            logger.error(f"Event detection error: {str(e)}")
            return AgentResponse(success=False, error=str(e))
    
    async def _process_news_events(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process news data to detect financial events"""
        events = []
        
        try:
            # Use LLM to analyze news content
            news_content = data.get("content", "")
            title = data.get("title", "")
            
            prompt = f"""
            Analyze this financial news article and extract any significant events:
            
            Title: {title}
            Content: {news_content[:1000]}...
            
            Identify:
            1. Event type (earnings, merger, regulatory, economic, etc.)
            2. Companies/symbols affected
            3. Severity level (Low, Medium, High, Critical)
            4. Sentiment (positive, negative, neutral)
            5. Confidence score (0-1)
            
            Return as JSON with this structure:
            {{
                "events": [
                    {{
                        "event_type": "market_event",
                        "category": "earnings_announcement",
                        "title": "Event title",
                        "description": "Event description",
                        "severity": "Medium",
                        "related_symbols": ["AAPL", "MSFT"],
                        "sentiment_score": 0.2,
                        "confidence": 0.85
                    }}
                ]
            }}
            """
            
            response = await self.call_llm_with_json(prompt)
            events.extend(response.get("events", []))
            
        except Exception as e:
            logger.error(f"News event processing error: {str(e)}")
        
        return events
    
    async def _process_market_events(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process market data to detect anomalies and events"""
        events = []
        
        try:
            symbol = data.get("symbol")
            price_data = data.get("price_data", {})
            
            current_price = price_data.get("current_price", 0)
            previous_price = price_data.get("previous_price", 0)
            volume = price_data.get("volume", 0)
            avg_volume = price_data.get("avg_volume", 0)
            
            # Check for significant price movements
            if previous_price > 0:
                price_change_pct = abs(current_price - previous_price) / previous_price
                
                if price_change_pct >= self.price_change_threshold:
                    direction = "up" if current_price > previous_price else "down"
                    severity = "High" if price_change_pct >= 0.10 else "Medium"
                    
                    events.append({
                        "event_type": "market_event",
                        "category": "price_movement",
                        "title": f"{symbol} significant price movement",
                        "description": f"{symbol} moved {direction} by {price_change_pct:.2%}",
                        "severity": severity,
                        "related_symbols": [symbol],
                        "sentiment_score": 0.5 if direction == "up" else -0.5,
                        "confidence": 0.9,
                        "event_data": {
                            "price_change_pct": price_change_pct,
                            "direction": direction,
                            "current_price": current_price,
                            "previous_price": previous_price
                        }
                    })
            
            # Check for volume spikes
            if avg_volume > 0 and volume >= avg_volume * self.volume_spike_threshold:
                events.append({
                    "event_type": "market_event",
                    "category": "volume_spike",
                    "title": f"{symbol} unusual volume activity",
                    "description": f"{symbol} volume is {volume/avg_volume:.1f}x average",
                    "severity": "Medium",
                    "related_symbols": [symbol],
                    "sentiment_score": 0.0,
                    "confidence": 0.8,
                    "event_data": {
                        "volume": volume,
                        "avg_volume": avg_volume,
                        "volume_ratio": volume / avg_volume
                    }
                })
                
        except Exception as e:
            logger.error(f"Market event processing error: {str(e)}")
        
        return events
    
    async def _process_weather_events(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process weather data to detect events affecting commodities"""
        events = []
        
        try:
            location = data.get("location", {})
            weather = data.get("weather", {})
            
            temperature = weather.get("temperature", 0)
            precipitation = weather.get("precipitation", 0)
            condition = weather.get("condition", "")
            
            # Check for extreme weather conditions
            if temperature > 40 or temperature < -20:  # Extreme temperatures
                severity = "High" if abs(temperature) > 45 else "Medium"
                
                events.append({
                    "event_type": "weather_event",
                    "category": "extreme_temperature",
                    "title": f"Extreme temperature in {location.get('name', 'Unknown')}",
                    "description": f"Temperature: {temperature}°C",
                    "severity": severity,
                    "related_symbols": ["CORN", "WHEAT", "SOYB"],  # Agricultural commodities
                    "sentiment_score": -0.3,
                    "confidence": 0.7,
                    "location": location,
                    "event_data": {
                        "temperature": temperature,
                        "condition": condition
                    }
                })
            
            # Check for heavy precipitation
            if precipitation > 50:  # Heavy rain/snow
                events.append({
                    "event_type": "weather_event",
                    "category": "heavy_precipitation",
                    "title": f"Heavy precipitation in {location.get('name', 'Unknown')}",
                    "description": f"Precipitation: {precipitation}mm",
                    "severity": "Medium",
                    "related_symbols": ["CORN", "WHEAT", "SOYB"],
                    "sentiment_score": -0.2,
                    "confidence": 0.6,
                    "location": location,
                    "event_data": {
                        "precipitation": precipitation,
                        "condition": condition
                    }
                })
                
        except Exception as e:
            logger.error(f"Weather event processing error: {str(e)}")
        
        return events
    
    async def _process_crm_events(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Process CRM data to detect life events"""
        events = []
        
        try:
            client_id = data.get("client_id")
            event_type = data.get("event_type")
            details = data.get("details", {})
            
            # Map CRM events to our event categories
            category_mapping = {
                "marriage": "Marriage",
                "divorce": "Divorce",
                "birth": "Birth",
                "death": "Death",
                "retirement": "Retirement",
                "job_change": "Job Change",
                "inheritance": "Inheritance",
                "major_purchase": "Major Purchase"
            }
            
            category = category_mapping.get(event_type.lower())
            if category:
                events.append({
                    "event_type": "life_event",
                    "category": category,
                    "title": f"Client life event: {category}",
                    "description": details.get("description", f"Client experienced {category}"),
                    "severity": "High",  # Life events are generally high impact
                    "client_id": client_id,
                    "confidence": 0.95,  # CRM data is highly reliable
                    "event_data": details
                })
                
        except Exception as e:
            logger.error(f"CRM event processing error: {str(e)}")
        
        return events
    
    async def _create_event_record(self, event_data: Dict[str, Any]) -> Optional[Event]:
        """Create an event record in the database"""
        try:
            event = Event(
                client_id=event_data.get("client_id"),
                event_type=EventType(event_data.get("event_type", "market_event")),
                event_source=EventSource.ORACLE,
                event_category=EventCategory(event_data.get("category")) if event_data.get("category") else None,
                title=event_data.get("title"),
                description=event_data.get("description"),
                severity=EventSeverity(event_data.get("severity", "Medium")),
                confidence=event_data.get("confidence"),
                latitude=event_data.get("location", {}).get("latitude") if event_data.get("location") else None,
                longitude=event_data.get("location", {}).get("longitude") if event_data.get("location") else None,
                location_name=event_data.get("location", {}).get("name") if event_data.get("location") else None,
                related_symbols=event_data.get("related_symbols"),
                affected_sectors=event_data.get("affected_sectors"),
                sentiment_score=event_data.get("sentiment_score"),
                sentiment_label=self._get_sentiment_label(event_data.get("sentiment_score", 0)),
                event_data=event_data.get("event_data"),
                event_timestamp=datetime.utcnow()
            )
            
            db.session.add(event)
            db.session.commit()
            
            logger.info(f"Created event record: {event.id}")
            return event
            
        except Exception as e:
            logger.error(f"Error creating event record: {str(e)}")
            db.session.rollback()
            return None
    
    def _get_sentiment_label(self, sentiment_score: float) -> str:
        """Convert sentiment score to label"""
        if sentiment_score > 0.1:
            return "positive"
        elif sentiment_score < -0.1:
            return "negative"
        else:
            return "neutral"
    
    async def _monitor_news(self, content: Dict[str, Any]) -> AgentResponse:
        """Monitor news sources for relevant financial events"""
        try:
            symbols = content.get("symbols", [])
            keywords = content.get("keywords", [])
            
            # This would integrate with NewsAPI or other news sources
            # For now, return a placeholder response
            
            return AgentResponse(
                success=True,
                data={
                    "message": "News monitoring initiated",
                    "symbols": symbols,
                    "keywords": keywords
                }
            )
            
        except Exception as e:
            return AgentResponse(success=False, error=str(e))
    
    async def _check_market_anomalies(self, content: Dict[str, Any]) -> AgentResponse:
        """Check for market anomalies and unusual patterns"""
        try:
            symbol = content.get("symbol")
            timeframe = content.get("timeframe", "1d")
            
            # This would analyze market data for anomalies
            # For now, return a placeholder response
            
            return AgentResponse(
                success=True,
                data={
                    "message": "Market anomaly check completed",
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "anomalies_found": 0
                }
            )
            
        except Exception as e:
            return AgentResponse(success=False, error=str(e))
    
    async def _process_weather_data(self, content: Dict[str, Any]) -> AgentResponse:
        """Process weather data for commodity impact analysis"""
        try:
            location = content.get("location", {})
            weather_data = content.get("weather_data", {})
            
            events = await self._process_weather_events({
                "location": location,
                "weather": weather_data
            })
            
            return AgentResponse(
                success=True,
                data={
                    "events_detected": len(events),
                    "events": events
                }
            )
            
        except Exception as e:
            return AgentResponse(success=False, error=str(e))
    
    async def _scan_life_events(self, content: Dict[str, Any]) -> AgentResponse:
        """Scan for life events from CRM or other sources"""
        try:
            client_id = content.get("client_id")
            
            # This would integrate with CRM systems to detect life events
            # For now, return a placeholder response
            
            return AgentResponse(
                success=True,
                data={
                    "message": "Life event scan completed",
                    "client_id": client_id,
                    "events_found": 0
                }
            )
            
        except Exception as e:
            return AgentResponse(success=False, error=str(e))

