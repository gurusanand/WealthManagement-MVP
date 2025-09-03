import asyncio
import json
import requests
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from .base_agent import BaseAgent, AgentType, AgentMessage, AgentResponse, AgentCapability
from src.models.event import Event, EventStatus
from src.models.client import Client
from src.models.portfolio import Portfolio, Holding
from src.models.external_data import MarketData, NewsData
from src.models.user import db
import logging

logger = logging.getLogger(__name__)

class EnricherAgent(BaseAgent):
    """
    Enricher Agent - Responsible for enriching events with additional context and information
    
    Capabilities:
    1. Analyze event impact on specific client portfolios
    2. Gather additional market data and context
    3. Perform sentiment analysis on related news
    4. Calculate potential financial impact
    5. Identify related events and patterns
    6. Enrich event data with comprehensive context
    """
    
    def __init__(self, agent_id: str = "enricher_001"):
        capabilities = [
            AgentCapability(
                name="event_enrichment",
                description="Enrich events with additional context and analysis",
                input_schema={
                    "type": "object",
                    "properties": {
                        "event_id": {"type": "string"},
                        "enrichment_type": {"type": "string"}
                    }
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "enriched_data": {"type": "object"},
                        "confidence": {"type": "number"}
                    }
                }
            ),
            AgentCapability(
                name="portfolio_impact_analysis",
                description="Analyze how events impact specific portfolios",
                input_schema={
                    "type": "object",
                    "properties": {
                        "event_id": {"type": "string"},
                        "portfolio_id": {"type": "string"}
                    }
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "impact_analysis": {"type": "object"},
                        "affected_holdings": {"type": "array"}
                    }
                }
            ),
            AgentCapability(
                name="sentiment_analysis",
                description="Perform advanced sentiment analysis on event-related content",
                input_schema={
                    "type": "object",
                    "properties": {
                        "content": {"type": "string"},
                        "context": {"type": "object"}
                    }
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "sentiment": {"type": "object"},
                        "key_themes": {"type": "array"}
                    }
                }
            )
        ]
        
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.ENRICHER,
            name="Enricher Agent",
            description="Enriches events with additional context, analysis, and impact assessment",
            capabilities=capabilities
        )
    
    def get_system_prompt(self) -> str:
        return """You are the Enricher Agent in a wealth management system. Your role is to enrich events with comprehensive context and analysis.

Your responsibilities:
1. Analyze events in the context of specific client portfolios
2. Gather additional relevant information from multiple sources
3. Perform sophisticated sentiment analysis
4. Calculate potential financial impacts
5. Identify correlations and patterns
6. Provide comprehensive risk assessments

You have access to:
- Client portfolio data and holdings
- Historical market data and trends
- News and sentiment data
- Economic indicators
- Weather and environmental data

Always provide detailed, actionable analysis with confidence scores and clear reasoning."""
    
    async def process_message(self, message: AgentMessage) -> AgentResponse:
        """Process incoming messages and route to appropriate handlers"""
        try:
            action = message.content.get("action")
            
            if action == "ping":
                return AgentResponse(success=True, data={"status": "pong"})
            
            elif action == "enrich_event":
                return await self._enrich_event(message.content)
            
            elif action == "analyze_portfolio_impact":
                return await self._analyze_portfolio_impact(message.content)
            
            elif action == "perform_sentiment_analysis":
                return await self._perform_sentiment_analysis(message.content)
            
            elif action == "gather_market_context":
                return await self._gather_market_context(message.content)
            
            elif action == "identify_correlations":
                return await self._identify_correlations(message.content)
            
            else:
                return AgentResponse(
                    success=False,
                    error=f"Unknown action: {action}"
                )
                
        except Exception as e:
            logger.error(f"Enricher agent error: {str(e)}")
            return AgentResponse(success=False, error=str(e))
    
    async def _enrich_event(self, content: Dict[str, Any]) -> AgentResponse:
        """Main event enrichment orchestrator"""
        try:
            event_id = content.get("event_id")
            enrichment_types = content.get("enrichment_types", ["all"])
            
            # Get the event from database
            event = Event.query.get(event_id)
            if not event:
                return AgentResponse(success=False, error="Event not found")
            
            enriched_data = {}
            
            # Perform different types of enrichment
            if "all" in enrichment_types or "market_context" in enrichment_types:
                market_context = await self._gather_market_context_for_event(event)
                enriched_data["market_context"] = market_context
            
            if "all" in enrichment_types or "sentiment" in enrichment_types:
                sentiment_analysis = await self._analyze_event_sentiment(event)
                enriched_data["sentiment_analysis"] = sentiment_analysis
            
            if "all" in enrichment_types or "impact" in enrichment_types:
                impact_analysis = await self._analyze_broad_impact(event)
                enriched_data["impact_analysis"] = impact_analysis
            
            if "all" in enrichment_types or "correlations" in enrichment_types:
                correlations = await self._find_event_correlations(event)
                enriched_data["correlations"] = correlations
            
            # Update event with enriched data
            if event.event_data:
                event.event_data.update(enriched_data)
            else:
                event.event_data = enriched_data
            
            event.status = EventStatus.ENRICHED
            event.updated_at = datetime.utcnow()
            
            db.session.commit()
            
            return AgentResponse(
                success=True,
                data={
                    "event_id": event_id,
                    "enriched_data": enriched_data,
                    "enrichment_types": enrichment_types
                }
            )
            
        except Exception as e:
            logger.error(f"Event enrichment error: {str(e)}")
            db.session.rollback()
            return AgentResponse(success=False, error=str(e))
    
    async def _gather_market_context_for_event(self, event: Event) -> Dict[str, Any]:
        """Gather comprehensive market context for an event"""
        try:
            context = {
                "related_symbols": event.related_symbols or [],
                "market_conditions": {},
                "sector_performance": {},
                "volatility_analysis": {}
            }
            
            # Analyze related symbols
            if event.related_symbols:
                for symbol in event.related_symbols[:5]:  # Limit to 5 symbols
                    symbol_context = await self._analyze_symbol_context(symbol)
                    context["market_conditions"][symbol] = symbol_context
            
            # Get broader market context using LLM
            prompt = f"""
            Analyze the market context for this event:
            
            Event: {event.title}
            Description: {event.description}
            Type: {event.event_type.value if event.event_type else 'Unknown'}
            Related Symbols: {event.related_symbols}
            
            Provide analysis on:
            1. Current market conditions
            2. Sector implications
            3. Historical precedents
            4. Potential market reactions
            5. Key risk factors
            
            Return as JSON with this structure:
            {{
                "market_sentiment": "bullish/bearish/neutral",
                "sector_implications": ["sector1", "sector2"],
                "historical_precedents": "description",
                "potential_reactions": ["reaction1", "reaction2"],
                "risk_factors": ["risk1", "risk2"],
                "confidence": 0.85
            }}
            """
            
            llm_analysis = await self.call_llm_with_json(prompt)
            context.update(llm_analysis)
            
            return context
            
        except Exception as e:
            logger.error(f"Market context gathering error: {str(e)}")
            return {"error": str(e)}
    
    async def _analyze_symbol_context(self, symbol: str) -> Dict[str, Any]:
        """Analyze context for a specific symbol"""
        try:
            # Get recent market data
            recent_data = MarketData.query.filter(
                MarketData.symbol == symbol
            ).order_by(MarketData.data_date.desc()).limit(30).all()
            
            if not recent_data:
                return {"error": "No market data available"}
            
            # Calculate basic metrics
            prices = [float(data.close_price) for data in recent_data if data.close_price]
            volumes = [int(data.volume) for data in recent_data if data.volume]
            
            if prices:
                current_price = prices[0]
                price_change_30d = (prices[0] - prices[-1]) / prices[-1] if len(prices) > 1 else 0
                avg_volume = sum(volumes) / len(volumes) if volumes else 0
                
                # Calculate volatility (simplified)
                if len(prices) > 1:
                    returns = [(prices[i] - prices[i+1]) / prices[i+1] for i in range(len(prices)-1)]
                    volatility = (sum([r**2 for r in returns]) / len(returns)) ** 0.5
                else:
                    volatility = 0
                
                return {
                    "current_price": current_price,
                    "price_change_30d": price_change_30d,
                    "volatility": volatility,
                    "avg_volume": avg_volume,
                    "data_points": len(recent_data)
                }
            
            return {"error": "Insufficient price data"}
            
        except Exception as e:
            logger.error(f"Symbol context analysis error: {str(e)}")
            return {"error": str(e)}
    
    async def _analyze_event_sentiment(self, event: Event) -> Dict[str, Any]:
        """Perform advanced sentiment analysis on event"""
        try:
            # Combine event information for analysis
            content = f"{event.title}\n{event.description}"
            
            prompt = f"""
            Perform comprehensive sentiment analysis on this financial event:
            
            Content: {content}
            Event Type: {event.event_type.value if event.event_type else 'Unknown'}
            Related Symbols: {event.related_symbols}
            
            Analyze:
            1. Overall sentiment (positive/negative/neutral with score -1 to 1)
            2. Market sentiment implications
            3. Key emotional themes
            4. Confidence level in analysis
            5. Potential sentiment shifts over time
            
            Return as JSON:
            {{
                "overall_sentiment": {{
                    "score": 0.3,
                    "label": "positive",
                    "confidence": 0.85
                }},
                "market_implications": {{
                    "short_term": "positive/negative/neutral",
                    "long_term": "positive/negative/neutral"
                }},
                "key_themes": ["theme1", "theme2"],
                "emotional_indicators": ["fear", "optimism", "uncertainty"],
                "sentiment_drivers": ["driver1", "driver2"]
            }}
            """
            
            sentiment_analysis = await self.call_llm_with_json(prompt)
            
            # Update event sentiment if not already set
            if not event.sentiment_score and sentiment_analysis.get("overall_sentiment"):
                event.sentiment_score = sentiment_analysis["overall_sentiment"].get("score")
                event.sentiment_label = sentiment_analysis["overall_sentiment"].get("label")
            
            return sentiment_analysis
            
        except Exception as e:
            logger.error(f"Sentiment analysis error: {str(e)}")
            return {"error": str(e)}
    
    async def _analyze_broad_impact(self, event: Event) -> Dict[str, Any]:
        """Analyze broad impact of event across different dimensions"""
        try:
            prompt = f"""
            Analyze the broad impact of this financial event:
            
            Event: {event.title}
            Description: {event.description}
            Type: {event.event_type.value if event.event_type else 'Unknown'}
            Severity: {event.severity.value if event.severity else 'Unknown'}
            Related Symbols: {event.related_symbols}
            
            Assess impact across:
            1. Asset classes (equities, bonds, commodities, currencies)
            2. Geographic regions
            3. Industry sectors
            4. Time horizons (short, medium, long term)
            5. Risk factors
            
            Return as JSON:
            {{
                "asset_class_impact": {{
                    "equities": {{"impact": "positive/negative/neutral", "magnitude": 0.3}},
                    "bonds": {{"impact": "positive/negative/neutral", "magnitude": 0.1}},
                    "commodities": {{"impact": "positive/negative/neutral", "magnitude": 0.2}},
                    "currencies": {{"impact": "positive/negative/neutral", "magnitude": 0.1}}
                }},
                "sector_impact": {{
                    "technology": 0.2,
                    "healthcare": -0.1,
                    "energy": 0.3
                }},
                "geographic_impact": {{
                    "US": 0.2,
                    "Europe": 0.1,
                    "Asia": 0.0
                }},
                "time_horizon": {{
                    "short_term": "high/medium/low",
                    "medium_term": "high/medium/low",
                    "long_term": "high/medium/low"
                }},
                "confidence": 0.8
            }}
            """
            
            impact_analysis = await self.call_llm_with_json(prompt)
            return impact_analysis
            
        except Exception as e:
            logger.error(f"Impact analysis error: {str(e)}")
            return {"error": str(e)}
    
    async def _find_event_correlations(self, event: Event) -> Dict[str, Any]:
        """Find correlations with other events and patterns"""
        try:
            # Look for similar events in the past
            similar_events = Event.query.filter(
                Event.event_type == event.event_type,
                Event.event_category == event.event_category,
                Event.id != event.id,
                Event.created_at >= datetime.utcnow() - timedelta(days=365)
            ).limit(10).all()
            
            correlations = {
                "similar_events": [],
                "patterns": [],
                "frequency_analysis": {}
            }
            
            for similar_event in similar_events:
                correlations["similar_events"].append({
                    "event_id": similar_event.id,
                    "title": similar_event.title,
                    "date": similar_event.created_at.isoformat(),
                    "severity": similar_event.severity.value if similar_event.severity else None,
                    "related_symbols": similar_event.related_symbols
                })
            
            # Analyze patterns using LLM
            if similar_events:
                events_summary = "\n".join([
                    f"- {e.title} ({e.created_at.strftime('%Y-%m-%d')})"
                    for e in similar_events[:5]
                ])
                
                prompt = f"""
                Analyze patterns in these similar events:
                
                Current Event: {event.title}
                Similar Events:
                {events_summary}
                
                Identify:
                1. Recurring patterns
                2. Seasonal trends
                3. Market cycle correlations
                4. Predictive indicators
                
                Return as JSON:
                {{
                    "patterns": ["pattern1", "pattern2"],
                    "seasonal_trends": "description",
                    "cycle_correlations": "description",
                    "predictive_value": 0.7
                }}
                """
                
                pattern_analysis = await self.call_llm_with_json(prompt)
                correlations.update(pattern_analysis)
            
            return correlations
            
        except Exception as e:
            logger.error(f"Correlation analysis error: {str(e)}")
            return {"error": str(e)}
    
    async def _analyze_portfolio_impact(self, content: Dict[str, Any]) -> AgentResponse:
        """Analyze how an event impacts a specific portfolio"""
        try:
            event_id = content.get("event_id")
            portfolio_id = content.get("portfolio_id")
            
            event = Event.query.get(event_id)
            portfolio = Portfolio.query.get(portfolio_id)
            
            if not event or not portfolio:
                return AgentResponse(success=False, error="Event or portfolio not found")
            
            # Analyze impact on portfolio holdings
            affected_holdings = []
            total_exposure = 0
            
            for holding in portfolio.holdings:
                exposure = await self._calculate_holding_exposure(holding, event)
                if exposure["impact_score"] != 0:
                    affected_holdings.append({
                        "holding_id": holding.id,
                        "symbol": holding.symbol,
                        "exposure": exposure,
                        "market_value": float(holding.market_value or 0)
                    })
                    total_exposure += abs(exposure["impact_score"]) * float(holding.market_value or 0)
            
            # Calculate overall portfolio impact
            portfolio_value = float(portfolio.total_value or 0)
            impact_percentage = (total_exposure / portfolio_value) if portfolio_value > 0 else 0
            
            impact_analysis = {
                "portfolio_id": portfolio_id,
                "event_id": event_id,
                "total_exposure": total_exposure,
                "impact_percentage": impact_percentage,
                "affected_holdings_count": len(affected_holdings),
                "risk_level": self._assess_risk_level(impact_percentage),
                "recommendations": await self._generate_impact_recommendations(
                    event, portfolio, affected_holdings, impact_percentage
                )
            }
            
            return AgentResponse(
                success=True,
                data={
                    "impact_analysis": impact_analysis,
                    "affected_holdings": affected_holdings
                }
            )
            
        except Exception as e:
            logger.error(f"Portfolio impact analysis error: {str(e)}")
            return AgentResponse(success=False, error=str(e))
    
    async def _calculate_holding_exposure(self, holding: Holding, event: Event) -> Dict[str, Any]:
        """Calculate how much a holding is exposed to an event"""
        try:
            impact_score = 0
            exposure_factors = []
            
            # Direct symbol match
            if event.related_symbols and holding.symbol in event.related_symbols:
                impact_score += 0.8
                exposure_factors.append("direct_symbol_match")
            
            # Sector match
            if event.affected_sectors and holding.sector in event.affected_sectors:
                impact_score += 0.4
                exposure_factors.append("sector_exposure")
            
            # Asset class considerations
            if event.event_type and event.event_type.value == "market_event":
                if holding.asset_class and holding.asset_class.value == "Equity":
                    impact_score += 0.2
                    exposure_factors.append("equity_market_exposure")
            
            # Apply event sentiment
            if event.sentiment_score:
                impact_score *= abs(event.sentiment_score)
            
            return {
                "impact_score": min(impact_score, 1.0),  # Cap at 1.0
                "exposure_factors": exposure_factors,
                "confidence": 0.7 if exposure_factors else 0.1
            }
            
        except Exception as e:
            logger.error(f"Holding exposure calculation error: {str(e)}")
            return {"impact_score": 0, "exposure_factors": [], "confidence": 0}
    
    def _assess_risk_level(self, impact_percentage: float) -> str:
        """Assess risk level based on impact percentage"""
        if impact_percentage >= 0.2:  # 20% or more
            return "High"
        elif impact_percentage >= 0.1:  # 10-20%
            return "Medium"
        elif impact_percentage >= 0.05:  # 5-10%
            return "Low"
        else:
            return "Minimal"
    
    async def _generate_impact_recommendations(
        self,
        event: Event,
        portfolio: Portfolio,
        affected_holdings: List[Dict],
        impact_percentage: float
    ) -> List[str]:
        """Generate recommendations based on impact analysis"""
        try:
            prompt = f"""
            Generate investment recommendations based on this event impact analysis:
            
            Event: {event.title}
            Event Type: {event.event_type.value if event.event_type else 'Unknown'}
            Portfolio Impact: {impact_percentage:.2%}
            Affected Holdings: {len(affected_holdings)}
            
            Holdings Details:
            {json.dumps(affected_holdings[:3], indent=2)}
            
            Provide 3-5 specific, actionable recommendations considering:
            1. Risk management
            2. Opportunity identification
            3. Portfolio rebalancing
            4. Timing considerations
            
            Return as JSON array of strings:
            ["recommendation1", "recommendation2", "recommendation3"]
            """
            
            recommendations = await self.call_llm_with_json(prompt)
            return recommendations if isinstance(recommendations, list) else []
            
        except Exception as e:
            logger.error(f"Recommendation generation error: {str(e)}")
            return ["Monitor situation closely", "Consider risk management measures"]
    
    async def _perform_sentiment_analysis(self, content: Dict[str, Any]) -> AgentResponse:
        """Perform standalone sentiment analysis"""
        try:
            text_content = content.get("content", "")
            context = content.get("context", {})
            
            sentiment_result = await self._analyze_text_sentiment(text_content, context)
            
            return AgentResponse(
                success=True,
                data=sentiment_result
            )
            
        except Exception as e:
            return AgentResponse(success=False, error=str(e))
    
    async def _analyze_text_sentiment(self, text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze sentiment of arbitrary text"""
        try:
            prompt = f"""
            Perform sentiment analysis on this text:
            
            Text: {text}
            Context: {json.dumps(context)}
            
            Provide:
            1. Overall sentiment score (-1 to 1)
            2. Sentiment label (positive/negative/neutral)
            3. Confidence level (0 to 1)
            4. Key emotional themes
            5. Sentiment drivers
            
            Return as JSON:
            {{
                "sentiment_score": 0.3,
                "sentiment_label": "positive",
                "confidence": 0.85,
                "themes": ["optimism", "growth"],
                "drivers": ["strong earnings", "market expansion"]
            }}
            """
            
            return await self.call_llm_with_json(prompt)
            
        except Exception as e:
            logger.error(f"Text sentiment analysis error: {str(e)}")
            return {"error": str(e)}
    
    async def _gather_market_context(self, content: Dict[str, Any]) -> AgentResponse:
        """Gather market context for analysis"""
        try:
            symbols = content.get("symbols", [])
            timeframe = content.get("timeframe", "30d")
            
            market_context = {}
            
            for symbol in symbols[:10]:  # Limit to 10 symbols
                context = await self._analyze_symbol_context(symbol)
                market_context[symbol] = context
            
            return AgentResponse(
                success=True,
                data={
                    "market_context": market_context,
                    "timeframe": timeframe
                }
            )
            
        except Exception as e:
            return AgentResponse(success=False, error=str(e))
    
    async def _identify_correlations(self, content: Dict[str, Any]) -> AgentResponse:
        """Identify correlations and patterns"""
        try:
            event_id = content.get("event_id")
            
            if event_id:
                event = Event.query.get(event_id)
                if event:
                    correlations = await self._find_event_correlations(event)
                    return AgentResponse(success=True, data=correlations)
            
            return AgentResponse(success=False, error="Event not found")
            
        except Exception as e:
            return AgentResponse(success=False, error=str(e))

