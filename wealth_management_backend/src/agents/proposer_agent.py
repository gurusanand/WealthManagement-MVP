import asyncio
import json
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from .base_agent import BaseAgent, AgentType, AgentMessage, AgentResponse, AgentCapability
from src.models.event import Event, Proposal, EventStatus
from src.models.client import Client
from src.models.portfolio import Portfolio, Holding
from src.models.external_data import MarketData
from src.models.user import db
import logging

logger = logging.getLogger(__name__)

class ProposerAgent(BaseAgent):
    """
    Proposer Agent - Responsible for generating portfolio optimization proposals
    
    Capabilities:
    1. Generate portfolio rebalancing recommendations
    2. Propose new investment opportunities
    3. Suggest divestment strategies
    4. Optimize for different objectives (Sharpe ratio, ESG, tax efficiency)
    5. Create detailed trade execution plans
    6. Assess proposal risk and expected returns
    """
    
    def __init__(self, agent_id: str = "proposer_001"):
        capabilities = [
            AgentCapability(
                name="portfolio_optimization",
                description="Generate optimal portfolio allocation recommendations",
                input_schema={
                    "type": "object",
                    "properties": {
                        "portfolio_id": {"type": "string"},
                        "objective": {"type": "string"},
                        "constraints": {"type": "object"}
                    }
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "proposed_allocation": {"type": "object"},
                        "expected_return": {"type": "number"},
                        "risk_metrics": {"type": "object"}
                    }
                }
            ),
            AgentCapability(
                name="trade_proposal_generation",
                description="Generate specific trade proposals based on events",
                input_schema={
                    "type": "object",
                    "properties": {
                        "event_id": {"type": "string"},
                        "portfolio_id": {"type": "string"},
                        "strategy": {"type": "string"}
                    }
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "trades": {"type": "array"},
                        "rationale": {"type": "string"},
                        "expected_impact": {"type": "object"}
                    }
                }
            ),
            AgentCapability(
                name="risk_return_optimization",
                description="Optimize portfolio for risk-return characteristics",
                input_schema={
                    "type": "object",
                    "properties": {
                        "portfolio_id": {"type": "string"},
                        "risk_tolerance": {"type": "string"},
                        "return_target": {"type": "number"}
                    }
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "optimized_weights": {"type": "object"},
                        "expected_metrics": {"type": "object"}
                    }
                }
            )
        ]
        
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.PROPOSER,
            name="Proposer Agent",
            description="Generates portfolio optimization proposals and trade recommendations",
            capabilities=capabilities
        )
        
        # Optimization parameters
        self.risk_free_rate = 0.02  # 2% risk-free rate
        self.rebalancing_threshold = 0.05  # 5% deviation threshold
        self.max_position_size = 0.10  # 10% maximum position size
        self.min_trade_size = 1000  # Minimum trade size in USD
    
    def get_system_prompt(self) -> str:
        return """You are the Proposer Agent in a wealth management system. Your role is to generate optimal portfolio proposals and trade recommendations.

Your responsibilities:
1. Analyze portfolio performance and risk characteristics
2. Generate rebalancing recommendations based on events and market conditions
3. Propose new investment opportunities aligned with client objectives
4. Optimize portfolios for various objectives (Sharpe ratio, ESG, tax efficiency)
5. Create detailed trade execution plans with rationale
6. Assess expected returns and risk metrics for proposals

You have access to:
- Client risk profiles and investment objectives
- Portfolio holdings and performance data
- Market data and technical indicators
- Event context and impact analysis
- ESG scores and sustainability metrics

Always provide well-reasoned proposals with clear rationale, risk assessment, and expected outcomes."""
    
    async def process_message(self, message: AgentMessage) -> AgentResponse:
        """Process incoming messages and route to appropriate handlers"""
        try:
            action = message.content.get("action")
            
            if action == "ping":
                return AgentResponse(success=True, data={"status": "pong"})
            
            elif action == "generate_proposal":
                return await self._generate_proposal(message.content)
            
            elif action == "optimize_portfolio":
                return await self._optimize_portfolio(message.content)
            
            elif action == "rebalance_recommendation":
                return await self._generate_rebalancing_recommendation(message.content)
            
            elif action == "event_response_proposal":
                return await self._generate_event_response_proposal(message.content)
            
            elif action == "esg_optimization":
                return await self._generate_esg_optimization(message.content)
            
            elif action == "tax_optimization":
                return await self._generate_tax_optimization(message.content)
            
            else:
                return AgentResponse(
                    success=False,
                    error=f"Unknown action: {action}"
                )
                
        except Exception as e:
            logger.error(f"Proposer agent error: {str(e)}")
            return AgentResponse(success=False, error=str(e))
    
    async def _generate_proposal(self, content: Dict[str, Any]) -> AgentResponse:
        """Main proposal generation orchestrator"""
        try:
            event_id = content.get("event_id")
            portfolio_id = content.get("portfolio_id")
            objective = content.get("objective", "SharpeMax")
            
            event = Event.query.get(event_id) if event_id else None
            portfolio = Portfolio.query.get(portfolio_id)
            
            if not portfolio:
                return AgentResponse(success=False, error="Portfolio not found")
            
            # Generate proposal based on objective
            if objective == "SharpeMax":
                proposal_data = await self._generate_sharpe_optimization_proposal(portfolio, event)
            elif objective == "DrawdownMin":
                proposal_data = await self._generate_drawdown_minimization_proposal(portfolio, event)
            elif objective == "ESG":
                proposal_data = await self._generate_esg_proposal(portfolio, event)
            elif objective == "TaxAware":
                proposal_data = await self._generate_tax_aware_proposal(portfolio, event)
            else:
                proposal_data = await self._generate_balanced_proposal(portfolio, event)
            
            # Create proposal record in database
            proposal = await self._create_proposal_record(
                event_id, portfolio_id, objective, proposal_data
            )
            
            if proposal:
                # Update event status
                if event:
                    event.status = EventStatus.PROPOSED
                    event.updated_at = datetime.utcnow()
                    db.session.commit()
                
                return AgentResponse(
                    success=True,
                    data={
                        "proposal_id": proposal.id,
                        "proposal_data": proposal_data
                    }
                )
            else:
                return AgentResponse(success=False, error="Failed to create proposal")
                
        except Exception as e:
            logger.error(f"Proposal generation error: {str(e)}")
            return AgentResponse(success=False, error=str(e))
    
    async def _generate_sharpe_optimization_proposal(
        self, 
        portfolio: Portfolio, 
        event: Optional[Event] = None
    ) -> Dict[str, Any]:
        """Generate proposal optimized for Sharpe ratio"""
        try:
            # Get current holdings and their performance data
            holdings_data = await self._get_holdings_performance_data(portfolio)
            
            # Calculate current portfolio metrics
            current_metrics = await self._calculate_portfolio_metrics(holdings_data)
            
            # Generate optimization using LLM with financial reasoning
            prompt = f"""
            Generate a Sharpe ratio optimization proposal for this portfolio:
            
            Current Holdings:
            {json.dumps(holdings_data, indent=2)}
            
            Current Metrics:
            - Expected Return: {current_metrics.get('expected_return', 0):.2%}
            - Volatility: {current_metrics.get('volatility', 0):.2%}
            - Sharpe Ratio: {current_metrics.get('sharpe_ratio', 0):.2f}
            
            Event Context: {event.title if event else 'No specific event'}
            
            Provide optimization recommendations considering:
            1. Risk-return efficiency
            2. Diversification benefits
            3. Market conditions
            4. Transaction costs
            
            Return as JSON:
            {{
                "proposed_trades": [
                    {{
                        "symbol": "AAPL",
                        "action": "buy/sell",
                        "quantity": 100,
                        "rationale": "reasoning"
                    }}
                ],
                "expected_impact": {{
                    "expected_return": 0.08,
                    "expected_volatility": 0.12,
                    "expected_sharpe": 0.5,
                    "confidence": 0.75
                }},
                "rationale": "Overall reasoning for the proposal",
                "risk_factors": ["factor1", "factor2"]
            }}
            """
            
            optimization_result = await self.call_llm_with_json(prompt)
            
            # Add additional analysis
            optimization_result["optimization_type"] = "SharpeMax"
            optimization_result["current_metrics"] = current_metrics
            optimization_result["proposal_timestamp"] = datetime.utcnow().isoformat()
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Sharpe optimization error: {str(e)}")
            return {"error": str(e)}
    
    async def _generate_drawdown_minimization_proposal(
        self, 
        portfolio: Portfolio, 
        event: Optional[Event] = None
    ) -> Dict[str, Any]:
        """Generate proposal optimized for minimum drawdown"""
        try:
            holdings_data = await self._get_holdings_performance_data(portfolio)
            current_metrics = await self._calculate_portfolio_metrics(holdings_data)
            
            prompt = f"""
            Generate a drawdown minimization proposal for this portfolio:
            
            Current Holdings: {json.dumps(holdings_data, indent=2)}
            Current Max Drawdown: {current_metrics.get('max_drawdown', 0):.2%}
            
            Event Context: {event.title if event else 'No specific event'}
            
            Focus on:
            1. Reducing portfolio volatility
            2. Adding defensive assets
            3. Improving downside protection
            4. Maintaining reasonable returns
            
            Return optimization proposal as JSON with same structure as Sharpe optimization.
            """
            
            optimization_result = await self.call_llm_with_json(prompt)
            optimization_result["optimization_type"] = "DrawdownMin"
            optimization_result["current_metrics"] = current_metrics
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Drawdown minimization error: {str(e)}")
            return {"error": str(e)}
    
    async def _generate_esg_proposal(
        self, 
        portfolio: Portfolio, 
        event: Optional[Event] = None
    ) -> Dict[str, Any]:
        """Generate ESG-focused proposal"""
        try:
            holdings_data = await self._get_holdings_performance_data(portfolio)
            
            # Get client ESG preferences
            client = Client.query.get(portfolio.client_id)
            esg_preferences = {
                "environmental": client.esg_environmental if client else False,
                "social": client.esg_social if client else False,
                "governance": client.esg_governance if client else False
            }
            
            prompt = f"""
            Generate an ESG-optimized proposal for this portfolio:
            
            Current Holdings: {json.dumps(holdings_data, indent=2)}
            Client ESG Preferences: {json.dumps(esg_preferences)}
            
            Event Context: {event.title if event else 'No specific event'}
            
            Focus on:
            1. Improving ESG scores while maintaining returns
            2. Aligning with client ESG preferences
            3. Identifying sustainable investment opportunities
            4. Divesting from non-ESG compliant holdings
            
            Return optimization proposal as JSON.
            """
            
            optimization_result = await self.call_llm_with_json(prompt)
            optimization_result["optimization_type"] = "ESG"
            optimization_result["esg_preferences"] = esg_preferences
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"ESG optimization error: {str(e)}")
            return {"error": str(e)}
    
    async def _generate_tax_aware_proposal(
        self, 
        portfolio: Portfolio, 
        event: Optional[Event] = None
    ) -> Dict[str, Any]:
        """Generate tax-optimized proposal"""
        try:
            holdings_data = await self._get_holdings_performance_data(portfolio)
            
            prompt = f"""
            Generate a tax-aware optimization proposal for this portfolio:
            
            Current Holdings: {json.dumps(holdings_data, indent=2)}
            
            Event Context: {event.title if event else 'No specific event'}
            
            Focus on:
            1. Tax-loss harvesting opportunities
            2. Long-term vs short-term capital gains
            3. Tax-efficient fund selections
            4. Asset location optimization
            
            Return optimization proposal as JSON.
            """
            
            optimization_result = await self.call_llm_with_json(prompt)
            optimization_result["optimization_type"] = "TaxAware"
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Tax optimization error: {str(e)}")
            return {"error": str(e)}
    
    async def _generate_balanced_proposal(
        self, 
        portfolio: Portfolio, 
        event: Optional[Event] = None
    ) -> Dict[str, Any]:
        """Generate balanced optimization proposal"""
        try:
            holdings_data = await self._get_holdings_performance_data(portfolio)
            current_metrics = await self._calculate_portfolio_metrics(holdings_data)
            
            prompt = f"""
            Generate a balanced optimization proposal for this portfolio:
            
            Current Holdings: {json.dumps(holdings_data, indent=2)}
            Current Metrics: {json.dumps(current_metrics)}
            
            Event Context: {event.title if event else 'No specific event'}
            
            Balance:
            1. Risk and return optimization
            2. Diversification across asset classes
            3. Cost efficiency
            4. Liquidity considerations
            
            Return optimization proposal as JSON.
            """
            
            optimization_result = await self.call_llm_with_json(prompt)
            optimization_result["optimization_type"] = "Balanced"
            optimization_result["current_metrics"] = current_metrics
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Balanced optimization error: {str(e)}")
            return {"error": str(e)}
    
    async def _get_holdings_performance_data(self, portfolio: Portfolio) -> List[Dict[str, Any]]:
        """Get performance data for all holdings in portfolio"""
        holdings_data = []
        
        for holding in portfolio.holdings:
            # Get recent market data for the holding
            recent_data = MarketData.query.filter(
                MarketData.symbol == holding.symbol
            ).order_by(MarketData.data_date.desc()).limit(30).all()
            
            # Calculate basic performance metrics
            prices = [float(data.close_price) for data in recent_data if data.close_price]
            
            if prices and len(prices) > 1:
                returns = [(prices[i] - prices[i+1]) / prices[i+1] for i in range(len(prices)-1)]
                avg_return = sum(returns) / len(returns) if returns else 0
                volatility = (sum([(r - avg_return)**2 for r in returns]) / len(returns))**0.5 if returns else 0
            else:
                avg_return = 0
                volatility = 0
            
            holding_data = {
                "symbol": holding.symbol,
                "quantity": float(holding.quantity or 0),
                "market_value": float(holding.market_value or 0),
                "weight": float(holding.market_value or 0) / float(portfolio.total_value or 1),
                "sector": holding.sector,
                "asset_class": holding.asset_class.value if holding.asset_class else None,
                "avg_return": avg_return,
                "volatility": volatility,
                "esg_score": float(holding.esg_score or 0),
                "data_points": len(recent_data)
            }
            
            holdings_data.append(holding_data)
        
        return holdings_data
    
    async def _calculate_portfolio_metrics(self, holdings_data: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate current portfolio performance metrics"""
        try:
            if not holdings_data:
                return {}
            
            # Calculate weighted portfolio metrics
            total_weight = sum(h["weight"] for h in holdings_data)
            
            if total_weight == 0:
                return {}
            
            # Weighted average return
            portfolio_return = sum(h["avg_return"] * h["weight"] for h in holdings_data) / total_weight
            
            # Portfolio volatility (simplified - assumes zero correlation)
            portfolio_volatility = (sum((h["volatility"] * h["weight"])**2 for h in holdings_data))**0.5 / total_weight
            
            # Sharpe ratio
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_volatility if portfolio_volatility > 0 else 0
            
            # ESG score
            portfolio_esg = sum(h["esg_score"] * h["weight"] for h in holdings_data) / total_weight
            
            return {
                "expected_return": portfolio_return,
                "volatility": portfolio_volatility,
                "sharpe_ratio": sharpe_ratio,
                "esg_score": portfolio_esg,
                "num_holdings": len(holdings_data),
                "concentration": max(h["weight"] for h in holdings_data) if holdings_data else 0
            }
            
        except Exception as e:
            logger.error(f"Portfolio metrics calculation error: {str(e)}")
            return {}
    
    async def _create_proposal_record(
        self,
        event_id: Optional[str],
        portfolio_id: str,
        objective: str,
        proposal_data: Dict[str, Any]
    ) -> Optional[Proposal]:
        """Create a proposal record in the database"""
        try:
            proposal = Proposal(
                event_id=event_id,
                portfolio_id=portfolio_id,
                proposal_type="optimization",
                objective=objective,
                proposed_trades=proposal_data.get("proposed_trades", []),
                expected_impact=proposal_data.get("expected_impact", {}),
                risk_assessment={
                    "risk_factors": proposal_data.get("risk_factors", []),
                    "confidence": proposal_data.get("expected_impact", {}).get("confidence", 0.5)
                }
            )
            
            db.session.add(proposal)
            db.session.commit()
            
            logger.info(f"Created proposal record: {proposal.id}")
            return proposal
            
        except Exception as e:
            logger.error(f"Error creating proposal record: {str(e)}")
            db.session.rollback()
            return None
    
    async def _optimize_portfolio(self, content: Dict[str, Any]) -> AgentResponse:
        """Optimize portfolio allocation"""
        try:
            portfolio_id = content.get("portfolio_id")
            objective = content.get("objective", "SharpeMax")
            constraints = content.get("constraints", {})
            
            portfolio = Portfolio.query.get(portfolio_id)
            if not portfolio:
                return AgentResponse(success=False, error="Portfolio not found")
            
            # Generate optimization proposal
            if objective == "SharpeMax":
                proposal_data = await self._generate_sharpe_optimization_proposal(portfolio)
            elif objective == "ESG":
                proposal_data = await self._generate_esg_proposal(portfolio)
            else:
                proposal_data = await self._generate_balanced_proposal(portfolio)
            
            return AgentResponse(
                success=True,
                data=proposal_data
            )
            
        except Exception as e:
            return AgentResponse(success=False, error=str(e))
    
    async def _generate_rebalancing_recommendation(self, content: Dict[str, Any]) -> AgentResponse:
        """Generate portfolio rebalancing recommendations"""
        try:
            portfolio_id = content.get("portfolio_id")
            target_allocation = content.get("target_allocation", {})
            
            portfolio = Portfolio.query.get(portfolio_id)
            if not portfolio:
                return AgentResponse(success=False, error="Portfolio not found")
            
            # Calculate current allocation vs target
            current_allocation = {}
            total_value = float(portfolio.total_value or 0)
            
            for holding in portfolio.holdings:
                asset_class = holding.asset_class.value if holding.asset_class else "Other"
                current_weight = float(holding.market_value or 0) / total_value if total_value > 0 else 0
                current_allocation[asset_class] = current_allocation.get(asset_class, 0) + current_weight
            
            # Generate rebalancing trades
            rebalancing_trades = []
            for asset_class, target_weight in target_allocation.items():
                current_weight = current_allocation.get(asset_class, 0)
                deviation = abs(target_weight - current_weight)
                
                if deviation > self.rebalancing_threshold:
                    action = "buy" if target_weight > current_weight else "sell"
                    amount = deviation * total_value
                    
                    if amount >= self.min_trade_size:
                        rebalancing_trades.append({
                            "asset_class": asset_class,
                            "action": action,
                            "target_weight": target_weight,
                            "current_weight": current_weight,
                            "amount": amount,
                            "deviation": deviation
                        })
            
            return AgentResponse(
                success=True,
                data={
                    "rebalancing_trades": rebalancing_trades,
                    "current_allocation": current_allocation,
                    "target_allocation": target_allocation,
                    "total_deviation": sum(abs(target_allocation.get(ac, 0) - cw) 
                                         for ac, cw in current_allocation.items())
                }
            )
            
        except Exception as e:
            return AgentResponse(success=False, error=str(e))
    
    async def _generate_event_response_proposal(self, content: Dict[str, Any]) -> AgentResponse:
        """Generate proposal in response to a specific event"""
        try:
            event_id = content.get("event_id")
            portfolio_id = content.get("portfolio_id")
            strategy = content.get("strategy", "defensive")
            
            event = Event.query.get(event_id)
            portfolio = Portfolio.query.get(portfolio_id)
            
            if not event or not portfolio:
                return AgentResponse(success=False, error="Event or portfolio not found")
            
            # Generate event-specific proposal
            proposal_data = await self._generate_event_specific_proposal(event, portfolio, strategy)
            
            # Create proposal record
            proposal = await self._create_proposal_record(
                event_id, portfolio_id, f"EventResponse_{strategy}", proposal_data
            )
            
            return AgentResponse(
                success=True,
                data={
                    "proposal_id": proposal.id if proposal else None,
                    "proposal_data": proposal_data
                }
            )
            
        except Exception as e:
            return AgentResponse(success=False, error=str(e))
    
    async def _generate_event_specific_proposal(
        self, 
        event: Event, 
        portfolio: Portfolio, 
        strategy: str
    ) -> Dict[str, Any]:
        """Generate proposal specific to an event"""
        try:
            holdings_data = await self._get_holdings_performance_data(portfolio)
            
            prompt = f"""
            Generate a {strategy} investment proposal in response to this event:
            
            Event: {event.title}
            Description: {event.description}
            Type: {event.event_type.value if event.event_type else 'Unknown'}
            Severity: {event.severity.value if event.severity else 'Unknown'}
            Related Symbols: {event.related_symbols}
            
            Current Portfolio: {json.dumps(holdings_data, indent=2)}
            
            Strategy: {strategy}
            
            Generate specific trades and rationale considering:
            1. Event impact on current holdings
            2. Risk mitigation strategies
            3. Opportunity identification
            4. Portfolio protection measures
            
            Return as JSON with proposed trades and detailed rationale.
            """
            
            proposal_data = await self.call_llm_with_json(prompt)
            proposal_data["event_context"] = {
                "event_id": event.id,
                "event_title": event.title,
                "event_type": event.event_type.value if event.event_type else None,
                "strategy": strategy
            }
            
            return proposal_data
            
        except Exception as e:
            logger.error(f"Event-specific proposal error: {str(e)}")
            return {"error": str(e)}
    
    async def _generate_esg_optimization(self, content: Dict[str, Any]) -> AgentResponse:
        """Generate ESG optimization proposal"""
        try:
            portfolio_id = content.get("portfolio_id")
            esg_targets = content.get("esg_targets", {})
            
            portfolio = Portfolio.query.get(portfolio_id)
            if not portfolio:
                return AgentResponse(success=False, error="Portfolio not found")
            
            proposal_data = await self._generate_esg_proposal(portfolio)
            proposal_data["esg_targets"] = esg_targets
            
            return AgentResponse(success=True, data=proposal_data)
            
        except Exception as e:
            return AgentResponse(success=False, error=str(e))
    
    async def _generate_tax_optimization(self, content: Dict[str, Any]) -> AgentResponse:
        """Generate tax optimization proposal"""
        try:
            portfolio_id = content.get("portfolio_id")
            tax_situation = content.get("tax_situation", {})
            
            portfolio = Portfolio.query.get(portfolio_id)
            if not portfolio:
                return AgentResponse(success=False, error="Portfolio not found")
            
            proposal_data = await self._generate_tax_aware_proposal(portfolio)
            proposal_data["tax_situation"] = tax_situation
            
            return AgentResponse(success=True, data=proposal_data)
            
        except Exception as e:
            return AgentResponse(success=False, error=str(e))

