import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from .base_agent import BaseAgent, AgentType, AgentMessage, AgentResponse, AgentCapability
from src.models.event import Event, Proposal, ProposalStatus
from src.models.client import Client
from src.models.portfolio import Portfolio, Holding
from src.models.user import db
import logging

logger = logging.getLogger(__name__)

class NarratorAgent(BaseAgent):
    """
    Narrator Agent - Responsible for client communication and reporting
    
    Capabilities:
    1. Generate client-friendly explanations of events and proposals
    2. Create portfolio performance reports
    3. Compose personalized client communications
    4. Generate executive summaries and insights
    5. Create regulatory and compliance reports
    6. Translate technical analysis into accessible language
    """
    
    def __init__(self, agent_id: str = "narrator_001"):
        capabilities = [
            AgentCapability(
                name="client_communication",
                description="Generate personalized client communications and explanations",
                input_schema={
                    "type": "object",
                    "properties": {
                        "client_id": {"type": "string"},
                        "communication_type": {"type": "string"},
                        "content": {"type": "object"}
                    }
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "message": {"type": "string"},
                        "tone": {"type": "string"},
                        "recommendations": {"type": "array"}
                    }
                }
            ),
            AgentCapability(
                name="report_generation",
                description="Generate comprehensive portfolio and performance reports",
                input_schema={
                    "type": "object",
                    "properties": {
                        "portfolio_id": {"type": "string"},
                        "report_type": {"type": "string"},
                        "period": {"type": "string"}
                    }
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "report": {"type": "object"},
                        "summary": {"type": "string"},
                        "key_insights": {"type": "array"}
                    }
                }
            ),
            AgentCapability(
                name="event_explanation",
                description="Explain events and their implications in client-friendly language",
                input_schema={
                    "type": "object",
                    "properties": {
                        "event_id": {"type": "string"},
                        "client_id": {"type": "string"}
                    }
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "explanation": {"type": "string"},
                        "impact_summary": {"type": "string"},
                        "next_steps": {"type": "array"}
                    }
                }
            )
        ]
        
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.NARRATOR,
            name="Narrator Agent",
            description="Generates client communications, reports, and explanations",
            capabilities=capabilities
        )
        
        # Communication parameters
        self.tone_styles = {
            "professional": "formal and professional",
            "friendly": "warm and approachable",
            "educational": "informative and explanatory",
            "urgent": "direct and action-oriented",
            "reassuring": "calm and confidence-building"
        }
    
    def get_system_prompt(self) -> str:
        return """You are the Narrator Agent in a wealth management system. Your role is to communicate complex financial information in clear, accessible language tailored to each client.

Your responsibilities:
1. Translate technical analysis into client-friendly explanations
2. Generate personalized portfolio reports and insights
3. Create compelling narratives around investment decisions
4. Explain market events and their portfolio implications
5. Compose professional client communications
6. Provide actionable recommendations and next steps

You have access to:
- Client profiles and communication preferences
- Portfolio performance data and analytics
- Market events and their analysis
- Proposal details and rationale
- Historical performance and trends

Always tailor your communication style to the client's sophistication level, preferences, and current situation. Use clear, jargon-free language while maintaining accuracy and professionalism."""
    
    async def process_message(self, message: AgentMessage) -> AgentResponse:
        """Process incoming messages and route to appropriate handlers"""
        try:
            action = message.content.get("action")
            
            if action == "ping":
                return AgentResponse(success=True, data={"status": "pong"})
            
            elif action == "generate_client_communication":
                return await self._generate_client_communication(message.content)
            
            elif action == "create_portfolio_report":
                return await self._create_portfolio_report(message.content)
            
            elif action == "explain_event":
                return await self._explain_event(message.content)
            
            elif action == "explain_proposal":
                return await self._explain_proposal(message.content)
            
            elif action == "generate_performance_summary":
                return await self._generate_performance_summary(message.content)
            
            elif action == "create_market_update":
                return await self._create_market_update(message.content)
            
            else:
                return AgentResponse(
                    success=False,
                    error=f"Unknown action: {action}"
                )
                
        except Exception as e:
            logger.error(f"Narrator agent error: {str(e)}")
            return AgentResponse(success=False, error=str(e))
    
    async def _generate_client_communication(self, content: Dict[str, Any]) -> AgentResponse:
        """Generate personalized client communication"""
        try:
            client_id = content.get("client_id")
            communication_type = content.get("communication_type", "general")
            message_content = content.get("content", {})
            tone = content.get("tone", "professional")
            
            client = Client.query.get(client_id)
            if not client:
                return AgentResponse(success=False, error="Client not found")
            
            # Get client's portfolio for context
            portfolio = Portfolio.query.filter(Portfolio.client_id == client_id).first()
            
            # Generate communication based on type
            if communication_type == "event_notification":
                communication = await self._generate_event_notification(client, portfolio, message_content, tone)
            elif communication_type == "proposal_explanation":
                communication = await self._generate_proposal_explanation(client, portfolio, message_content, tone)
            elif communication_type == "performance_update":
                communication = await self._generate_performance_update(client, portfolio, message_content, tone)
            elif communication_type == "market_commentary":
                communication = await self._generate_market_commentary(client, portfolio, message_content, tone)
            else:
                communication = await self._generate_general_communication(client, portfolio, message_content, tone)
            
            return AgentResponse(
                success=True,
                data=communication
            )
            
        except Exception as e:
            logger.error(f"Client communication generation error: {str(e)}")
            return AgentResponse(success=False, error=str(e))
    
    async def _generate_event_notification(
        self, 
        client: Client, 
        portfolio: Optional[Portfolio], 
        content: Dict[str, Any], 
        tone: str
    ) -> Dict[str, Any]:
        """Generate event notification for client"""
        try:
            event_id = content.get("event_id")
            event = Event.query.get(event_id) if event_id else None
            
            if not event:
                return {"error": "Event not found"}
            
            # Determine client sophistication level
            sophistication = self._assess_client_sophistication(client)
            
            prompt = f"""
            Create a client notification about this market event:
            
            Client Profile:
            - Name: {client.first_name} {client.last_name}
            - Investment Experience: {client.investment_experience}
            - Risk Tolerance: {client.risk_tolerance.value if client.risk_tolerance else 'Unknown'}
            - Sophistication Level: {sophistication}
            
            Event Details:
            - Title: {event.title}
            - Description: {event.description}
            - Type: {event.event_type.value if event.event_type else 'Unknown'}
            - Severity: {event.severity.value if event.severity else 'Unknown'}
            - Related Symbols: {event.related_symbols}
            
            Portfolio Context:
            - Total Value: ${float(portfolio.total_value or 0):,.2f} if portfolio else 'No portfolio'
            - Holdings Count: {len(portfolio.holdings) if portfolio else 0}
            
            Communication Requirements:
            - Tone: {self.tone_styles.get(tone, 'professional')}
            - Sophistication: {sophistication}
            - Include impact assessment
            - Provide next steps or recommendations
            
            Generate a personalized message that:
            1. Explains the event in appropriate language
            2. Describes potential impact on their portfolio
            3. Provides reassurance or guidance as needed
            4. Suggests next steps if any
            
            Return as JSON:
            {{
                "subject": "Email subject line",
                "message": "Full message content",
                "key_points": ["point1", "point2", "point3"],
                "recommendations": ["rec1", "rec2"],
                "urgency": "low/medium/high",
                "follow_up_required": true/false
            }}
            """
            
            communication = await self.call_llm_with_json(prompt)
            communication["communication_type"] = "event_notification"
            communication["tone"] = tone
            
            return communication
            
        except Exception as e:
            logger.error(f"Event notification generation error: {str(e)}")
            return {"error": str(e)}
    
    async def _generate_proposal_explanation(
        self, 
        client: Client, 
        portfolio: Optional[Portfolio], 
        content: Dict[str, Any], 
        tone: str
    ) -> Dict[str, Any]:
        """Generate proposal explanation for client"""
        try:
            proposal_id = content.get("proposal_id")
            proposal = Proposal.query.get(proposal_id) if proposal_id else None
            
            if not proposal:
                return {"error": "Proposal not found"}
            
            sophistication = self._assess_client_sophistication(client)
            
            prompt = f"""
            Explain this investment proposal to the client:
            
            Client Profile:
            - Name: {client.first_name} {client.last_name}
            - Investment Experience: {client.investment_experience}
            - Risk Tolerance: {client.risk_tolerance.value if client.risk_tolerance else 'Unknown'}
            - Investment Objectives: {client.investment_objectives}
            - Sophistication Level: {sophistication}
            
            Proposal Details:
            - Type: {proposal.proposal_type}
            - Objective: {proposal.objective}
            - Proposed Trades: {json.dumps(proposal.proposed_trades, indent=2)}
            - Expected Impact: {json.dumps(proposal.expected_impact, indent=2)}
            - Status: {proposal.status.value if proposal.status else 'Unknown'}
            
            Communication Requirements:
            - Tone: {self.tone_styles.get(tone, 'professional')}
            - Explain rationale clearly
            - Address potential concerns
            - Highlight benefits and risks
            
            Generate a clear explanation that:
            1. Summarizes what we're proposing to do
            2. Explains why we're recommending this
            3. Describes expected benefits and risks
            4. Addresses likely client questions
            5. Provides clear next steps
            
            Return as JSON:
            {{
                "subject": "Proposal explanation subject",
                "message": "Detailed explanation",
                "rationale_summary": "Why we're recommending this",
                "benefits": ["benefit1", "benefit2"],
                "risks": ["risk1", "risk2"],
                "next_steps": ["step1", "step2"],
                "requires_approval": true/false
            }}
            """
            
            communication = await self.call_llm_with_json(prompt)
            communication["communication_type"] = "proposal_explanation"
            communication["tone"] = tone
            
            return communication
            
        except Exception as e:
            logger.error(f"Proposal explanation generation error: {str(e)}")
            return {"error": str(e)}
    
    async def _generate_performance_update(
        self, 
        client: Client, 
        portfolio: Optional[Portfolio], 
        content: Dict[str, Any], 
        tone: str
    ) -> Dict[str, Any]:
        """Generate performance update for client"""
        try:
            if not portfolio:
                return {"error": "No portfolio found for client"}
            
            period = content.get("period", "monthly")
            sophistication = self._assess_client_sophistication(client)
            
            # Calculate performance metrics
            performance_data = await self._calculate_portfolio_performance(portfolio, period)
            
            prompt = f"""
            Create a portfolio performance update for the client:
            
            Client Profile:
            - Name: {client.first_name} {client.last_name}
            - Investment Experience: {client.investment_experience}
            - Risk Tolerance: {client.risk_tolerance.value if client.risk_tolerance else 'Unknown'}
            - Sophistication Level: {sophistication}
            
            Portfolio Performance ({period}):
            {json.dumps(performance_data, indent=2)}
            
            Communication Requirements:
            - Tone: {self.tone_styles.get(tone, 'professional')}
            - Focus on key insights
            - Explain performance in context
            - Highlight achievements and areas of concern
            
            Generate a performance update that:
            1. Summarizes overall performance
            2. Highlights key winners and losers
            3. Provides market context
            4. Identifies trends and patterns
            5. Suggests any needed actions
            
            Return as JSON:
            {{
                "subject": "Performance update subject",
                "message": "Performance summary and analysis",
                "key_metrics": {{
                    "total_return": "X.X%",
                    "benchmark_comparison": "outperformed/underperformed by X.X%",
                    "best_performer": "Asset name",
                    "worst_performer": "Asset name"
                }},
                "insights": ["insight1", "insight2"],
                "recommendations": ["rec1", "rec2"]
            }}
            """
            
            communication = await self.call_llm_with_json(prompt)
            communication["communication_type"] = "performance_update"
            communication["tone"] = tone
            communication["performance_data"] = performance_data
            
            return communication
            
        except Exception as e:
            logger.error(f"Performance update generation error: {str(e)}")
            return {"error": str(e)}
    
    async def _generate_market_commentary(
        self, 
        client: Client, 
        portfolio: Optional[Portfolio], 
        content: Dict[str, Any], 
        tone: str
    ) -> Dict[str, Any]:
        """Generate market commentary for client"""
        try:
            market_events = content.get("market_events", [])
            sophistication = self._assess_client_sophistication(client)
            
            prompt = f"""
            Create a market commentary for the client:
            
            Client Profile:
            - Name: {client.first_name} {client.last_name}
            - Investment Experience: {client.investment_experience}
            - Risk Tolerance: {client.risk_tolerance.value if client.risk_tolerance else 'Unknown'}
            - Sophistication Level: {sophistication}
            
            Recent Market Events:
            {json.dumps(market_events, indent=2)}
            
            Portfolio Context:
            - Total Value: ${float(portfolio.total_value or 0):,.2f} if portfolio else 'No portfolio'
            - Asset Allocation: Mixed portfolio if portfolio else 'No portfolio'
            
            Communication Requirements:
            - Tone: {self.tone_styles.get(tone, 'professional')}
            - Provide market perspective
            - Connect to client's situation
            - Offer reassurance or guidance
            
            Generate market commentary that:
            1. Summarizes key market developments
            2. Explains implications for investors
            3. Relates to client's portfolio if applicable
            4. Provides perspective and context
            5. Offers guidance for the period ahead
            
            Return as JSON:
            {{
                "subject": "Market update subject",
                "message": "Market commentary and analysis",
                "market_summary": "Brief market overview",
                "key_themes": ["theme1", "theme2"],
                "outlook": "Market outlook and expectations",
                "client_implications": "What this means for you"
            }}
            """
            
            communication = await self.call_llm_with_json(prompt)
            communication["communication_type"] = "market_commentary"
            communication["tone"] = tone
            
            return communication
            
        except Exception as e:
            logger.error(f"Market commentary generation error: {str(e)}")
            return {"error": str(e)}
    
    async def _generate_general_communication(
        self, 
        client: Client, 
        portfolio: Optional[Portfolio], 
        content: Dict[str, Any], 
        tone: str
    ) -> Dict[str, Any]:
        """Generate general client communication"""
        try:
            message_topic = content.get("topic", "General Update")
            custom_content = content.get("custom_content", "")
            
            sophistication = self._assess_client_sophistication(client)
            
            prompt = f"""
            Create a general communication for the client:
            
            Client Profile:
            - Name: {client.first_name} {client.last_name}
            - Investment Experience: {client.investment_experience}
            - Sophistication Level: {sophistication}
            
            Topic: {message_topic}
            Content: {custom_content}
            
            Communication Requirements:
            - Tone: {self.tone_styles.get(tone, 'professional')}
            - Personalized and relevant
            - Clear and actionable
            
            Generate a personalized message that addresses the topic appropriately.
            
            Return as JSON:
            {{
                "subject": "Communication subject",
                "message": "Message content",
                "key_points": ["point1", "point2"],
                "next_steps": ["step1", "step2"]
            }}
            """
            
            communication = await self.call_llm_with_json(prompt)
            communication["communication_type"] = "general"
            communication["tone"] = tone
            
            return communication
            
        except Exception as e:
            logger.error(f"General communication generation error: {str(e)}")
            return {"error": str(e)}
    
    def _assess_client_sophistication(self, client: Client) -> str:
        """Assess client sophistication level for communication tailoring"""
        try:
            experience = client.investment_experience or ""
            net_worth = float(client.net_worth or 0)
            
            if "professional" in experience.lower() or "extensive" in experience.lower():
                return "high"
            elif net_worth > 1000000 or "moderate" in experience.lower():
                return "medium"
            else:
                return "basic"
                
        except Exception:
            return "basic"
    
    async def _calculate_portfolio_performance(self, portfolio: Portfolio, period: str) -> Dict[str, Any]:
        """Calculate portfolio performance metrics"""
        try:
            current_value = float(portfolio.total_value or 0)
            cost_basis = float(portfolio.total_cost_basis or 0)
            
            # Calculate basic metrics
            total_return = current_value - cost_basis
            total_return_pct = (total_return / cost_basis) if cost_basis > 0 else 0
            
            # Get holdings performance
            holdings_performance = []
            for holding in portfolio.holdings:
                holding_return = float(holding.market_value or 0) - (float(holding.cost_basis or 0) * float(holding.quantity or 0))
                holding_return_pct = holding_return / (float(holding.cost_basis or 0) * float(holding.quantity or 0)) if holding.cost_basis and holding.quantity else 0
                
                holdings_performance.append({
                    "symbol": holding.symbol,
                    "return": holding_return,
                    "return_pct": holding_return_pct,
                    "market_value": float(holding.market_value or 0)
                })
            
            # Sort by performance
            holdings_performance.sort(key=lambda x: x["return_pct"], reverse=True)
            
            return {
                "period": period,
                "current_value": current_value,
                "cost_basis": cost_basis,
                "total_return": total_return,
                "total_return_pct": total_return_pct,
                "best_performer": holdings_performance[0] if holdings_performance else None,
                "worst_performer": holdings_performance[-1] if holdings_performance else None,
                "holdings_count": len(holdings_performance)
            }
            
        except Exception as e:
            logger.error(f"Performance calculation error: {str(e)}")
            return {"error": str(e)}
    
    async def _create_portfolio_report(self, content: Dict[str, Any]) -> AgentResponse:
        """Create comprehensive portfolio report"""
        try:
            portfolio_id = content.get("portfolio_id")
            report_type = content.get("report_type", "comprehensive")
            period = content.get("period", "quarterly")
            
            portfolio = Portfolio.query.get(portfolio_id)
            if not portfolio:
                return AgentResponse(success=False, error="Portfolio not found")
            
            client = Client.query.get(portfolio.client_id)
            if not client:
                return AgentResponse(success=False, error="Client not found")
            
            # Generate comprehensive report
            report_data = await self._generate_comprehensive_report(portfolio, client, report_type, period)
            
            return AgentResponse(
                success=True,
                data=report_data
            )
            
        except Exception as e:
            return AgentResponse(success=False, error=str(e))
    
    async def _generate_comprehensive_report(
        self, 
        portfolio: Portfolio, 
        client: Client, 
        report_type: str, 
        period: str
    ) -> Dict[str, Any]:
        """Generate comprehensive portfolio report"""
        try:
            performance_data = await self._calculate_portfolio_performance(portfolio, period)
            
            prompt = f"""
            Generate a comprehensive {report_type} portfolio report:
            
            Client: {client.first_name} {client.last_name}
            Portfolio Value: ${float(portfolio.total_value or 0):,.2f}
            Report Period: {period}
            
            Performance Data:
            {json.dumps(performance_data, indent=2)}
            
            Holdings Summary:
            {json.dumps([{
                "symbol": h.symbol,
                "value": float(h.market_value or 0),
                "sector": h.sector,
                "asset_class": h.asset_class.value if h.asset_class else None
            } for h in portfolio.holdings], indent=2)}
            
            Generate a professional report with:
            1. Executive summary
            2. Performance analysis
            3. Asset allocation review
            4. Risk assessment
            5. Recommendations
            6. Market outlook
            
            Return as JSON:
            {{
                "executive_summary": "High-level overview",
                "performance_analysis": "Detailed performance review",
                "asset_allocation": "Current allocation analysis",
                "risk_assessment": "Risk profile and metrics",
                "recommendations": ["rec1", "rec2", "rec3"],
                "market_outlook": "Forward-looking perspective",
                "key_insights": ["insight1", "insight2", "insight3"]
            }}
            """
            
            report = await self.call_llm_with_json(prompt)
            report["report_type"] = report_type
            report["period"] = period
            report["generated_at"] = datetime.utcnow().isoformat()
            report["performance_data"] = performance_data
            
            return report
            
        except Exception as e:
            logger.error(f"Comprehensive report generation error: {str(e)}")
            return {"error": str(e)}
    
    async def _explain_event(self, content: Dict[str, Any]) -> AgentResponse:
        """Explain an event to a client"""
        try:
            event_id = content.get("event_id")
            client_id = content.get("client_id")
            
            event = Event.query.get(event_id)
            client = Client.query.get(client_id)
            
            if not event or not client:
                return AgentResponse(success=False, error="Event or client not found")
            
            explanation = await self._generate_event_explanation(event, client)
            
            return AgentResponse(success=True, data=explanation)
            
        except Exception as e:
            return AgentResponse(success=False, error=str(e))
    
    async def _generate_event_explanation(self, event: Event, client: Client) -> Dict[str, Any]:
        """Generate detailed event explanation"""
        try:
            sophistication = self._assess_client_sophistication(client)
            
            prompt = f"""
            Explain this market event to the client in clear, accessible language:
            
            Event: {event.title}
            Description: {event.description}
            Type: {event.event_type.value if event.event_type else 'Unknown'}
            Severity: {event.severity.value if event.severity else 'Unknown'}
            
            Client Sophistication: {sophistication}
            
            Provide:
            1. Clear explanation of what happened
            2. Why it matters for investors
            3. Potential implications
            4. Historical context if relevant
            5. What investors should consider
            
            Return as JSON:
            {{
                "explanation": "Clear explanation of the event",
                "why_it_matters": "Why this is important for investors",
                "implications": ["implication1", "implication2"],
                "historical_context": "Similar events in the past",
                "investor_considerations": ["consideration1", "consideration2"]
            }}
            """
            
            explanation = await self.call_llm_with_json(prompt)
            return explanation
            
        except Exception as e:
            logger.error(f"Event explanation generation error: {str(e)}")
            return {"error": str(e)}
    
    async def _explain_proposal(self, content: Dict[str, Any]) -> AgentResponse:
        """Explain a proposal to a client"""
        try:
            proposal_id = content.get("proposal_id")
            client_id = content.get("client_id")
            
            proposal = Proposal.query.get(proposal_id)
            client = Client.query.get(client_id)
            
            if not proposal or not client:
                return AgentResponse(success=False, error="Proposal or client not found")
            
            portfolio = Portfolio.query.get(proposal.portfolio_id)
            
            explanation = await self._generate_proposal_explanation(client, portfolio, {"proposal_id": proposal_id}, "professional")
            
            return AgentResponse(success=True, data=explanation)
            
        except Exception as e:
            return AgentResponse(success=False, error=str(e))
    
    async def _generate_performance_summary(self, content: Dict[str, Any]) -> AgentResponse:
        """Generate performance summary"""
        try:
            portfolio_id = content.get("portfolio_id")
            period = content.get("period", "monthly")
            
            portfolio = Portfolio.query.get(portfolio_id)
            if not portfolio:
                return AgentResponse(success=False, error="Portfolio not found")
            
            client = Client.query.get(portfolio.client_id)
            
            performance_summary = await self._generate_performance_update(client, portfolio, {"period": period}, "professional")
            
            return AgentResponse(success=True, data=performance_summary)
            
        except Exception as e:
            return AgentResponse(success=False, error=str(e))
    
    async def _create_market_update(self, content: Dict[str, Any]) -> AgentResponse:
        """Create market update communication"""
        try:
            client_id = content.get("client_id")
            market_events = content.get("market_events", [])
            
            client = Client.query.get(client_id)
            if not client:
                return AgentResponse(success=False, error="Client not found")
            
            portfolio = Portfolio.query.filter(Portfolio.client_id == client_id).first()
            
            market_update = await self._generate_market_commentary(
                client, 
                portfolio, 
                {"market_events": market_events}, 
                "professional"
            )
            
            return AgentResponse(success=True, data=market_update)
            
        except Exception as e:
            return AgentResponse(success=False, error=str(e))

