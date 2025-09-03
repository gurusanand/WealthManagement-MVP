import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from .base_agent import BaseAgent, AgentType, AgentMessage, AgentResponse, AgentCapability
from src.models.event import Event, Proposal
from src.models.client import Client, RiskTolerance
from src.models.portfolio import Portfolio, Holding
from src.models.user import db
import logging

logger = logging.getLogger(__name__)

class CheckerAgent(BaseAgent):
    """
    Checker Agent - Responsible for compliance validation and risk assessment
    
    Capabilities:
    1. Validate proposals against compliance rules
    2. Assess risk levels and suitability
    3. Check regulatory requirements
    4. Verify client investment restrictions
    5. Validate concentration limits
    6. Perform stress testing on proposals
    """
    
    def __init__(self, agent_id: str = "checker_001"):
        capabilities = [
            AgentCapability(
                name="compliance_validation",
                description="Validate proposals against compliance rules and regulations",
                input_schema={
                    "type": "object",
                    "properties": {
                        "proposal_id": {"type": "string"},
                        "validation_type": {"type": "string"}
                    }
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "compliance_status": {"type": "string"},
                        "violations": {"type": "array"},
                        "recommendations": {"type": "array"}
                    }
                }
            ),
            AgentCapability(
                name="risk_assessment",
                description="Assess risk levels and suitability of proposals",
                input_schema={
                    "type": "object",
                    "properties": {
                        "proposal_id": {"type": "string"},
                        "client_id": {"type": "string"}
                    }
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "risk_score": {"type": "number"},
                        "suitability": {"type": "string"},
                        "risk_factors": {"type": "array"}
                    }
                }
            ),
            AgentCapability(
                name="stress_testing",
                description="Perform stress testing on portfolio proposals",
                input_schema={
                    "type": "object",
                    "properties": {
                        "proposal_id": {"type": "string"},
                        "scenarios": {"type": "array"}
                    }
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "stress_results": {"type": "object"},
                        "worst_case_scenario": {"type": "object"}
                    }
                }
            )  # <-- This should be a parenthesis
        ]      # <-- This should be a square bracket
        
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.CHECKER,
            name="Checker Agent",
            description="Validates compliance and assesses risk for portfolio proposals",
            capabilities=capabilities
        )
        
        # Compliance and risk parameters
        self.max_concentration_limit = 0.20  # 20% maximum single position
        self.max_sector_concentration = 0.30  # 30% maximum sector concentration
        self.min_diversification_holdings = 10  # Minimum holdings for diversification
        self.max_leverage_ratio = 1.5  # Maximum leverage allowed
        self.stress_test_scenarios = [
            {"name": "Market Crash", "equity_shock": -0.30, "bond_shock": -0.10},
            {"name": "Interest Rate Rise", "equity_shock": -0.15, "bond_shock": -0.20},
            {"name": "Inflation Spike", "equity_shock": -0.10, "bond_shock": -0.15},
            {"name": "Recession", "equity_shock": -0.25, "bond_shock": -0.05}
        ]
    
    def get_system_prompt(self) -> str:
        return """You are the Checker Agent in a wealth management system. Your role is to validate proposals for compliance and assess their risk characteristics.

Your responsibilities:
1. Validate all proposals against regulatory requirements
2. Check compliance with client investment restrictions
3. Assess risk levels and suitability for client profiles
4. Verify concentration limits and diversification requirements
5. Perform stress testing under various market scenarios
6. Identify potential compliance violations and risks

You have access to:
- Regulatory compliance rules and requirements
- Client risk profiles and investment restrictions
- Portfolio concentration and diversification metrics
- Historical market data for stress testing
- Industry best practices and guidelines

Always provide thorough analysis with clear pass/fail determinations and specific recommendations for addressing any issues."""
    
    async def process_message(self, message: AgentMessage) -> AgentResponse:
        """Process incoming messages and route to appropriate handlers"""
        try:
            action = message.content.get("action")
            
            if action == "ping":
                return AgentResponse(success=True, data={"status": "pong"})
            
            elif action == "validate_proposal":
                return await self._validate_proposal(message.content)
            
            elif action == "assess_risk":
                return await self._assess_risk(message.content)
            
            elif action == "check_compliance":
                return await self._check_compliance(message.content)
            
            elif action == "stress_test":
                return await self._perform_stress_test(message.content)
            
            elif action == "validate_concentration":
                return await self._validate_concentration(message.content)
            
            elif action == "check_suitability":
                return await self._check_suitability(message.content)
            
            else:
                return AgentResponse(
                    success=False,
                    error=f"Unknown action: {action}"
                )
                
        except Exception as e:
            logger.error(f"Checker agent error: {str(e)}")
            return AgentResponse(success=False, error=str(e))
    
    async def _validate_proposal(self, content: Dict[str, Any]) -> AgentResponse:
        """Main proposal validation orchestrator"""
        try:
            proposal_id = content.get("proposal_id")
            validation_types = content.get("validation_types", ["all"])
            
            proposal = Proposal.query.get(proposal_id)
            if not proposal:
                return AgentResponse(success=False, error="Proposal not found")
            
            portfolio = Portfolio.query.get(proposal.portfolio_id)
            client = Client.query.get(portfolio.client_id) if portfolio else None
            
            if not portfolio or not client:
                return AgentResponse(success=False, error="Portfolio or client not found")
            
            validation_results = {
                "proposal_id": proposal_id,
                "overall_status": "PENDING",
                "validations": {},
                "violations": [],
                "recommendations": [],
                "risk_score": 0
            }
            
            # Perform different types of validation
            if "all" in validation_types or "compliance" in validation_types:
                compliance_result = await self._perform_compliance_check(proposal, portfolio, client)
                validation_results["validations"]["compliance"] = compliance_result
                validation_results["violations"].extend(compliance_result.get("violations", []))
            
            if "all" in validation_types or "risk" in validation_types:
                risk_result = await self._perform_risk_assessment(proposal, portfolio, client)
                validation_results["validations"]["risk"] = risk_result
                validation_results["risk_score"] = risk_result.get("risk_score", 0)
            
            if "all" in validation_types or "concentration" in validation_types:
                concentration_result = await self._check_concentration_limits(proposal, portfolio)
                validation_results["validations"]["concentration"] = concentration_result
                validation_results["violations"].extend(concentration_result.get("violations", []))
            
            if "all" in validation_types or "suitability" in validation_types:
                suitability_result = await self._assess_suitability(proposal, client)
                validation_results["validations"]["suitability"] = suitability_result
                if not suitability_result.get("suitable", True):
                    validation_results["violations"].append("Proposal not suitable for client profile")
            
            if "all" in validation_types or "stress_test" in validation_types:
                stress_result = await self._run_stress_tests(proposal, portfolio)
                validation_results["validations"]["stress_test"] = stress_result
            
            # Determine overall status
            if validation_results["violations"]:
                validation_results["overall_status"] = "REJECTED"
                validation_results["recommendations"] = await self._generate_remediation_recommendations(
                    validation_results["violations"], proposal, client
                )
            else:
                validation_results["overall_status"] = "APPROVED"
            
            # Update proposal status
            proposal.status = ProposalStatus.APPROVED if validation_results["overall_status"] == "APPROVED" else ProposalStatus.REJECTED
            proposal.validation_results = validation_results
            proposal.updated_at = datetime.utcnow()
            
            db.session.commit()
            
            return AgentResponse(
                success=True,
                data=validation_results
            )
            
        except Exception as e:
            logger.error(f"Proposal validation error: {str(e)}")
            db.session.rollback()
            return AgentResponse(success=False, error=str(e))
    
    async def _perform_compliance_check(
        self, 
        proposal: Proposal, 
        portfolio: Portfolio, 
        client: Client
    ) -> Dict[str, Any]:
        """Perform comprehensive compliance checking"""
        try:
            violations = []
            compliance_checks = []
            
            # Check client investment restrictions
            if client.investment_restrictions:
                restricted_sectors = client.investment_restrictions.get("restricted_sectors", [])
                restricted_symbols = client.investment_restrictions.get("restricted_symbols", [])
                
                for trade in proposal.proposed_trades:
                    symbol = trade.get("symbol", "")
                    
                    # Check restricted symbols
                    if symbol in restricted_symbols:
                        violations.append(f"Symbol {symbol} is restricted for this client")
                    
                    # Check restricted sectors (would need sector lookup)
                    # This is simplified - in practice would lookup sector for symbol
                
                compliance_checks.append({
                    "check": "investment_restrictions",
                    "status": "PASS" if not violations else "FAIL",
                    "details": f"Checked {len(proposal.proposed_trades)} trades against restrictions"
                })
            
            # Check regulatory requirements using LLM
            prompt = f"""
            Perform regulatory compliance check for this investment proposal:
            
            Client Profile:
            - Risk Tolerance: {client.risk_tolerance.value if client.risk_tolerance else 'Unknown'}
            - Investment Experience: {client.investment_experience}
            - Net Worth: {client.net_worth}
            - Annual Income: {client.annual_income}
            
            Proposed Trades:
            {json.dumps(proposal.proposed_trades, indent=2)}
            
            Check for:
            1. Suitability requirements
            2. Concentration limits
            3. Liquidity requirements
            4. Regulatory restrictions
            5. Fiduciary duty compliance
            
            Return as JSON:
            {{
                "regulatory_violations": ["violation1", "violation2"],
                "compliance_score": 0.85,
                "regulatory_checks": [
                    {{"check": "suitability", "status": "PASS", "details": "explanation"}},
                    {{"check": "concentration", "status": "FAIL", "details": "explanation"}}
                ]
            }}
            """
            
            regulatory_result = await self.call_llm_with_json(prompt)
            
            violations.extend(regulatory_result.get("regulatory_violations", []))
            compliance_checks.extend(regulatory_result.get("regulatory_checks", []))
            
            return {
                "status": "PASS" if not violations else "FAIL",
                "violations": violations,
                "compliance_checks": compliance_checks,
                "compliance_score": regulatory_result.get("compliance_score", 0.5)
            }
            
        except Exception as e:
            logger.error(f"Compliance check error: {str(e)}")
            return {"status": "ERROR", "error": str(e)}
    
    async def _perform_risk_assessment(
        self, 
        proposal: Proposal, 
        portfolio: Portfolio, 
        client: Client
    ) -> Dict[str, Any]:
        """Perform comprehensive risk assessment"""
        try:
            # Calculate proposed portfolio risk metrics
            current_holdings = await self._get_current_holdings_data(portfolio)
            proposed_changes = proposal.proposed_trades
            
            # Simulate portfolio after proposed changes
            simulated_portfolio = await self._simulate_portfolio_changes(current_holdings, proposed_changes)
            
            # Calculate risk metrics
            risk_metrics = await self._calculate_risk_metrics(simulated_portfolio)
            
            # Assess against client risk tolerance
            risk_tolerance_mapping = {
                RiskTolerance.CONSERVATIVE: {"max_volatility": 0.10, "max_drawdown": 0.15},
                RiskTolerance.MODERATE: {"max_volatility": 0.15, "max_drawdown": 0.25},
                RiskTolerance.AGGRESSIVE: {"max_volatility": 0.25, "max_drawdown": 0.40}
            }
            
            client_limits = risk_tolerance_mapping.get(client.risk_tolerance, risk_tolerance_mapping[RiskTolerance.MODERATE])
            
            risk_violations = []
            if risk_metrics.get("volatility", 0) > client_limits["max_volatility"]:
                risk_violations.append(f"Portfolio volatility {risk_metrics['volatility']:.2%} exceeds client limit {client_limits['max_volatility']:.2%}")
            
            if risk_metrics.get("max_drawdown", 0) > client_limits["max_drawdown"]:
                risk_violations.append(f"Max drawdown {risk_metrics['max_drawdown']:.2%} exceeds client limit {client_limits['max_drawdown']:.2%}")
            
            # Calculate overall risk score (0-100)
            risk_score = min(100, (risk_metrics.get("volatility", 0) * 100 + 
                                 risk_metrics.get("max_drawdown", 0) * 100) / 2)
            
            return {
                "risk_score": risk_score,
                "risk_metrics": risk_metrics,
                "risk_violations": risk_violations,
                "client_limits": client_limits,
                "risk_level": self._categorize_risk_level(risk_score),
                "suitable_for_client": len(risk_violations) == 0
            }
            
        except Exception as e:
            logger.error(f"Risk assessment error: {str(e)}")
            return {"error": str(e)}
    
    async def _check_concentration_limits(
        self, 
        proposal: Proposal, 
        portfolio: Portfolio
    ) -> Dict[str, Any]:
        """Check concentration limits after proposed changes"""
        try:
            violations = []
            
            # Get current holdings
            current_holdings = await self._get_current_holdings_data(portfolio)
            
            # Simulate portfolio after changes
            simulated_portfolio = await self._simulate_portfolio_changes(current_holdings, proposal.proposed_trades)
            
            total_value = sum(holding["market_value"] for holding in simulated_portfolio)
            
            # Check individual position concentration
            for holding in simulated_portfolio:
                concentration = holding["market_value"] / total_value if total_value > 0 else 0
                if concentration > self.max_concentration_limit:
                    violations.append(f"Position {holding['symbol']} concentration {concentration:.2%} exceeds limit {self.max_concentration_limit:.2%}")
            
            # Check sector concentration
            sector_concentrations = {}
            for holding in simulated_portfolio:
                sector = holding.get("sector", "Unknown")
                sector_concentrations[sector] = sector_concentrations.get(sector, 0) + holding["market_value"]
            
            for sector, value in sector_concentrations.items():
                concentration = value / total_value if total_value > 0 else 0
                if concentration > self.max_sector_concentration:
                    violations.append(f"Sector {sector} concentration {concentration:.2%} exceeds limit {self.max_sector_concentration:.2%}")
            
            # Check minimum diversification
            if len(simulated_portfolio) < self.min_diversification_holdings:
                violations.append(f"Portfolio has {len(simulated_portfolio)} holdings, minimum required: {self.min_diversification_holdings}")
            
            return {
                "status": "PASS" if not violations else "FAIL",
                "violations": violations,
                "concentration_metrics": {
                    "max_position_concentration": max((h["market_value"] / total_value for h in simulated_portfolio), default=0),
                    "sector_concentrations": {k: v/total_value for k, v in sector_concentrations.items()},
                    "number_of_holdings": len(simulated_portfolio)
                }
            }
            
        except Exception as e:
            logger.error(f"Concentration check error: {str(e)}")
            return {"status": "ERROR", "error": str(e)}
    
    async def _assess_suitability(self, proposal: Proposal, client: Client) -> Dict[str, Any]:
        """Assess proposal suitability for client"""
        try:
            prompt = f"""
            Assess the suitability of this investment proposal for the client:
            
            Client Profile:
            - Age: {client.age}
            - Risk Tolerance: {client.risk_tolerance.value if client.risk_tolerance else 'Unknown'}
            - Investment Experience: {client.investment_experience}
            - Investment Horizon: {client.investment_horizon}
            - Net Worth: {client.net_worth}
            - Annual Income: {client.annual_income}
            - Investment Objectives: {client.investment_objectives}
            - ESG Preferences: Environmental={client.esg_environmental}, Social={client.esg_social}, Governance={client.esg_governance}
            
            Proposed Trades:
            {json.dumps(proposal.proposed_trades, indent=2)}
            
            Expected Impact:
            {json.dumps(proposal.expected_impact, indent=2)}
            
            Assess:
            1. Alignment with risk tolerance
            2. Appropriateness for investment horizon
            3. Consistency with investment objectives
            4. Suitability for client's experience level
            5. ESG alignment if applicable
            
            Return as JSON:
            {{
                "suitable": true/false,
                "suitability_score": 0.85,
                "alignment_factors": {{
                    "risk_alignment": 0.9,
                    "horizon_alignment": 0.8,
                    "objective_alignment": 0.85,
                    "experience_alignment": 0.9,
                    "esg_alignment": 0.7
                }},
                "concerns": ["concern1", "concern2"],
                "recommendations": ["recommendation1", "recommendation2"]
            }}
            """
            
            suitability_result = await self.call_llm_with_json(prompt)
            return suitability_result
            
        except Exception as e:
            logger.error(f"Suitability assessment error: {str(e)}")
            return {"suitable": False, "error": str(e)}
    
    async def _run_stress_tests(self, proposal: Proposal, portfolio: Portfolio) -> Dict[str, Any]:
        """Run stress tests on the proposed portfolio"""
        try:
            # Get current holdings and simulate changes
            current_holdings = await self._get_current_holdings_data(portfolio)
            simulated_portfolio = await self._simulate_portfolio_changes(current_holdings, proposal.proposed_trades)
            
            stress_results = {}
            worst_case_loss = 0
            worst_scenario = ""
            
            for scenario in self.stress_test_scenarios:
                scenario_loss = 0
                total_value = sum(h["market_value"] for h in simulated_portfolio)
                
                for holding in simulated_portfolio:
                    asset_class = holding.get("asset_class", "Equity")
                    
                    if asset_class == "Equity":
                        shock = scenario.get("equity_shock", 0)
                    elif asset_class == "Bond":
                        shock = scenario.get("bond_shock", 0)
                    else:
                        shock = scenario.get("equity_shock", 0) * 0.5  # Assume other assets have half equity sensitivity
                    
                    holding_loss = holding["market_value"] * abs(shock)
                    scenario_loss += holding_loss
                
                scenario_loss_pct = scenario_loss / total_value if total_value > 0 else 0
                
                stress_results[scenario["name"]] = {
                    "absolute_loss": scenario_loss,
                    "percentage_loss": scenario_loss_pct,
                    "remaining_value": total_value - scenario_loss
                }
                
                if scenario_loss_pct > worst_case_loss:
                    worst_case_loss = scenario_loss_pct
                    worst_scenario = scenario["name"]
            
            return {
                "stress_results": stress_results,
                "worst_case_scenario": {
                    "scenario": worst_scenario,
                    "loss_percentage": worst_case_loss,
                    "pass_threshold": worst_case_loss < 0.35  # 35% max loss threshold
                },
                "overall_stress_score": max(0, 100 - (worst_case_loss * 100))
            }
            
        except Exception as e:
            logger.error(f"Stress test error: {str(e)}")
            return {"error": str(e)}
    
    async def _get_current_holdings_data(self, portfolio: Portfolio) -> List[Dict[str, Any]]:
        """Get current holdings data for analysis"""
        holdings_data = []
        
        for holding in portfolio.holdings:
            holding_data = {
                "symbol": holding.symbol,
                "quantity": float(holding.quantity or 0),
                "market_value": float(holding.market_value or 0),
                "sector": holding.sector,
                "asset_class": holding.asset_class.value if holding.asset_class else "Equity"
            }
            holdings_data.append(holding_data)
        
        return holdings_data
    
    async def _simulate_portfolio_changes(
        self, 
        current_holdings: List[Dict[str, Any]], 
        proposed_trades: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Simulate portfolio after applying proposed trades"""
        try:
            # Create a copy of current holdings
            simulated_holdings = {h["symbol"]: h.copy() for h in current_holdings}
            
            # Apply proposed trades
            for trade in proposed_trades:
                symbol = trade.get("symbol", "")
                action = trade.get("action", "").lower()
                quantity = trade.get("quantity", 0)
                
                if symbol not in simulated_holdings:
                    # New position
                    simulated_holdings[symbol] = {
                        "symbol": symbol,
                        "quantity": 0,
                        "market_value": 0,
                        "sector": trade.get("sector", "Unknown"),
                        "asset_class": trade.get("asset_class", "Equity")
                    }
                
                if action == "buy":
                    simulated_holdings[symbol]["quantity"] += quantity
                    # Estimate market value (simplified - would use current price)
                    estimated_price = trade.get("price", 100)  # Default price
                    simulated_holdings[symbol]["market_value"] += quantity * estimated_price
                elif action == "sell":
                    simulated_holdings[symbol]["quantity"] -= quantity
                    estimated_price = trade.get("price", 100)
                    simulated_holdings[symbol]["market_value"] -= quantity * estimated_price
                    
                    # Remove position if quantity becomes zero or negative
                    if simulated_holdings[symbol]["quantity"] <= 0:
                        del simulated_holdings[symbol]
            
            return list(simulated_holdings.values())
            
        except Exception as e:
            logger.error(f"Portfolio simulation error: {str(e)}")
            return current_holdings
    
    async def _calculate_risk_metrics(self, portfolio_holdings: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate risk metrics for a portfolio"""
        try:
            if not portfolio_holdings:
                return {}
            
            total_value = sum(h["market_value"] for h in portfolio_holdings)
            
            # Simplified risk calculations
            # In practice, would use historical returns and correlations
            
            # Estimate volatility based on asset classes
            asset_class_volatilities = {
                "Equity": 0.20,
                "Bond": 0.05,
                "Commodity": 0.25,
                "Real Estate": 0.15,
                "Cash": 0.01
            }
            
            weighted_volatility = 0
            for holding in portfolio_holdings:
                weight = holding["market_value"] / total_value if total_value > 0 else 0
                asset_class = holding.get("asset_class", "Equity")
                volatility = asset_class_volatilities.get(asset_class, 0.20)
                weighted_volatility += weight * volatility
            
            # Estimate max drawdown (simplified)
            max_drawdown = weighted_volatility * 2  # Rough approximation
            
            return {
                "volatility": weighted_volatility,
                "max_drawdown": max_drawdown,
                "total_value": total_value,
                "number_of_holdings": len(portfolio_holdings)
            }
            
        except Exception as e:
            logger.error(f"Risk metrics calculation error: {str(e)}")
            return {}
    
    def _categorize_risk_level(self, risk_score: float) -> str:
        """Categorize risk level based on risk score"""
        if risk_score >= 75:
            return "High"
        elif risk_score >= 50:
            return "Medium"
        elif risk_score >= 25:
            return "Low"
        else:
            return "Very Low"
    
    async def _generate_remediation_recommendations(
        self, 
        violations: List[str], 
        proposal: Proposal, 
        client: Client
    ) -> List[str]:
        """Generate recommendations to address violations"""
        try:
            prompt = f"""
            Generate specific recommendations to address these compliance violations:
            
            Violations:
            {json.dumps(violations, indent=2)}
            
            Original Proposal:
            {json.dumps(proposal.proposed_trades, indent=2)}
            
            Client Profile:
            - Risk Tolerance: {client.risk_tolerance.value if client.risk_tolerance else 'Unknown'}
            - Investment Objectives: {client.investment_objectives}
            
            Provide 3-5 specific, actionable recommendations to make the proposal compliant:
            
            Return as JSON array:
            ["recommendation1", "recommendation2", "recommendation3"]
            """
            
            recommendations = await self.call_llm_with_json(prompt)
            return recommendations if isinstance(recommendations, list) else []
            
        except Exception as e:
            logger.error(f"Remediation recommendations error: {str(e)}")
            return ["Review proposal for compliance issues", "Consult with compliance team"]
    
    async def _assess_risk(self, content: Dict[str, Any]) -> AgentResponse:
        """Assess risk for a proposal"""
        try:
            proposal_id = content.get("proposal_id")
            client_id = content.get("client_id")
            
            proposal = Proposal.query.get(proposal_id)
            client = Client.query.get(client_id)
            portfolio = Portfolio.query.get(proposal.portfolio_id) if proposal else None
            
            if not proposal or not client or not portfolio:
                return AgentResponse(success=False, error="Proposal, client, or portfolio not found")
            
            risk_assessment = await self._perform_risk_assessment(proposal, portfolio, client)
            
            return AgentResponse(success=True, data=risk_assessment)
            
        except Exception as e:
            return AgentResponse(success=False, error=str(e))
    
    async def _check_compliance(self, content: Dict[str, Any]) -> AgentResponse:
        """Check compliance for a proposal"""
        try:
            proposal_id = content.get("proposal_id")
            
            proposal = Proposal.query.get(proposal_id)
            if not proposal:
                return AgentResponse(success=False, error="Proposal not found")
            
            portfolio = Portfolio.query.get(proposal.portfolio_id)
            client = Client.query.get(portfolio.client_id) if portfolio else None
            
            if not portfolio or not client:
                return AgentResponse(success=False, error="Portfolio or client not found")
            
            compliance_result = await self._perform_compliance_check(proposal, portfolio, client)
            
            return AgentResponse(success=True, data=compliance_result)
            
        except Exception as e:
            return AgentResponse(success=False, error=str(e))
    
    async def _perform_stress_test(self, content: Dict[str, Any]) -> AgentResponse:
        """Perform stress testing"""
        try:
            proposal_id = content.get("proposal_id")
            scenarios = content.get("scenarios", self.stress_test_scenarios)
            
            proposal = Proposal.query.get(proposal_id)
            if not proposal:
                return AgentResponse(success=False, error="Proposal not found")
            
            portfolio = Portfolio.query.get(proposal.portfolio_id)
            if not portfolio:
                return AgentResponse(success=False, error="Portfolio not found")
            
            stress_results = await self._run_stress_tests(proposal, portfolio)
            
            return AgentResponse(success=True, data=stress_results)
            
        except Exception as e:
            return AgentResponse(success=False, error=str(e))
    
    async def _validate_concentration(self, content: Dict[str, Any]) -> AgentResponse:
        """Validate concentration limits"""
        try:
            proposal_id = content.get("proposal_id")
            
            proposal = Proposal.query.get(proposal_id)
            if not proposal:
                return AgentResponse(success=False, error="Proposal not found")
            
            portfolio = Portfolio.query.get(proposal.portfolio_id)
            if not portfolio:
                return AgentResponse(success=False, error="Portfolio not found")
            
            concentration_result = await self._check_concentration_limits(proposal, portfolio)
            
            return AgentResponse(success=True, data=concentration_result)
            
        except Exception as e:
            return AgentResponse(success=False, error=str(e))
    
    async def _check_suitability(self, content: Dict[str, Any]) -> AgentResponse:
        """Check suitability of proposal for client"""
        try:
            proposal_id = content.get("proposal_id")
            
            proposal = Proposal.query.get(proposal_id)
            if not proposal:
                return AgentResponse(success=False, error="Proposal not found")
            
            portfolio = Portfolio.query.get(proposal.portfolio_id)
            client = Client.query.get(portfolio.client_id) if portfolio else None
            
            if not client:
                return AgentResponse(success=False, error="Client not found")
            
            suitability_result = await self._assess_suitability(proposal, client)
            
            return AgentResponse(success=True, data=suitability_result)
            
        except Exception as e:
            return AgentResponse(success=False, error=str(e))

