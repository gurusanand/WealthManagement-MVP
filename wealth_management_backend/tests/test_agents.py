import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock, MagicMock

from agents.base_agent import BaseAgent, AgentMessage, AgentRegistry
from agents.oracle_agent import OracleAgent
from agents.enricher_agent import EnricherAgent
from agents.proposer_agent import ProposerAgent
from agents.checker_agent import CheckerAgent
from agents.executor_agent import ExecutorAgent
from agents.narrator_agent import NarratorAgent
from agents.orchestrator import AgentOrchestrator


class TestBaseAgent:
    """Test cases for Base Agent"""
    
    @pytest.fixture
    def base_agent(self, mock_openai_client):
        """Create base agent instance"""
        with patch('agents.base_agent.OpenAI', return_value=mock_openai_client):
            agent = BaseAgent("test_agent", "Test Agent for testing purposes")
            return agent
    
    def test_agent_initialization(self, base_agent):
        """Test agent initialization"""
        assert base_agent.agent_id == "test_agent"
        assert base_agent.description == "Test Agent for testing purposes"
        assert base_agent.status == "idle"
        assert base_agent.capabilities == []
        assert isinstance(base_agent.message_history, list)
    
    def test_agent_message_creation(self):
        """Test agent message creation"""
        message = AgentMessage(
            message_id="msg_001",
            sender="agent_1",
            recipient="agent_2",
            message_type="request",
            content={"action": "analyze", "data": "test_data"},
            timestamp=datetime.now()
        )
        
        assert message.message_id == "msg_001"
        assert message.sender == "agent_1"
        assert message.recipient == "agent_2"
        assert message.message_type == "request"
        assert message.content["action"] == "analyze"
    
    def test_agent_message_validation(self):
        """Test agent message validation"""
        # Valid message types
        valid_types = ["request", "response", "notification", "error", "heartbeat"]
        
        for msg_type in valid_types:
            message = AgentMessage(
                message_id=f"msg_{msg_type}",
                sender="agent_1",
                recipient="agent_2",
                message_type=msg_type,
                content={},
                timestamp=datetime.now()
            )
            assert message.message_type == msg_type
    
    @pytest.mark.asyncio
    async def test_agent_message_processing(self, base_agent):
        """Test agent message processing"""
        message = AgentMessage(
            message_id="msg_001",
            sender="test_sender",
            recipient=base_agent.agent_id,
            message_type="request",
            content={"action": "test", "data": "test_data"},
            timestamp=datetime.now()
        )
        
        response = await base_agent.process_message(message)
        
        assert response is not None
        assert response.sender == base_agent.agent_id
        assert response.recipient == message.sender
        assert response.message_type == "response"
    
    def test_agent_status_management(self, base_agent):
        """Test agent status management"""
        # Test status transitions
        base_agent.set_status("processing")
        assert base_agent.status == "processing"
        
        base_agent.set_status("idle")
        assert base_agent.status == "idle"
        
        # Test invalid status
        with pytest.raises(ValueError):
            base_agent.set_status("invalid_status")
    
    def test_agent_capability_management(self, base_agent):
        """Test agent capability management"""
        capabilities = ["data_analysis", "risk_assessment", "portfolio_optimization"]
        
        for capability in capabilities:
            base_agent.add_capability(capability)
        
        assert len(base_agent.capabilities) == len(capabilities)
        assert all(cap in base_agent.capabilities for cap in capabilities)
        
        # Test capability removal
        base_agent.remove_capability("risk_assessment")
        assert "risk_assessment" not in base_agent.capabilities
        assert len(base_agent.capabilities) == len(capabilities) - 1
    
    def test_agent_health_check(self, base_agent):
        """Test agent health check"""
        health_status = base_agent.get_health_status()
        
        assert "agent_id" in health_status
        assert "status" in health_status
        assert "uptime" in health_status
        assert "message_count" in health_status
        assert "last_activity" in health_status
        
        assert health_status["agent_id"] == base_agent.agent_id
        assert health_status["status"] == base_agent.status


class TestAgentRegistry:
    """Test cases for Agent Registry"""
    
    @pytest.fixture
    def registry(self):
        """Create agent registry instance"""
        return AgentRegistry()
    
    @pytest.fixture
    def sample_agents(self, mock_openai_client):
        """Create sample agents for testing"""
        with patch('agents.base_agent.OpenAI', return_value=mock_openai_client):
            agents = [
                BaseAgent("agent_1", "First test agent"),
                BaseAgent("agent_2", "Second test agent"),
                BaseAgent("agent_3", "Third test agent")
            ]
            return agents
    
    def test_agent_registration(self, registry, sample_agents):
        """Test agent registration"""
        for agent in sample_agents:
            registry.register_agent(agent)
        
        assert len(registry.agents) == len(sample_agents)
        
        for agent in sample_agents:
            assert agent.agent_id in registry.agents
            assert registry.agents[agent.agent_id] == agent
    
    def test_agent_discovery(self, registry, sample_agents):
        """Test agent discovery"""
        for agent in sample_agents:
            registry.register_agent(agent)
        
        # Test get agent by ID
        agent_1 = registry.get_agent("agent_1")
        assert agent_1 is not None
        assert agent_1.agent_id == "agent_1"
        
        # Test get non-existent agent
        non_existent = registry.get_agent("non_existent")
        assert non_existent is None
    
    def test_agent_capability_search(self, registry, sample_agents):
        """Test agent capability search"""
        # Add capabilities to agents
        sample_agents[0].add_capability("data_analysis")
        sample_agents[1].add_capability("data_analysis")
        sample_agents[1].add_capability("risk_assessment")
        sample_agents[2].add_capability("portfolio_optimization")
        
        for agent in sample_agents:
            registry.register_agent(agent)
        
        # Search for agents with specific capability
        data_analysts = registry.find_agents_by_capability("data_analysis")
        assert len(data_analysts) == 2
        assert all(agent.agent_id in ["agent_1", "agent_2"] for agent in data_analysts)
        
        risk_assessors = registry.find_agents_by_capability("risk_assessment")
        assert len(risk_assessors) == 1
        assert risk_assessors[0].agent_id == "agent_2"
    
    def test_agent_status_monitoring(self, registry, sample_agents):
        """Test agent status monitoring"""
        for agent in sample_agents:
            registry.register_agent(agent)
        
        # Set different statuses
        sample_agents[0].set_status("processing")
        sample_agents[1].set_status("idle")
        sample_agents[2].set_status("error")
        
        # Get agents by status
        processing_agents = registry.get_agents_by_status("processing")
        idle_agents = registry.get_agents_by_status("idle")
        error_agents = registry.get_agents_by_status("error")
        
        assert len(processing_agents) == 1
        assert len(idle_agents) == 1
        assert len(error_agents) == 1
        
        assert processing_agents[0].agent_id == "agent_1"
        assert idle_agents[0].agent_id == "agent_2"
        assert error_agents[0].agent_id == "agent_3"
    
    def test_registry_health_check(self, registry, sample_agents):
        """Test registry health check"""
        for agent in sample_agents:
            registry.register_agent(agent)
        
        health_report = registry.get_system_health()
        
        assert "total_agents" in health_report
        assert "agents_by_status" in health_report
        assert "system_uptime" in health_report
        
        assert health_report["total_agents"] == len(sample_agents)
        assert "idle" in health_report["agents_by_status"]


class TestOracleAgent:
    """Test cases for Oracle Agent"""
    
    @pytest.fixture
    def oracle_agent(self, mock_openai_client):
        """Create Oracle agent instance"""
        with patch('agents.oracle_agent.OpenAI', return_value=mock_openai_client):
            agent = OracleAgent()
            return agent
    
    @pytest.mark.asyncio
    async def test_news_event_detection(self, oracle_agent, sample_news_data):
        """Test news event detection"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {"articles": sample_news_data}
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            events = await oracle_agent.detect_news_events()
            
            assert isinstance(events, list)
            assert len(events) > 0
            
            for event in events:
                assert "event_id" in event
                assert "event_type" in event
                assert "severity" in event
                assert "confidence_score" in event
                assert event["event_type"] == "news_event"
    
    @pytest.mark.asyncio
    async def test_market_event_detection(self, oracle_agent, sample_market_data):
        """Test market event detection"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = {
                "Time Series (Daily)": {
                    "2024-01-15": {
                        "1. open": "150.00",
                        "2. high": "155.00",
                        "3. low": "148.00",
                        "4. close": "152.00",
                        "5. volume": "2000000"
                    }
                }
            }
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            events = await oracle_agent.detect_market_events("AAPL")
            
            assert isinstance(events, list)
            
            if events:  # If events detected
                for event in events:
                    assert "event_id" in event
                    assert "event_type" in event
                    assert event["event_type"] == "market_event"
    
    @pytest.mark.asyncio
    async def test_weather_event_detection(self, oracle_agent, sample_weather_data):
        """Test weather event detection"""
        with patch('requests.get') as mock_get:
            mock_response = Mock()
            mock_response.json.return_value = sample_weather_data
            mock_response.status_code = 200
            mock_get.return_value = mock_response
            
            events = await oracle_agent.detect_weather_events("New York")
            
            assert isinstance(events, list)
            
            if events:  # If events detected
                for event in events:
                    assert "event_id" in event
                    assert "event_type" in event
                    assert event["event_type"] == "weather_event"
    
    @pytest.mark.asyncio
    async def test_event_classification(self, oracle_agent):
        """Test event classification"""
        sample_event = {
            "title": "Federal Reserve Raises Interest Rates",
            "description": "The Fed announced a 0.25% rate increase",
            "source": "Reuters"
        }
        
        classification = await oracle_agent.classify_event(sample_event)
        
        assert "event_type" in classification
        assert "severity" in classification
        assert "confidence_score" in classification
        assert "affected_sectors" in classification
        
        assert classification["event_type"] in ["market_event", "economic_event", "news_event"]
        assert classification["severity"] in ["low", "medium", "high", "critical", "emergency"]
        assert 0 <= classification["confidence_score"] <= 1


class TestEnricherAgent:
    """Test cases for Enricher Agent"""
    
    @pytest.fixture
    def enricher_agent(self, mock_openai_client):
        """Create Enricher agent instance"""
        with patch('agents.enricher_agent.OpenAI', return_value=mock_openai_client):
            agent = EnricherAgent()
            return agent
    
    @pytest.mark.asyncio
    async def test_event_enrichment(self, enricher_agent, sample_event_data):
        """Test event enrichment"""
        enriched_event = await enricher_agent.enrich_event(sample_event_data)
        
        assert "enrichment" in enriched_event
        assert "market_context" in enriched_event["enrichment"]
        assert "sentiment_analysis" in enriched_event["enrichment"]
        assert "impact_assessment" in enriched_event["enrichment"]
        assert "historical_precedents" in enriched_event["enrichment"]
        
        # Validate sentiment analysis
        sentiment = enriched_event["enrichment"]["sentiment_analysis"]
        assert "sentiment_score" in sentiment
        assert "confidence" in sentiment
        assert -1 <= sentiment["sentiment_score"] <= 1
        assert 0 <= sentiment["confidence"] <= 1
    
    @pytest.mark.asyncio
    async def test_portfolio_impact_analysis(self, enricher_agent, sample_event_data, sample_portfolio_data):
        """Test portfolio impact analysis"""
        impact_analysis = await enricher_agent.analyze_portfolio_impact(
            sample_event_data, sample_portfolio_data
        )
        
        assert "portfolio_exposure" in impact_analysis
        assert "risk_impact" in impact_analysis
        assert "expected_return_impact" in impact_analysis
        assert "sector_impacts" in impact_analysis
        assert "recommendations" in impact_analysis
        
        # Validate impact scores
        assert isinstance(impact_analysis["risk_impact"], (int, float))
        assert isinstance(impact_analysis["expected_return_impact"], (int, float))
    
    @pytest.mark.asyncio
    async def test_sentiment_analysis(self, enricher_agent):
        """Test sentiment analysis"""
        positive_text = "Strong earnings growth and positive outlook for the company"
        negative_text = "Significant losses and declining market share"
        neutral_text = "The company reported quarterly results"
        
        positive_sentiment = await enricher_agent.analyze_sentiment(positive_text)
        negative_sentiment = await enricher_agent.analyze_sentiment(negative_text)
        neutral_sentiment = await enricher_agent.analyze_sentiment(neutral_text)
        
        # Validate sentiment scores
        assert positive_sentiment["sentiment_score"] > 0
        assert negative_sentiment["sentiment_score"] < 0
        assert abs(neutral_sentiment["sentiment_score"]) < 0.3  # Should be close to neutral
        
        # Validate confidence scores
        for sentiment in [positive_sentiment, negative_sentiment, neutral_sentiment]:
            assert 0 <= sentiment["confidence"] <= 1
    
    @pytest.mark.asyncio
    async def test_correlation_analysis(self, enricher_agent, sample_event_data):
        """Test correlation analysis"""
        correlations = await enricher_agent.analyze_correlations(sample_event_data)
        
        assert "market_correlations" in correlations
        assert "sector_correlations" in correlations
        assert "historical_correlations" in correlations
        
        # Validate correlation values
        for correlation_type, values in correlations.items():
            if isinstance(values, dict):
                for correlation_value in values.values():
                    if isinstance(correlation_value, (int, float)):
                        assert -1 <= correlation_value <= 1


class TestProposerAgent:
    """Test cases for Proposer Agent"""
    
    @pytest.fixture
    def proposer_agent(self, mock_openai_client):
        """Create Proposer agent instance"""
        with patch('agents.proposer_agent.OpenAI', return_value=mock_openai_client):
            agent = ProposerAgent()
            return agent
    
    @pytest.mark.asyncio
    async def test_proposal_generation(self, proposer_agent, sample_event_data, sample_portfolio_data):
        """Test proposal generation"""
        proposal = await proposer_agent.generate_proposal(sample_event_data, sample_portfolio_data)
        
        assert "proposal_id" in proposal
        assert "proposal_type" in proposal
        assert "recommendations" in proposal
        assert "expected_impact" in proposal
        assert "confidence_score" in proposal
        assert "rationale" in proposal
        
        # Validate proposal type
        valid_types = ["rebalancing", "hedging", "tactical_allocation", "risk_reduction", "opportunity_capture"]
        assert proposal["proposal_type"] in valid_types
        
        # Validate confidence score
        assert 0 <= proposal["confidence_score"] <= 1
    
    @pytest.mark.asyncio
    async def test_portfolio_optimization_proposal(self, proposer_agent, sample_portfolio_data):
        """Test portfolio optimization proposal"""
        optimization_request = {
            "objective": "max_sharpe",
            "constraints": {"max_volatility": 0.15},
            "rebalancing_threshold": 0.05
        }
        
        proposal = await proposer_agent.generate_optimization_proposal(
            sample_portfolio_data, optimization_request
        )
        
        assert "proposed_allocation" in proposal
        assert "current_allocation" in proposal
        assert "optimization_metrics" in proposal
        assert "trade_recommendations" in proposal
        
        # Validate proposed allocation sums to 1
        if proposal["proposed_allocation"]:
            allocation_sum = sum(proposal["proposed_allocation"].values())
            assert abs(allocation_sum - 1.0) < 0.01
    
    @pytest.mark.asyncio
    async def test_risk_mitigation_proposal(self, proposer_agent, sample_portfolio_data):
        """Test risk mitigation proposal"""
        risk_scenario = {
            "risk_type": "market_volatility",
            "severity": "high",
            "affected_sectors": ["technology", "growth"]
        }
        
        proposal = await proposer_agent.generate_risk_mitigation_proposal(
            sample_portfolio_data, risk_scenario
        )
        
        assert "mitigation_strategies" in proposal
        assert "hedge_recommendations" in proposal
        assert "position_adjustments" in proposal
        assert "expected_risk_reduction" in proposal
        
        # Validate risk reduction is positive
        if proposal["expected_risk_reduction"]:
            assert proposal["expected_risk_reduction"] > 0
    
    @pytest.mark.asyncio
    async def test_proposal_prioritization(self, proposer_agent):
        """Test proposal prioritization"""
        proposals = [
            {"proposal_id": "prop_1", "confidence_score": 0.8, "expected_impact": {"return": 0.02}},
            {"proposal_id": "prop_2", "confidence_score": 0.6, "expected_impact": {"return": 0.05}},
            {"proposal_id": "prop_3", "confidence_score": 0.9, "expected_impact": {"return": 0.01}}
        ]
        
        prioritized = await proposer_agent.prioritize_proposals(proposals)
        
        assert len(prioritized) == len(proposals)
        assert "priority_score" in prioritized[0]
        
        # Validate prioritization order (higher priority scores first)
        for i in range(len(prioritized) - 1):
            assert prioritized[i]["priority_score"] >= prioritized[i + 1]["priority_score"]


class TestCheckerAgent:
    """Test cases for Checker Agent"""
    
    @pytest.fixture
    def checker_agent(self, mock_openai_client):
        """Create Checker agent instance"""
        with patch('agents.checker_agent.OpenAI', return_value=mock_openai_client):
            agent = CheckerAgent()
            return agent
    
    @pytest.mark.asyncio
    async def test_compliance_validation(self, checker_agent, sample_client_data):
        """Test compliance validation"""
        proposal = {
            "proposal_id": "prop_001",
            "client_id": sample_client_data["client_id"],
            "recommendations": [
                {"action": "buy", "symbol": "AAPL", "quantity": 100}
            ]
        }
        
        validation_result = await checker_agent.validate_compliance(proposal, sample_client_data)
        
        assert "compliance_status" in validation_result
        assert "violations" in validation_result
        assert "risk_assessment" in validation_result
        assert "approval_required" in validation_result
        
        # Validate compliance status
        valid_statuses = ["compliant", "non_compliant", "requires_review", "conditional_approval"]
        assert validation_result["compliance_status"] in valid_statuses
    
    @pytest.mark.asyncio
    async def test_suitability_check(self, checker_agent, sample_client_data):
        """Test investment suitability check"""
        investment = {
            "symbol": "TSLA",
            "asset_type": "equity",
            "risk_level": "high",
            "complexity": "moderate",
            "amount": 50000
        }
        
        suitability_result = await checker_agent.check_suitability(investment, sample_client_data)
        
        assert "suitability_score" in suitability_result
        assert "suitability_status" in suitability_result
        assert "risk_alignment" in suitability_result
        assert "recommendations" in suitability_result
        
        # Validate suitability score
        assert 0 <= suitability_result["suitability_score"] <= 1
        
        # Validate suitability status
        valid_statuses = ["suitable", "unsuitable", "marginal", "requires_disclosure"]
        assert suitability_result["suitability_status"] in valid_statuses
    
    @pytest.mark.asyncio
    async def test_risk_limit_validation(self, checker_agent, sample_portfolio_data):
        """Test risk limit validation"""
        proposal = {
            "proposed_allocation": {"equities": 0.8, "bonds": 0.2},
            "expected_volatility": 0.18,
            "expected_var": -0.05
        }
        
        risk_limits = {
            "max_volatility": 0.15,
            "max_var": -0.03,
            "max_equity_allocation": 0.7
        }
        
        validation_result = await checker_agent.validate_risk_limits(
            proposal, sample_portfolio_data, risk_limits
        )
        
        assert "limit_violations" in validation_result
        assert "risk_status" in validation_result
        assert "override_required" in validation_result
        
        # Check for expected violations
        violations = validation_result["limit_violations"]
        assert any("volatility" in v["limit_type"] for v in violations)
        assert any("equity_allocation" in v["limit_type"] for v in violations)
    
    @pytest.mark.asyncio
    async def test_regulatory_compliance(self, checker_agent, sample_client_data):
        """Test regulatory compliance check"""
        transaction = {
            "transaction_type": "buy",
            "symbol": "AAPL",
            "quantity": 1000,
            "value": 150000,
            "client_id": sample_client_data["client_id"]
        }
        
        compliance_result = await checker_agent.check_regulatory_compliance(
            transaction, sample_client_data
        )
        
        assert "regulatory_status" in compliance_result
        assert "required_disclosures" in compliance_result
        assert "reporting_requirements" in compliance_result
        assert "approval_workflow" in compliance_result
        
        # Validate regulatory status
        valid_statuses = ["compliant", "requires_disclosure", "requires_approval", "prohibited"]
        assert compliance_result["regulatory_status"] in valid_statuses


class TestExecutorAgent:
    """Test cases for Executor Agent"""
    
    @pytest.fixture
    def executor_agent(self, mock_openai_client):
        """Create Executor agent instance"""
        with patch('agents.executor_agent.OpenAI', return_value=mock_openai_client):
            agent = ExecutorAgent()
            return agent
    
    @pytest.mark.asyncio
    async def test_proposal_execution(self, executor_agent, sample_portfolio_data):
        """Test proposal execution"""
        proposal = {
            "proposal_id": "prop_001",
            "recommendations": [
                {"action": "sell", "symbol": "AAPL", "quantity": 100},
                {"action": "buy", "symbol": "MSFT", "quantity": 50}
            ],
            "client_id": "test_client_001",
            "portfolio_id": sample_portfolio_data["portfolio_id"]
        }
        
        execution_result = await executor_agent.execute_proposal(proposal)
        
        assert "execution_id" in execution_result
        assert "execution_status" in execution_result
        assert "executed_trades" in execution_result
        assert "execution_summary" in execution_result
        
        # Validate execution status
        valid_statuses = ["completed", "partial", "failed", "pending"]
        assert execution_result["execution_status"] in valid_statuses
    
    @pytest.mark.asyncio
    async def test_trade_execution(self, executor_agent):
        """Test individual trade execution"""
        trade = {
            "trade_id": "trade_001",
            "action": "buy",
            "symbol": "AAPL",
            "quantity": 100,
            "order_type": "market",
            "portfolio_id": "portfolio_001"
        }
        
        execution_result = await executor_agent.execute_trade(trade)
        
        assert "trade_id" in execution_result
        assert "execution_status" in execution_result
        assert "execution_price" in execution_result
        assert "execution_time" in execution_result
        assert "transaction_costs" in execution_result
        
        # Validate execution details
        if execution_result["execution_status"] == "completed":
            assert execution_result["execution_price"] > 0
            assert execution_result["transaction_costs"] >= 0
    
    @pytest.mark.asyncio
    async def test_portfolio_rebalancing(self, executor_agent, sample_portfolio_data):
        """Test portfolio rebalancing execution"""
        target_allocation = {
            "AAPL": 0.25,
            "MSFT": 0.25,
            "GOOGL": 0.25,
            "AMZN": 0.25
        }
        
        rebalancing_result = await executor_agent.execute_rebalancing(
            sample_portfolio_data["portfolio_id"], target_allocation
        )
        
        assert "rebalancing_id" in rebalancing_result
        assert "trades_executed" in rebalancing_result
        assert "final_allocation" in rebalancing_result
        assert "rebalancing_costs" in rebalancing_result
        
        # Validate final allocation
        if rebalancing_result["final_allocation"]:
            allocation_sum = sum(rebalancing_result["final_allocation"].values())
            assert abs(allocation_sum - 1.0) < 0.01
    
    @pytest.mark.asyncio
    async def test_execution_monitoring(self, executor_agent):
        """Test execution monitoring"""
        execution_id = "exec_001"
        
        monitoring_result = await executor_agent.monitor_execution(execution_id)
        
        assert "execution_id" in monitoring_result
        assert "current_status" in monitoring_result
        assert "progress" in monitoring_result
        assert "estimated_completion" in monitoring_result
        
        # Validate progress
        assert 0 <= monitoring_result["progress"] <= 1


class TestNarratorAgent:
    """Test cases for Narrator Agent"""
    
    @pytest.fixture
    def narrator_agent(self, mock_openai_client):
        """Create Narrator agent instance"""
        with patch('agents.narrator_agent.OpenAI', return_value=mock_openai_client):
            agent = NarratorAgent()
            return agent
    
    @pytest.mark.asyncio
    async def test_client_communication_generation(self, narrator_agent, sample_client_data):
        """Test client communication generation"""
        event = {
            "event_type": "market_event",
            "title": "Market Volatility Increase",
            "impact": "Portfolio value may fluctuate more than usual"
        }
        
        communication = await narrator_agent.generate_client_communication(
            event, sample_client_data
        )
        
        assert "communication_id" in communication
        assert "subject" in communication
        assert "message" in communication
        assert "tone" in communication
        assert "urgency" in communication
        assert "delivery_method" in communication
        
        # Validate tone and urgency
        valid_tones = ["professional", "reassuring", "urgent", "informative", "personal"]
        valid_urgencies = ["low", "medium", "high", "critical"]
        
        assert communication["tone"] in valid_tones
        assert communication["urgency"] in valid_urgencies
    
    @pytest.mark.asyncio
    async def test_portfolio_report_generation(self, narrator_agent, sample_portfolio_data, sample_client_data):
        """Test portfolio report generation"""
        report_type = "monthly"
        
        report = await narrator_agent.generate_portfolio_report(
            sample_portfolio_data, sample_client_data, report_type
        )
        
        assert "report_id" in report
        assert "report_type" in report
        assert "executive_summary" in report
        assert "performance_analysis" in report
        assert "market_commentary" in report
        assert "recommendations" in report
        
        # Validate report content
        assert len(report["executive_summary"]) > 0
        assert len(report["performance_analysis"]) > 0
    
    @pytest.mark.asyncio
    async def test_explanation_generation(self, narrator_agent):
        """Test explanation generation"""
        complex_concept = {
            "topic": "Value at Risk (VaR)",
            "context": "risk_management",
            "audience": "retail_client"
        }
        
        explanation = await narrator_agent.generate_explanation(complex_concept)
        
        assert "explanation" in explanation
        assert "key_points" in explanation
        assert "examples" in explanation
        assert "complexity_level" in explanation
        
        # Validate complexity level
        valid_levels = ["basic", "intermediate", "advanced", "expert"]
        assert explanation["complexity_level"] in valid_levels
    
    @pytest.mark.asyncio
    async def test_tone_adaptation(self, narrator_agent, sample_client_data):
        """Test tone adaptation based on client profile"""
        message_content = "Your portfolio has experienced some volatility this month."
        
        # Test different client types
        conservative_client = {**sample_client_data, "risk_tolerance": "conservative"}
        aggressive_client = {**sample_client_data, "risk_tolerance": "aggressive"}
        
        conservative_message = await narrator_agent.adapt_tone(message_content, conservative_client)
        aggressive_message = await narrator_agent.adapt_tone(message_content, aggressive_client)
        
        assert "adapted_message" in conservative_message
        assert "tone_used" in conservative_message
        assert "adapted_message" in aggressive_message
        assert "tone_used" in aggressive_message
        
        # Messages should be different for different risk tolerances
        assert conservative_message["adapted_message"] != aggressive_message["adapted_message"]


class TestAgentOrchestrator:
    """Test cases for Agent Orchestrator"""
    
    @pytest.fixture
    def orchestrator(self, mock_openai_client):
        """Create agent orchestrator instance"""
        with patch('agents.base_agent.OpenAI', return_value=mock_openai_client):
            orchestrator = AgentOrchestrator()
            return orchestrator
    
    @pytest.mark.asyncio
    async def test_workflow_execution(self, orchestrator, sample_event_data):
        """Test workflow execution"""
        workflow_config = {
            "workflow_id": "event_processing",
            "steps": [
                {"agent": "oracle", "action": "detect_events"},
                {"agent": "enricher", "action": "enrich_event"},
                {"agent": "proposer", "action": "generate_proposal"},
                {"agent": "checker", "action": "validate_proposal"},
                {"agent": "executor", "action": "execute_proposal"},
                {"agent": "narrator", "action": "communicate_results"}
            ]
        }
        
        workflow_result = await orchestrator.execute_workflow(workflow_config, sample_event_data)
        
        assert "workflow_id" in workflow_result
        assert "execution_status" in workflow_result
        assert "step_results" in workflow_result
        assert "execution_time" in workflow_result
        
        # Validate workflow execution
        valid_statuses = ["completed", "failed", "partial", "in_progress"]
        assert workflow_result["execution_status"] in valid_statuses
    
    @pytest.mark.asyncio
    async def test_agent_coordination(self, orchestrator):
        """Test agent coordination"""
        coordination_request = {
            "primary_agent": "proposer",
            "supporting_agents": ["enricher", "checker"],
            "task": "portfolio_optimization",
            "data": {"portfolio_id": "portfolio_001"}
        }
        
        coordination_result = await orchestrator.coordinate_agents(coordination_request)
        
        assert "coordination_id" in coordination_result
        assert "agent_assignments" in coordination_result
        assert "coordination_status" in coordination_result
        
        # Validate agent assignments
        assignments = coordination_result["agent_assignments"]
        assert "proposer" in assignments
        assert "enricher" in assignments
        assert "checker" in assignments
    
    @pytest.mark.asyncio
    async def test_workflow_monitoring(self, orchestrator):
        """Test workflow monitoring"""
        workflow_id = "workflow_001"
        
        monitoring_result = await orchestrator.monitor_workflow(workflow_id)
        
        assert "workflow_id" in monitoring_result
        assert "current_step" in monitoring_result
        assert "progress" in monitoring_result
        assert "estimated_completion" in monitoring_result
        
        # Validate progress
        assert 0 <= monitoring_result["progress"] <= 1
    
    @pytest.mark.asyncio
    async def test_error_handling(self, orchestrator):
        """Test error handling in workflows"""
        # Create a workflow with an invalid step
        invalid_workflow = {
            "workflow_id": "invalid_workflow",
            "steps": [
                {"agent": "nonexistent_agent", "action": "invalid_action"}
            ]
        }
        
        workflow_result = await orchestrator.execute_workflow(invalid_workflow, {})
        
        assert workflow_result["execution_status"] == "failed"
        assert "error_details" in workflow_result
    
    def test_workflow_configuration_validation(self, orchestrator):
        """Test workflow configuration validation"""
        # Valid workflow
        valid_workflow = {
            "workflow_id": "valid_workflow",
            "steps": [
                {"agent": "oracle", "action": "detect_events"}
            ]
        }
        
        # Invalid workflow (missing required fields)
        invalid_workflow = {
            "steps": [
                {"agent": "oracle"}  # Missing action
            ]
        }
        
        assert orchestrator.validate_workflow_config(valid_workflow) == True
        assert orchestrator.validate_workflow_config(invalid_workflow) == False

