import pytest
import asyncio
import json
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, AsyncMock

from main import create_app


class TestAPIIntegration:
    """Integration tests for API endpoints"""
    
    @pytest.fixture
    def client(self, app):
        """Create test client"""
        return app.test_client()
    
    def test_health_endpoint(self, client):
        """Test health check endpoint"""
        response = client.get('/api/health')
        assert response.status_code == 200
        
        data = json.loads(response.data)
        assert 'status' in data
        assert 'timestamp' in data
        assert data['status'] == 'healthy'
    
    def test_client_creation_endpoint(self, client, sample_client_data):
        """Test client creation API endpoint"""
        response = client.post('/api/clients', 
                             json=sample_client_data,
                             content_type='application/json')
        
        # Should return 201 for successful creation
        assert response.status_code in [200, 201]
        
        if response.status_code == 201:
            data = json.loads(response.data)
            assert 'client_id' in data
            assert data['client_id'] == sample_client_data['client_id']
    
    def test_portfolio_creation_endpoint(self, client, sample_portfolio_data):
        """Test portfolio creation API endpoint"""
        response = client.post('/api/portfolios',
                             json=sample_portfolio_data,
                             content_type='application/json')
        
        assert response.status_code in [200, 201]
        
        if response.status_code == 201:
            data = json.loads(response.data)
            assert 'portfolio_id' in data
    
    def test_event_processing_endpoint(self, client, sample_event_data):
        """Test event processing API endpoint"""
        response = client.post('/api/events',
                             json=sample_event_data,
                             content_type='application/json')
        
        assert response.status_code in [200, 201, 202]  # 202 for async processing
    
    def test_agent_workflow_endpoint(self, client):
        """Test agent workflow API endpoint"""
        workflow_request = {
            'workflow_type': 'event_processing',
            'event_id': 'test_event_001',
            'client_id': 'test_client_001'
        }
        
        response = client.post('/api/agents/workflows',
                             json=workflow_request,
                             content_type='application/json')
        
        assert response.status_code in [200, 202]  # 202 for async processing
    
    def test_portfolio_optimization_endpoint(self, client):
        """Test portfolio optimization API endpoint"""
        optimization_request = {
            'portfolio_id': 'test_portfolio_001',
            'objective': 'max_sharpe',
            'constraints': {
                'max_volatility': 0.15,
                'min_weight': 0.05,
                'max_weight': 0.30
            }
        }
        
        response = client.post('/api/portfolios/optimize',
                             json=optimization_request,
                             content_type='application/json')
        
        assert response.status_code in [200, 202]
    
    def test_compliance_check_endpoint(self, client):
        """Test compliance check API endpoint"""
        compliance_request = {
            'client_id': 'test_client_001',
            'investment': {
                'symbol': 'AAPL',
                'amount': 50000,
                'asset_type': 'equity'
            }
        }
        
        response = client.post('/api/compliance/check',
                             json=compliance_request,
                             content_type='application/json')
        
        assert response.status_code == 200
    
    def test_error_handling(self, client):
        """Test API error handling"""
        # Test invalid JSON
        response = client.post('/api/clients',
                             data='invalid json',
                             content_type='application/json')
        
        assert response.status_code == 400
        
        # Test missing required fields
        response = client.post('/api/clients',
                             json={'incomplete': 'data'},
                             content_type='application/json')
        
        assert response.status_code in [400, 422]  # Validation error


class TestSystemIntegration:
    """Integration tests for system components"""
    
    @pytest.mark.asyncio
    async def test_end_to_end_event_processing(self, mock_openai_client):
        """Test complete event processing workflow"""
        with patch('agents.base_agent.OpenAI', return_value=mock_openai_client):
            from agents.orchestrator import AgentOrchestrator
            
            orchestrator = AgentOrchestrator()
            
            # Mock event data
            event_data = {
                'event_id': 'test_event_001',
                'event_type': 'market_event',
                'severity': 'medium',
                'title': 'Market Volatility Spike',
                'description': 'Unusual market volatility detected'
            }
            
            # Define workflow
            workflow_config = {
                'workflow_id': 'test_event_processing',
                'steps': [
                    {'agent': 'oracle', 'action': 'classify_event'},
                    {'agent': 'enricher', 'action': 'enrich_event'},
                    {'agent': 'proposer', 'action': 'generate_proposal'}
                ]
            }
            
            # Execute workflow
            result = await orchestrator.execute_workflow(workflow_config, event_data)
            
            assert 'workflow_id' in result
            assert 'execution_status' in result
            assert result['workflow_id'] == 'test_event_processing'
    
    @pytest.mark.asyncio
    async def test_portfolio_optimization_integration(self, sample_portfolio_data, sample_market_data):
        """Test portfolio optimization integration"""
        from portfolio_engine.optimizer import PortfolioOptimizer
        
        # Create optimizer with sample data
        optimizer = PortfolioOptimizer(sample_market_data)
        
        # Run optimization
        result = optimizer.optimize('max_sharpe')
        
        assert 'weights' in result
        assert 'expected_return' in result
        assert 'volatility' in result
        assert 'sharpe_ratio' in result
        
        # Validate results
        assert abs(sum(result['weights'].values()) - 1.0) < 1e-6
        assert result['volatility'] > 0
        assert result['expected_return'] > 0
    
    def test_compliance_integration(self, sample_client_data):
        """Test compliance system integration"""
        from compliance.kyc_aml_engine import KYCAMLEngine
        from compliance.suitability_engine import SuitabilityEngine
        
        # Test KYC/AML
        kyc_engine = KYCAMLEngine()
        kyc_result = kyc_engine.verify_client(sample_client_data)
        
        assert 'verification_status' in kyc_result
        assert 'risk_score' in kyc_result
        
        # Test suitability
        suitability_engine = SuitabilityEngine()
        investment = {
            'symbol': 'AAPL',
            'asset_type': 'equity',
            'risk_level': 'moderate',
            'amount': 50000
        }
        
        suitability_result = suitability_engine.assess_suitability(
            investment, sample_client_data
        )
        
        assert 'suitability_score' in suitability_result
        assert 'suitability_status' in suitability_result
    
    def test_data_flow_integration(self, sample_client_data, sample_portfolio_data):
        """Test data flow between components"""
        # Test client -> portfolio relationship
        assert sample_portfolio_data['client_id'] == sample_client_data['client_id']
        
        # Test portfolio value consistency
        assert sample_portfolio_data['total_value'] > 0
        assert sample_portfolio_data['cash_balance'] >= 0
        assert sample_portfolio_data['cash_balance'] <= sample_portfolio_data['total_value']
        
        # Test allocation consistency
        allocation_sum = sum(sample_portfolio_data['target_allocation'].values())
        assert abs(allocation_sum - 1.0) < 0.01


class TestPerformanceIntegration:
    """Performance and load testing"""
    
    def test_api_response_time(self, client):
        """Test API response times"""
        import time
        
        start_time = time.time()
        response = client.get('/api/health')
        end_time = time.time()
        
        response_time = end_time - start_time
        
        assert response.status_code == 200
        assert response_time < 1.0  # Should respond within 1 second
    
    def test_concurrent_requests(self, client):
        """Test concurrent API requests"""
        import threading
        import time
        
        results = []
        
        def make_request():
            response = client.get('/api/health')
            results.append(response.status_code)
        
        # Create multiple threads
        threads = []
        for _ in range(10):
            thread = threading.Thread(target=make_request)
            threads.append(thread)
        
        # Start all threads
        start_time = time.time()
        for thread in threads:
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        end_time = time.time()
        
        # Validate results
        assert len(results) == 10
        assert all(status == 200 for status in results)
        assert (end_time - start_time) < 5.0  # Should complete within 5 seconds
    
    def test_large_data_processing(self, sample_market_data):
        """Test processing of large datasets"""
        from portfolio_engine.risk_engine import RiskEngine
        
        # Create large dataset
        large_returns = sample_market_data.iloc[:, 0]  # Use first column
        
        # Create risk engine
        risk_engine = RiskEngine(large_returns)
        
        # Test VaR calculation with large dataset
        var_result = risk_engine.calculate_var(confidence_level=0.95)
        
        assert isinstance(var_result, float)
        assert var_result < 0  # VaR should be negative
    
    def test_memory_usage(self, sample_market_data):
        """Test memory usage with large datasets"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Process large dataset
        from portfolio_engine.simulator import MonteCarloSimulator
        
        portfolio_data = {
            'initial_value': 1000000,
            'expected_return': 0.08,
            'volatility': 0.15,
            'time_horizon': 10
        }
        
        simulator = MonteCarloSimulator(portfolio_data)
        results = simulator.run_simulation('geometric_brownian_motion', n_simulations=1000)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024  # 100MB
        
        # Validate results
        assert 'final_values' in results
        assert len(results['final_values']) == 1000


class TestSecurityIntegration:
    """Security and validation testing"""
    
    def test_input_validation(self, client):
        """Test input validation and sanitization"""
        # Test SQL injection attempt
        malicious_input = {
            'client_id': "'; DROP TABLE clients; --",
            'first_name': 'Test',
            'last_name': 'User',
            'email': 'test@example.com'
        }
        
        response = client.post('/api/clients',
                             json=malicious_input,
                             content_type='application/json')
        
        # Should either reject the input or sanitize it
        assert response.status_code in [400, 422]  # Validation error
    
    def test_data_sanitization(self, client):
        """Test data sanitization"""
        # Test XSS attempt
        xss_input = {
            'client_id': 'test_client_001',
            'first_name': '<script>alert("xss")</script>',
            'last_name': 'User',
            'email': 'test@example.com'
        }
        
        response = client.post('/api/clients',
                             json=xss_input,
                             content_type='application/json')
        
        # Should sanitize or reject malicious input
        if response.status_code == 201:
            data = json.loads(response.data)
            # Script tags should be removed or escaped
            assert '<script>' not in str(data)
    
    def test_rate_limiting(self, client):
        """Test rate limiting (if implemented)"""
        # Make multiple rapid requests
        responses = []
        for _ in range(100):
            response = client.get('/api/health')
            responses.append(response.status_code)
        
        # All requests should succeed (or some should be rate limited)
        success_count = sum(1 for status in responses if status == 200)
        rate_limited_count = sum(1 for status in responses if status == 429)
        
        # Either all succeed or some are rate limited
        assert success_count + rate_limited_count == 100
    
    def test_authentication_required(self, client):
        """Test authentication requirements (if implemented)"""
        # Test accessing protected endpoints without authentication
        protected_endpoints = [
            '/api/clients',
            '/api/portfolios',
            '/api/agents/workflows'
        ]
        
        for endpoint in protected_endpoints:
            response = client.get(endpoint)
            # Should either require authentication or be publicly accessible
            assert response.status_code in [200, 401, 403]


class TestErrorHandlingIntegration:
    """Error handling and recovery testing"""
    
    def test_database_error_handling(self, client):
        """Test database error handling"""
        # Test with invalid data that might cause database errors
        invalid_data = {
            'client_id': None,  # Invalid null value
            'first_name': 'Test',
            'last_name': 'User'
        }
        
        response = client.post('/api/clients',
                             json=invalid_data,
                             content_type='application/json')
        
        # Should handle database errors gracefully
        assert response.status_code in [400, 422, 500]
        
        if response.status_code == 500:
            # Should return proper error message
            data = json.loads(response.data)
            assert 'error' in data or 'message' in data
    
    def test_external_api_failure_handling(self, client):
        """Test external API failure handling"""
        with patch('requests.get') as mock_get:
            # Mock API failure
            mock_get.side_effect = Exception("API unavailable")
            
            # Test endpoint that depends on external API
            response = client.get('/api/market-data/AAPL')
            
            # Should handle external API failures gracefully
            assert response.status_code in [200, 503, 500]
    
    def test_timeout_handling(self, client):
        """Test timeout handling"""
        with patch('requests.get') as mock_get:
            # Mock slow response
            def slow_response(*args, **kwargs):
                import time
                time.sleep(10)  # Simulate slow response
                return Mock(status_code=200, json=lambda: {})
            
            mock_get.side_effect = slow_response
            
            # Test endpoint with timeout
            response = client.get('/api/market-data/AAPL')
            
            # Should handle timeouts gracefully
            assert response.status_code in [200, 408, 503, 500]
    
    @pytest.mark.asyncio
    async def test_agent_failure_recovery(self, mock_openai_client):
        """Test agent failure recovery"""
        with patch('agents.base_agent.OpenAI', return_value=mock_openai_client):
            from agents.orchestrator import AgentOrchestrator
            
            orchestrator = AgentOrchestrator()
            
            # Create workflow with failing step
            workflow_config = {
                'workflow_id': 'test_failure_recovery',
                'steps': [
                    {'agent': 'oracle', 'action': 'detect_events'},
                    {'agent': 'nonexistent_agent', 'action': 'invalid_action'},  # This will fail
                    {'agent': 'narrator', 'action': 'generate_report'}
                ]
            }
            
            # Execute workflow
            result = await orchestrator.execute_workflow(workflow_config, {})
            
            # Should handle agent failures gracefully
            assert 'execution_status' in result
            assert result['execution_status'] in ['failed', 'partial']
            assert 'error_details' in result

