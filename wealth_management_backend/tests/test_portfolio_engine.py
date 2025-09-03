import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from unittest.mock import Mock, patch, MagicMock

from portfolio_engine.optimizer import PortfolioOptimizer
from portfolio_engine.risk_engine import RiskEngine
from portfolio_engine.simulator import MonteCarloSimulator
from portfolio_engine.performance import PerformanceAnalyzer
from portfolio_engine.factor_models import FactorModels


class TestPortfolioOptimizer:
    """Test cases for Portfolio Optimizer"""
    
    @pytest.fixture
    def sample_returns_data(self):
        """Sample returns data for optimization testing"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        assets = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
        
        # Generate correlated returns
        n_assets = len(assets)
        correlation_matrix = np.eye(n_assets) * 0.6 + np.ones((n_assets, n_assets)) * 0.2
        np.fill_diagonal(correlation_matrix, 1.0)
        
        returns = np.random.multivariate_normal(
            mean=[0.0008] * n_assets,
            cov=correlation_matrix * 0.02**2,
            size=len(dates)
        )
        
        return pd.DataFrame(returns, index=dates, columns=assets)
    
    @pytest.fixture
    def optimizer(self, sample_returns_data):
        """Create portfolio optimizer instance"""
        return PortfolioOptimizer(sample_returns_data)
    
    def test_optimizer_initialization(self, optimizer, sample_returns_data):
        """Test optimizer initialization"""
        assert optimizer.returns_data is not None
        assert len(optimizer.returns_data.columns) == len(sample_returns_data.columns)
        assert optimizer.risk_free_rate == 0.02  # Default value
    
    def test_sharpe_ratio_optimization(self, optimizer):
        """Test Sharpe ratio maximization"""
        result = optimizer.optimize('max_sharpe')
        
        assert 'weights' in result
        assert 'expected_return' in result
        assert 'volatility' in result
        assert 'sharpe_ratio' in result
        
        # Validate weights sum to 1
        assert abs(sum(result['weights'].values()) - 1.0) < 1e-6
        
        # Validate all weights are non-negative (long-only)
        assert all(w >= -1e-6 for w in result['weights'].values())
        
        # Validate Sharpe ratio calculation
        expected_sharpe = (result['expected_return'] - optimizer.risk_free_rate) / result['volatility']
        assert abs(result['sharpe_ratio'] - expected_sharpe) < 1e-6
    
    def test_minimum_variance_optimization(self, optimizer):
        """Test minimum variance optimization"""
        result = optimizer.optimize('min_variance')
        
        assert 'weights' in result
        assert 'volatility' in result
        
        # Validate weights sum to 1
        assert abs(sum(result['weights'].values()) - 1.0) < 1e-6
        
        # Validate volatility is positive
        assert result['volatility'] > 0
    
    def test_maximum_return_optimization(self, optimizer):
        """Test maximum return optimization"""
        result = optimizer.optimize('max_return', constraints={'max_volatility': 0.20})
        
        assert 'weights' in result
        assert 'expected_return' in result
        assert 'volatility' in result
        
        # Validate volatility constraint
        assert result['volatility'] <= 0.20 + 1e-6  # Allow small numerical error
    
    def test_risk_parity_optimization(self, optimizer):
        """Test risk parity optimization"""
        result = optimizer.optimize('risk_parity')
        
        assert 'weights' in result
        assert 'risk_contributions' in result
        
        # Validate weights sum to 1
        assert abs(sum(result['weights'].values()) - 1.0) < 1e-6
        
        # Validate risk contributions are approximately equal
        risk_contribs = list(result['risk_contributions'].values())
        target_contrib = 1.0 / len(risk_contribs)
        
        for contrib in risk_contribs:
            assert abs(contrib - target_contrib) < 0.1  # Allow some deviation
    
    def test_esg_optimization(self, optimizer):
        """Test ESG-optimized portfolio"""
        # Mock ESG scores
        esg_scores = {asset: np.random.uniform(0.3, 0.9) for asset in optimizer.returns_data.columns}
        
        result = optimizer.optimize('esg_optimized', esg_scores=esg_scores, min_esg_score=0.5)
        
        assert 'weights' in result
        assert 'esg_score' in result
        
        # Validate ESG score constraint
        assert result['esg_score'] >= 0.5 - 1e-6
    
    def test_black_litterman_optimization(self, optimizer):
        """Test Black-Litterman optimization"""
        # Mock market views
        views = {
            'AAPL': {'expected_return': 0.12, 'confidence': 0.8},
            'MSFT': {'expected_return': 0.10, 'confidence': 0.6}
        }
        
        result = optimizer.optimize('black_litterman', market_views=views)
        
        assert 'weights' in result
        assert 'expected_return' in result
        assert 'bl_returns' in result
    
    def test_constraints_validation(self, optimizer):
        """Test constraint validation"""
        constraints = {
            'min_weight': 0.05,
            'max_weight': 0.30,
            'sector_limits': {'technology': 0.40},
            'max_volatility': 0.15
        }
        
        result = optimizer.optimize('max_sharpe', constraints=constraints)
        
        # Validate weight constraints
        for weight in result['weights'].values():
            assert weight >= constraints['min_weight'] - 1e-6
            assert weight <= constraints['max_weight'] + 1e-6
        
        # Validate volatility constraint
        assert result['volatility'] <= constraints['max_volatility'] + 1e-6
    
    def test_transaction_costs(self, optimizer):
        """Test transaction cost integration"""
        current_weights = {asset: 0.2 for asset in optimizer.returns_data.columns}
        transaction_cost = 0.001  # 10 bps
        
        result = optimizer.optimize('max_sharpe', 
                                  current_weights=current_weights,
                                  transaction_cost=transaction_cost)
        
        assert 'weights' in result
        assert 'transaction_costs' in result
        assert 'net_expected_return' in result
        
        # Validate transaction costs are calculated
        assert result['transaction_costs'] >= 0
    
    def test_optimization_failure_handling(self, optimizer):
        """Test handling of optimization failures"""
        # Create impossible constraints
        impossible_constraints = {
            'min_weight': 0.5,  # Each asset must be at least 50%
            'max_weight': 0.6   # But no more than 60%
        }
        
        with pytest.raises(Exception):
            optimizer.optimize('max_sharpe', constraints=impossible_constraints)


class TestRiskEngine:
    """Test cases for Risk Engine"""
    
    @pytest.fixture
    def sample_portfolio_returns(self):
        """Sample portfolio returns for risk testing"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        returns = np.random.normal(0.0008, 0.02, len(dates))
        return pd.Series(returns, index=dates)
    
    @pytest.fixture
    def risk_engine(self, sample_portfolio_returns):
        """Create risk engine instance"""
        return RiskEngine(sample_portfolio_returns)
    
    def test_var_calculation(self, risk_engine):
        """Test Value at Risk calculation"""
        var_95 = risk_engine.calculate_var(confidence_level=0.95)
        var_99 = risk_engine.calculate_var(confidence_level=0.99)
        
        assert var_95 < 0  # VaR should be negative (loss)
        assert var_99 < var_95  # 99% VaR should be more negative than 95% VaR
        assert isinstance(var_95, float)
        assert isinstance(var_99, float)
    
    def test_cvar_calculation(self, risk_engine):
        """Test Conditional Value at Risk calculation"""
        cvar_95 = risk_engine.calculate_cvar(confidence_level=0.95)
        var_95 = risk_engine.calculate_var(confidence_level=0.95)
        
        assert cvar_95 < var_95  # CVaR should be more negative than VaR
        assert isinstance(cvar_95, float)
    
    def test_drawdown_analysis(self, risk_engine):
        """Test drawdown analysis"""
        drawdown_metrics = risk_engine.analyze_drawdowns()
        
        assert 'max_drawdown' in drawdown_metrics
        assert 'max_drawdown_duration' in drawdown_metrics
        assert 'recovery_time' in drawdown_metrics
        assert 'drawdown_series' in drawdown_metrics
        
        # Validate drawdown is negative or zero
        assert drawdown_metrics['max_drawdown'] <= 0
        
        # Validate duration is positive
        assert drawdown_metrics['max_drawdown_duration'] >= 0
    
    def test_risk_metrics(self, risk_engine):
        """Test comprehensive risk metrics"""
        metrics = risk_engine.calculate_risk_metrics()
        
        expected_metrics = ['volatility', 'var_95', 'var_99', 'cvar_95', 'cvar_99',
                          'max_drawdown', 'sharpe_ratio', 'sortino_ratio', 'calmar_ratio']
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
    
    def test_stress_testing(self, risk_engine):
        """Test stress testing scenarios"""
        stress_results = risk_engine.run_stress_tests()
        
        assert isinstance(stress_results, dict)
        assert len(stress_results) > 0
        
        # Check that predefined scenarios exist
        expected_scenarios = ['market_crash', 'interest_rate_shock', 'credit_crisis']
        
        for scenario in expected_scenarios:
            if scenario in stress_results:
                assert 'portfolio_impact' in stress_results[scenario]
                assert 'sector_impacts' in stress_results[scenario]
                assert 'recovery_estimate' in stress_results[scenario]
    
    def test_custom_stress_scenario(self, risk_engine):
        """Test custom stress scenario"""
        custom_scenario = {
            'name': 'custom_test',
            'market_shock': -0.20,
            'volatility_multiplier': 2.0,
            'correlation_increase': 0.3
        }
        
        result = risk_engine.run_custom_stress_test(custom_scenario)
        
        assert 'portfolio_impact' in result
        assert 'scenario_details' in result
        assert result['portfolio_impact'] < 0  # Should be negative impact
    
    def test_factor_risk_attribution(self, risk_engine):
        """Test factor risk attribution"""
        # Mock factor exposures
        factor_exposures = {
            'market': 0.8,
            'size': 0.2,
            'value': -0.1,
            'momentum': 0.3,
            'quality': 0.4,
            'volatility': -0.2
        }
        
        attribution = risk_engine.calculate_factor_risk_attribution(factor_exposures)
        
        assert 'factor_contributions' in attribution
        assert 'total_risk' in attribution
        assert 'idiosyncratic_risk' in attribution
        
        # Validate risk contributions sum approximately to total risk
        total_factor_risk = sum(attribution['factor_contributions'].values())
        expected_total = (total_factor_risk**2 + attribution['idiosyncratic_risk']**2)**0.5
        assert abs(attribution['total_risk'] - expected_total) < 0.01


class TestMonteCarloSimulator:
    """Test cases for Monte Carlo Simulator"""
    
    @pytest.fixture
    def sample_portfolio_data(self):
        """Sample portfolio data for simulation"""
        return {
            'initial_value': 1000000,
            'expected_return': 0.08,
            'volatility': 0.15,
            'time_horizon': 10,  # years
            'rebalancing_frequency': 'quarterly'
        }
    
    @pytest.fixture
    def simulator(self, sample_portfolio_data):
        """Create Monte Carlo simulator instance"""
        return MonteCarloSimulator(sample_portfolio_data)
    
    def test_simulator_initialization(self, simulator, sample_portfolio_data):
        """Test simulator initialization"""
        assert simulator.initial_value == sample_portfolio_data['initial_value']
        assert simulator.expected_return == sample_portfolio_data['expected_return']
        assert simulator.volatility == sample_portfolio_data['volatility']
        assert simulator.time_horizon == sample_portfolio_data['time_horizon']
    
    def test_geometric_brownian_motion_simulation(self, simulator):
        """Test Geometric Brownian Motion simulation"""
        n_simulations = 1000
        results = simulator.run_simulation(
            method='geometric_brownian_motion',
            n_simulations=n_simulations
        )
        
        assert 'final_values' in results
        assert 'paths' in results
        assert 'statistics' in results
        
        # Validate number of simulations
        assert len(results['final_values']) == n_simulations
        
        # Validate statistics
        stats = results['statistics']
        assert 'mean' in stats
        assert 'median' in stats
        assert 'std' in stats
        assert 'percentiles' in stats
        
        # Validate final values are positive
        assert all(value > 0 for value in results['final_values'])
    
    def test_historical_bootstrap_simulation(self, simulator):
        """Test historical bootstrap simulation"""
        # Mock historical returns
        np.random.seed(42)
        historical_returns = pd.Series(np.random.normal(0.0008, 0.02, 1000))
        
        results = simulator.run_simulation(
            method='historical_bootstrap',
            n_simulations=500,
            historical_returns=historical_returns
        )
        
        assert 'final_values' in results
        assert 'statistics' in results
        assert len(results['final_values']) == 500
    
    def test_factor_model_simulation(self, simulator):
        """Test factor model simulation"""
        # Mock factor model parameters
        factor_loadings = {
            'market': 0.8,
            'size': 0.2,
            'value': -0.1
        }
        
        factor_returns = pd.DataFrame({
            'market': np.random.normal(0.0008, 0.02, 1000),
            'size': np.random.normal(0.0002, 0.015, 1000),
            'value': np.random.normal(0.0001, 0.018, 1000)
        })
        
        results = simulator.run_simulation(
            method='factor_model',
            n_simulations=500,
            factor_loadings=factor_loadings,
            factor_returns=factor_returns
        )
        
        assert 'final_values' in results
        assert 'factor_contributions' in results
        assert len(results['final_values']) == 500
    
    def test_goal_based_analysis(self, simulator):
        """Test goal-based analysis"""
        target_value = 2000000  # Double the initial investment
        
        results = simulator.run_goal_analysis(
            target_value=target_value,
            n_simulations=1000
        )
        
        assert 'success_probability' in results
        assert 'expected_shortfall' in results
        assert 'time_to_target' in results
        
        # Validate probability is between 0 and 1
        assert 0 <= results['success_probability'] <= 1
        
        # Validate expected shortfall
        if results['expected_shortfall'] is not None:
            assert results['expected_shortfall'] >= 0
    
    def test_scenario_analysis(self, simulator):
        """Test scenario analysis"""
        scenarios = [
            {'name': 'bull_market', 'return_adjustment': 0.03, 'volatility_multiplier': 0.8},
            {'name': 'bear_market', 'return_adjustment': -0.05, 'volatility_multiplier': 1.5},
            {'name': 'base_case', 'return_adjustment': 0.0, 'volatility_multiplier': 1.0}
        ]
        
        results = simulator.run_scenario_analysis(scenarios, n_simulations=500)
        
        assert isinstance(results, dict)
        assert len(results) == len(scenarios)
        
        for scenario in scenarios:
            scenario_name = scenario['name']
            assert scenario_name in results
            assert 'final_values' in results[scenario_name]
            assert 'statistics' in results[scenario_name]
    
    def test_distribution_fitting(self, simulator):
        """Test distribution fitting"""
        # Generate sample returns
        np.random.seed(42)
        returns = np.random.normal(0.0008, 0.02, 1000)
        
        best_distribution = simulator.fit_distribution(returns)
        
        assert 'distribution' in best_distribution
        assert 'parameters' in best_distribution
        assert 'goodness_of_fit' in best_distribution
        
        # Validate distribution name
        valid_distributions = ['normal', 't', 'skewnorm']
        assert best_distribution['distribution'] in valid_distributions


class TestPerformanceAnalyzer:
    """Test cases for Performance Analyzer"""
    
    @pytest.fixture
    def sample_performance_data(self):
        """Sample performance data for analysis"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        
        # Portfolio returns
        portfolio_returns = pd.Series(
            np.random.normal(0.0008, 0.02, len(dates)),
            index=dates
        )
        
        # Benchmark returns
        benchmark_returns = pd.Series(
            np.random.normal(0.0006, 0.018, len(dates)),
            index=dates
        )
        
        return portfolio_returns, benchmark_returns
    
    @pytest.fixture
    def analyzer(self, sample_performance_data):
        """Create performance analyzer instance"""
        portfolio_returns, benchmark_returns = sample_performance_data
        return PerformanceAnalyzer(portfolio_returns, benchmark_returns)
    
    def test_basic_performance_metrics(self, analyzer):
        """Test basic performance metrics calculation"""
        metrics = analyzer.calculate_performance_metrics()
        
        expected_metrics = ['total_return', 'annualized_return', 'volatility',
                          'sharpe_ratio', 'sortino_ratio', 'max_drawdown',
                          'calmar_ratio', 'alpha', 'beta', 'tracking_error',
                          'information_ratio']
        
        for metric in expected_metrics:
            assert metric in metrics
            assert isinstance(metrics[metric], (int, float))
        
        # Validate specific metric properties
        assert metrics['volatility'] >= 0
        assert -1 <= metrics['max_drawdown'] <= 0
        assert metrics['beta'] > 0  # Assuming positive correlation with benchmark
    
    def test_attribution_analysis(self, analyzer):
        """Test Brinson attribution analysis"""
        # Mock allocation and selection data
        portfolio_weights = pd.DataFrame({
            'Technology': [0.3, 0.32, 0.35],
            'Healthcare': [0.2, 0.18, 0.15],
            'Financials': [0.25, 0.25, 0.25],
            'Consumer': [0.25, 0.25, 0.25]
        }, index=pd.date_range('2023-01-01', periods=3, freq='Q'))
        
        benchmark_weights = pd.DataFrame({
            'Technology': [0.25, 0.25, 0.25],
            'Healthcare': [0.25, 0.25, 0.25],
            'Financials': [0.25, 0.25, 0.25],
            'Consumer': [0.25, 0.25, 0.25]
        }, index=pd.date_range('2023-01-01', periods=3, freq='Q'))
        
        sector_returns = pd.DataFrame({
            'Technology': [0.05, 0.03, 0.08],
            'Healthcare': [0.02, 0.04, 0.01],
            'Financials': [0.03, 0.02, 0.05],
            'Consumer': [0.04, 0.01, 0.03]
        }, index=pd.date_range('2023-01-01', periods=3, freq='Q'))
        
        attribution = analyzer.calculate_attribution_analysis(
            portfolio_weights, benchmark_weights, sector_returns
        )
        
        assert 'allocation_effect' in attribution
        assert 'selection_effect' in attribution
        assert 'interaction_effect' in attribution
        assert 'total_active_return' in attribution
        
        # Validate attribution components sum to total
        total_calculated = (attribution['allocation_effect'] + 
                          attribution['selection_effect'] + 
                          attribution['interaction_effect'])
        assert abs(total_calculated - attribution['total_active_return']) < 1e-6
    
    def test_rolling_performance(self, analyzer):
        """Test rolling performance analysis"""
        window_days = 252  # 1 year
        rolling_metrics = analyzer.calculate_rolling_performance(window_days)
        
        assert 'rolling_returns' in rolling_metrics
        assert 'rolling_volatility' in rolling_metrics
        assert 'rolling_sharpe' in rolling_metrics
        assert 'rolling_alpha' in rolling_metrics
        assert 'rolling_beta' in rolling_metrics
        
        # Validate rolling metrics have correct length
        expected_length = len(analyzer.portfolio_returns) - window_days + 1
        assert len(rolling_metrics['rolling_returns']) == expected_length
    
    def test_risk_adjusted_metrics(self, analyzer):
        """Test risk-adjusted performance metrics"""
        risk_metrics = analyzer.calculate_risk_adjusted_metrics()
        
        assert 'sharpe_ratio' in risk_metrics
        assert 'sortino_ratio' in risk_metrics
        assert 'treynor_ratio' in risk_metrics
        assert 'jensen_alpha' in risk_metrics
        assert 'modigliani_ratio' in risk_metrics
        
        # Validate Treynor ratio calculation
        if risk_metrics['beta'] != 0:
            expected_treynor = (risk_metrics['excess_return'] / risk_metrics['beta'])
            assert abs(risk_metrics['treynor_ratio'] - expected_treynor) < 1e-6
    
    def test_benchmark_comparison(self, analyzer):
        """Test benchmark comparison analysis"""
        comparison = analyzer.compare_to_benchmark()
        
        assert 'outperformance_periods' in comparison
        assert 'underperformance_periods' in comparison
        assert 'hit_ratio' in comparison
        assert 'up_capture' in comparison
        assert 'down_capture' in comparison
        
        # Validate hit ratio is between 0 and 1
        assert 0 <= comparison['hit_ratio'] <= 1
        
        # Validate capture ratios are positive
        assert comparison['up_capture'] >= 0
        assert comparison['down_capture'] >= 0
    
    def test_performance_persistence(self, analyzer):
        """Test performance persistence analysis"""
        persistence = analyzer.analyze_performance_persistence()
        
        assert 'persistence_score' in persistence
        assert 'consistency_ratio' in persistence
        assert 'performance_periods' in persistence
        
        # Validate persistence score
        assert -1 <= persistence['persistence_score'] <= 1
        
        # Validate consistency ratio
        assert 0 <= persistence['consistency_ratio'] <= 1


class TestFactorModels:
    """Test cases for Factor Models"""
    
    @pytest.fixture
    def sample_factor_data(self):
        """Sample factor data for testing"""
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2023-12-31', freq='D')
        
        # Generate factor returns
        factor_returns = pd.DataFrame({
            'Market': np.random.normal(0.0008, 0.02, len(dates)),
            'SMB': np.random.normal(0.0002, 0.015, len(dates)),  # Size factor
            'HML': np.random.normal(0.0001, 0.018, len(dates)),  # Value factor
            'RMW': np.random.normal(0.0001, 0.012, len(dates)),  # Profitability
            'CMA': np.random.normal(0.0000, 0.010, len(dates))   # Investment
        }, index=dates)
        
        # Generate asset returns with factor exposures
        asset_returns = pd.DataFrame({
            'AAPL': factor_returns['Market'] * 1.2 + factor_returns['SMB'] * -0.3 + np.random.normal(0, 0.01, len(dates)),
            'MSFT': factor_returns['Market'] * 1.0 + factor_returns['SMB'] * -0.2 + np.random.normal(0, 0.01, len(dates)),
            'BRK.B': factor_returns['Market'] * 0.8 + factor_returns['HML'] * 0.4 + np.random.normal(0, 0.01, len(dates))
        }, index=dates)
        
        return factor_returns, asset_returns
    
    @pytest.fixture
    def factor_models(self, sample_factor_data):
        """Create factor models instance"""
        factor_returns, asset_returns = sample_factor_data
        return FactorModels(factor_returns, asset_returns)
    
    def test_capm_model(self, factor_models):
        """Test CAPM model estimation"""
        asset = 'AAPL'
        capm_results = factor_models.estimate_capm(asset)
        
        assert 'alpha' in capm_results
        assert 'beta' in capm_results
        assert 'r_squared' in capm_results
        assert 't_stats' in capm_results
        assert 'p_values' in capm_results
        
        # Validate statistical significance
        assert 'alpha' in capm_results['t_stats']
        assert 'beta' in capm_results['t_stats']
        assert 'alpha' in capm_results['p_values']
        assert 'beta' in capm_results['p_values']
        
        # Validate R-squared is between 0 and 1
        assert 0 <= capm_results['r_squared'] <= 1
    
    def test_fama_french_3_factor(self, factor_models):
        """Test Fama-French 3-factor model"""
        asset = 'AAPL'
        ff3_results = factor_models.estimate_fama_french_3_factor(asset)
        
        assert 'alpha' in ff3_results
        assert 'market_beta' in ff3_results
        assert 'size_beta' in ff3_results
        assert 'value_beta' in ff3_results
        assert 'r_squared' in ff3_results
        
        # Validate R-squared improvement over CAPM
        capm_results = factor_models.estimate_capm(asset)
        assert ff3_results['r_squared'] >= capm_results['r_squared']
    
    def test_fama_french_5_factor(self, factor_models):
        """Test Fama-French 5-factor model"""
        asset = 'AAPL'
        ff5_results = factor_models.estimate_fama_french_5_factor(asset)
        
        expected_factors = ['alpha', 'market_beta', 'size_beta', 'value_beta', 
                          'profitability_beta', 'investment_beta']
        
        for factor in expected_factors:
            assert factor in ff5_results
        
        assert 'r_squared' in ff5_results
        assert 't_stats' in ff5_results
        assert 'p_values' in ff5_results
    
    def test_custom_factor_model(self, factor_models):
        """Test custom factor model"""
        custom_factors = ['Market', 'SMB']  # Use subset of factors
        asset = 'MSFT'
        
        custom_results = factor_models.estimate_custom_factor_model(asset, custom_factors)
        
        assert 'alpha' in custom_results
        assert 'factor_loadings' in custom_results
        assert 'r_squared' in custom_results
        
        # Validate factor loadings
        for factor in custom_factors:
            assert factor in custom_results['factor_loadings']
    
    def test_factor_risk_attribution(self, factor_models):
        """Test factor risk attribution"""
        portfolio_weights = {'AAPL': 0.4, 'MSFT': 0.3, 'BRK.B': 0.3}
        
        risk_attribution = factor_models.calculate_factor_risk_attribution(portfolio_weights)
        
        assert 'factor_exposures' in risk_attribution
        assert 'factor_contributions' in risk_attribution
        assert 'idiosyncratic_risk' in risk_attribution
        assert 'total_risk' in risk_attribution
        
        # Validate risk decomposition
        factor_risk_squared = sum(contrib**2 for contrib in risk_attribution['factor_contributions'].values())
        idiosyncratic_risk_squared = risk_attribution['idiosyncratic_risk']**2
        total_risk_squared = risk_attribution['total_risk']**2
        
        assert abs(factor_risk_squared + idiosyncratic_risk_squared - total_risk_squared) < 1e-6
    
    def test_factor_timing_analysis(self, factor_models):
        """Test factor timing analysis"""
        asset = 'AAPL'
        timing_results = factor_models.analyze_factor_timing(asset)
        
        assert 'timing_coefficients' in timing_results
        assert 'timing_significance' in timing_results
        assert 'market_timing' in timing_results['timing_coefficients']
        
        # Validate timing significance
        for factor, p_value in timing_results['timing_significance'].items():
            assert 0 <= p_value <= 1
    
    def test_rolling_factor_loadings(self, factor_models):
        """Test rolling factor loadings"""
        asset = 'AAPL'
        window = 252  # 1 year
        
        rolling_loadings = factor_models.calculate_rolling_factor_loadings(asset, window)
        
        assert 'rolling_alpha' in rolling_loadings
        assert 'rolling_betas' in rolling_loadings
        assert 'rolling_r_squared' in rolling_loadings
        
        # Validate rolling data length
        expected_length = len(factor_models.asset_returns) - window + 1
        assert len(rolling_loadings['rolling_alpha']) == expected_length
    
    def test_factor_model_comparison(self, factor_models):
        """Test factor model comparison"""
        asset = 'AAPL'
        comparison = factor_models.compare_factor_models(asset)
        
        assert 'CAPM' in comparison
        assert 'FF3' in comparison
        assert 'FF5' in comparison
        
        # Each model should have R-squared and AIC
        for model_name, results in comparison.items():
            assert 'r_squared' in results
            assert 'aic' in results
            assert 'bic' in results
        
        # Validate R-squared ordering (more factors should explain more variance)
        assert comparison['FF3']['r_squared'] >= comparison['CAPM']['r_squared']
        assert comparison['FF5']['r_squared'] >= comparison['FF3']['r_squared']

