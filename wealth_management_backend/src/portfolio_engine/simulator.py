import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import concurrent.futures
from multiprocessing import cpu_count

logger = logging.getLogger(__name__)

class SimulationMethod(Enum):
    """Monte Carlo simulation methods"""
    GEOMETRIC_BROWNIAN = "geometric_brownian"
    HISTORICAL_BOOTSTRAP = "historical_bootstrap"
    FACTOR_MODEL = "factor_model"
    COPULA_BASED = "copula_based"
    REGIME_SWITCHING = "regime_switching"
    JUMP_DIFFUSION = "jump_diffusion"

class DistributionType(Enum):
    """Distribution types for simulation"""
    NORMAL = "normal"
    T_DISTRIBUTION = "t_distribution"
    SKEWED_T = "skewed_t"
    EMPIRICAL = "empirical"

@dataclass
class SimulationParameters:
    """Monte Carlo simulation parameters"""
    n_simulations: int = 10000
    time_horizon: int = 252  # Trading days (1 year)
    method: SimulationMethod = SimulationMethod.GEOMETRIC_BROWNIAN
    distribution: DistributionType = DistributionType.NORMAL
    confidence_levels: List[float] = None
    rebalancing_frequency: int = 0  # 0 = no rebalancing, 21 = monthly
    transaction_costs: float = 0.001  # 0.1% per transaction
    include_dividends: bool = True
    inflation_adjustment: bool = False
    inflation_rate: float = 0.025  # 2.5% annual inflation
    
    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]

@dataclass
class SimulationResult:
    """Monte Carlo simulation result"""
    final_values: np.ndarray
    paths: np.ndarray
    statistics: Dict[str, float]
    percentiles: Dict[float, float]
    probability_metrics: Dict[str, float]
    drawdown_analysis: Dict[str, float]
    time_to_target: Optional[Dict[str, float]] = None
    rebalancing_costs: float = 0.0
    success_probability: float = 0.0

class MonteCarloSimulator:
    """
    Advanced Monte Carlo Portfolio Simulator
    
    Features:
    - Multiple simulation methods (GBM, Bootstrap, Factor models)
    - Various probability distributions
    - Portfolio rebalancing simulation
    - Transaction cost modeling
    - Drawdown and risk analysis
    - Goal-based probability calculations
    - Parallel processing for performance
    """
    
    def __init__(self, random_seed: Optional[int] = None):
        self.random_seed = random_seed
        if random_seed:
            np.random.seed(random_seed)
        
        # Simulation cache for performance
        self._cache = {}
        
        # Parallel processing
        self.n_cores = min(cpu_count(), 8)  # Limit to 8 cores
    
    def simulate_portfolio(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        initial_value: float = 100000,
        parameters: Optional[SimulationParameters] = None,
        target_value: Optional[float] = None,
        benchmark_returns: Optional[pd.Series] = None,
        factor_returns: Optional[pd.DataFrame] = None,
        correlation_matrix: Optional[np.ndarray] = None
    ) -> SimulationResult:
        """
        Run Monte Carlo simulation for portfolio
        
        Args:
            returns: Historical asset returns
            weights: Portfolio weights
            initial_value: Initial portfolio value
            parameters: Simulation parameters
            target_value: Target value for probability calculations
            benchmark_returns: Benchmark for relative analysis
            factor_returns: Factor returns for factor model simulation
            correlation_matrix: Custom correlation matrix
            
        Returns:
            SimulationResult with comprehensive analysis
        """
        try:
            if parameters is None:
                parameters = SimulationParameters()
            
            # Prepare data
            returns_clean = returns.dropna()
            if returns_clean.empty:
                raise ValueError("No valid returns data")
            
            assets = returns_clean.columns.tolist()
            n_assets = len(assets)
            
            # Calculate statistics
            mean_returns = returns_clean.mean().values
            cov_matrix = returns_clean.cov().values
            
            if correlation_matrix is not None:
                # Use custom correlation with historical volatilities
                vol_vector = np.sqrt(np.diag(cov_matrix))
                cov_matrix = np.outer(vol_vector, vol_vector) * correlation_matrix
            
            # Run simulation based on method
            if parameters.method == SimulationMethod.GEOMETRIC_BROWNIAN:
                paths = self._simulate_geometric_brownian(
                    mean_returns, cov_matrix, weights, initial_value, parameters
                )
            elif parameters.method == SimulationMethod.HISTORICAL_BOOTSTRAP:
                paths = self._simulate_historical_bootstrap(
                    returns_clean, weights, initial_value, parameters
                )
            elif parameters.method == SimulationMethod.FACTOR_MODEL:
                if factor_returns is not None:
                    paths = self._simulate_factor_model(
                        returns_clean, factor_returns, weights, initial_value, parameters
                    )
                else:
                    # Fallback to GBM
                    paths = self._simulate_geometric_brownian(
                        mean_returns, cov_matrix, weights, initial_value, parameters
                    )
            else:
                # Default to Geometric Brownian Motion
                paths = self._simulate_geometric_brownian(
                    mean_returns, cov_matrix, weights, initial_value, parameters
                )
            
            # Calculate final values
            final_values = paths[:, -1]
            
            # Calculate statistics
            statistics = self._calculate_simulation_statistics(final_values, initial_value)
            
            # Calculate percentiles
            percentiles = {}
            for conf_level in parameters.confidence_levels:
                percentiles[conf_level] = np.percentile(final_values, conf_level * 100)
            
            # Probability metrics
            probability_metrics = self._calculate_probability_metrics(
                final_values, initial_value, target_value
            )
            
            # Drawdown analysis
            drawdown_analysis = self._analyze_drawdowns(paths)
            
            # Time to target analysis
            time_to_target = None
            if target_value:
                time_to_target = self._analyze_time_to_target(paths, target_value)
            
            # Calculate rebalancing costs
            rebalancing_costs = self._calculate_rebalancing_costs(
                paths, weights, parameters
            )
            
            # Success probability
            success_probability = probability_metrics.get('probability_positive', 0.0)
            if target_value:
                success_probability = probability_metrics.get('probability_target', 0.0)
            
            return SimulationResult(
                final_values=final_values,
                paths=paths,
                statistics=statistics,
                percentiles=percentiles,
                probability_metrics=probability_metrics,
                drawdown_analysis=drawdown_analysis,
                time_to_target=time_to_target,
                rebalancing_costs=rebalancing_costs,
                success_probability=success_probability
            )
            
        except Exception as e:
            logger.error(f"Monte Carlo simulation error: {str(e)}")
            # Return empty result
            return SimulationResult(
                final_values=np.array([initial_value]),
                paths=np.array([[initial_value]]),
                statistics={},
                percentiles={},
                probability_metrics={},
                drawdown_analysis={},
                rebalancing_costs=0.0,
                success_probability=0.0
            )
    
    def _simulate_geometric_brownian(
        self,
        mean_returns: np.ndarray,
        cov_matrix: np.ndarray,
        weights: np.ndarray,
        initial_value: float,
        parameters: SimulationParameters
    ) -> np.ndarray:
        """Simulate using Geometric Brownian Motion"""
        
        n_sims = parameters.n_simulations
        n_days = parameters.time_horizon
        dt = 1/252  # Daily time step
        
        # Portfolio expected return and volatility
        portfolio_return = np.dot(weights, mean_returns)
        portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
        portfolio_vol = np.sqrt(portfolio_variance)
        
        # Adjust for distribution type
        if parameters.distribution == DistributionType.NORMAL:
            random_shocks = np.random.normal(0, 1, (n_sims, n_days))
        elif parameters.distribution == DistributionType.T_DISTRIBUTION:
            # Use t-distribution with 5 degrees of freedom
            random_shocks = np.random.standard_t(5, (n_sims, n_days)) / np.sqrt(5/3)  # Normalize variance
        else:
            # Default to normal
            random_shocks = np.random.normal(0, 1, (n_sims, n_days))
        
        # Generate paths using vectorized operations
        drift = (portfolio_return - 0.5 * portfolio_variance) * dt
        diffusion = portfolio_vol * np.sqrt(dt) * random_shocks
        
        # Calculate log returns
        log_returns = drift + diffusion
        
        # Convert to price paths
        log_prices = np.cumsum(log_returns, axis=1)
        paths = initial_value * np.exp(log_prices)
        
        # Add initial value as first column
        paths = np.column_stack([np.full(n_sims, initial_value), paths])
        
        return paths
    
    def _simulate_historical_bootstrap(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        initial_value: float,
        parameters: SimulationParameters
    ) -> np.ndarray:
        """Simulate using historical bootstrap method"""
        
        n_sims = parameters.n_simulations
        n_days = parameters.time_horizon
        
        # Calculate portfolio returns
        portfolio_returns = (returns * weights).sum(axis=1).values
        n_historical = len(portfolio_returns)
        
        # Bootstrap sampling
        paths = np.zeros((n_sims, n_days + 1))
        paths[:, 0] = initial_value
        
        for sim in range(n_sims):
            # Sample returns with replacement
            sampled_indices = np.random.choice(n_historical, n_days, replace=True)
            sampled_returns = portfolio_returns[sampled_indices]
            
            # Calculate cumulative path
            for day in range(n_days):
                paths[sim, day + 1] = paths[sim, day] * (1 + sampled_returns[day])
        
        return paths
    
    def _simulate_factor_model(
        self,
        returns: pd.DataFrame,
        factor_returns: pd.DataFrame,
        weights: np.ndarray,
        initial_value: float,
        parameters: SimulationParameters
    ) -> np.ndarray:
        """Simulate using factor model approach"""
        
        try:
            # Align data
            common_dates = returns.index.intersection(factor_returns.index)
            returns_aligned = returns.loc[common_dates]
            factors_aligned = factor_returns.loc[common_dates]
            
            # Calculate portfolio returns
            portfolio_returns = (returns_aligned * weights).sum(axis=1)
            
            # Fit factor model: portfolio_returns = alpha + beta * factors + error
            X = np.column_stack([np.ones(len(factors_aligned)), factors_aligned.values])
            y = portfolio_returns.values
            
            # OLS regression
            coefficients = np.linalg.lstsq(X, y, rcond=None)[0]
            alpha = coefficients[0]
            betas = coefficients[1:]
            
            # Calculate residuals
            y_pred = X @ coefficients
            residuals = y - y_pred
            residual_std = np.std(residuals)
            
            # Simulate factor returns and residuals
            n_sims = parameters.n_simulations
            n_days = parameters.time_horizon
            n_factors = len(betas)
            
            # Factor return simulation (simplified - using historical bootstrap)
            factor_data = factors_aligned.values
            n_historical = len(factor_data)
            
            paths = np.zeros((n_sims, n_days + 1))
            paths[:, 0] = initial_value
            
            for sim in range(n_sims):
                for day in range(n_days):
                    # Sample factor returns
                    factor_idx = np.random.choice(n_historical)
                    factor_returns_day = factor_data[factor_idx]
                    
                    # Sample residual
                    residual = np.random.normal(0, residual_std)
                    
                    # Calculate portfolio return
                    portfolio_return = alpha + np.dot(betas, factor_returns_day) + residual
                    
                    # Update path
                    paths[sim, day + 1] = paths[sim, day] * (1 + portfolio_return)
            
            return paths
            
        except Exception as e:
            logger.warning(f"Factor model simulation failed: {str(e)}, falling back to GBM")
            # Fallback to geometric Brownian motion
            mean_returns = returns.mean().values
            cov_matrix = returns.cov().values
            return self._simulate_geometric_brownian(
                mean_returns, cov_matrix, weights, initial_value, parameters
            )
    
    def _calculate_simulation_statistics(
        self,
        final_values: np.ndarray,
        initial_value: float
    ) -> Dict[str, float]:
        """Calculate comprehensive simulation statistics"""
        
        returns = (final_values - initial_value) / initial_value
        
        return {
            'mean_final_value': np.mean(final_values),
            'median_final_value': np.median(final_values),
            'std_final_value': np.std(final_values),
            'min_final_value': np.min(final_values),
            'max_final_value': np.max(final_values),
            'mean_return': np.mean(returns),
            'median_return': np.median(returns),
            'std_return': np.std(returns),
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns),
            'var_95': np.percentile(returns, 5),
            'var_99': np.percentile(returns, 1),
            'cvar_95': np.mean(returns[returns <= np.percentile(returns, 5)]),
            'cvar_99': np.mean(returns[returns <= np.percentile(returns, 1)])
        }
    
    def _calculate_probability_metrics(
        self,
        final_values: np.ndarray,
        initial_value: float,
        target_value: Optional[float] = None
    ) -> Dict[str, float]:
        """Calculate probability-based metrics"""
        
        metrics = {}
        
        # Probability of positive return
        positive_returns = final_values > initial_value
        metrics['probability_positive'] = np.mean(positive_returns)
        
        # Probability of loss
        metrics['probability_loss'] = 1 - metrics['probability_positive']
        
        # Probability of significant loss (>20%)
        significant_loss = final_values < initial_value * 0.8
        metrics['probability_significant_loss'] = np.mean(significant_loss)
        
        # Probability of doubling
        doubling = final_values >= initial_value * 2
        metrics['probability_doubling'] = np.mean(doubling)
        
        # Target-based probabilities
        if target_value:
            target_achieved = final_values >= target_value
            metrics['probability_target'] = np.mean(target_achieved)
            
            # Expected shortfall from target
            shortfall = np.maximum(target_value - final_values, 0)
            metrics['expected_shortfall_from_target'] = np.mean(shortfall)
            
            # Probability of being within 10% of target
            near_target = np.abs(final_values - target_value) <= target_value * 0.1
            metrics['probability_near_target'] = np.mean(near_target)
        
        return metrics
    
    def _analyze_drawdowns(self, paths: np.ndarray) -> Dict[str, float]:
        """Analyze drawdown characteristics across all paths"""
        
        # Calculate drawdowns for each path
        max_drawdowns = []
        avg_drawdowns = []
        drawdown_durations = []
        
        for path in paths:
            # Calculate running maximum
            running_max = np.maximum.accumulate(path)
            
            # Calculate drawdown
            drawdown = (path - running_max) / running_max
            
            # Maximum drawdown
            max_dd = np.min(drawdown)
            max_drawdowns.append(abs(max_dd))
            
            # Average drawdown
            negative_dd = drawdown[drawdown < 0]
            avg_dd = np.mean(negative_dd) if len(negative_dd) > 0 else 0
            avg_drawdowns.append(abs(avg_dd))
            
            # Drawdown duration (simplified)
            in_drawdown = drawdown < -0.05  # 5% drawdown threshold
            if np.any(in_drawdown):
                # Find longest consecutive drawdown period
                drawdown_periods = []
                current_period = 0
                for is_dd in in_drawdown:
                    if is_dd:
                        current_period += 1
                    else:
                        if current_period > 0:
                            drawdown_periods.append(current_period)
                        current_period = 0
                if current_period > 0:
                    drawdown_periods.append(current_period)
                
                max_duration = max(drawdown_periods) if drawdown_periods else 0
                drawdown_durations.append(max_duration)
            else:
                drawdown_durations.append(0)
        
        return {
            'mean_max_drawdown': np.mean(max_drawdowns),
            'median_max_drawdown': np.median(max_drawdowns),
            'worst_drawdown': np.max(max_drawdowns),
            'probability_large_drawdown': np.mean(np.array(max_drawdowns) > 0.20),  # >20% drawdown
            'mean_drawdown_duration': np.mean(drawdown_durations),
            'max_drawdown_duration': np.max(drawdown_durations) if drawdown_durations else 0
        }
    
    def _analyze_time_to_target(
        self,
        paths: np.ndarray,
        target_value: float
    ) -> Dict[str, float]:
        """Analyze time to reach target value"""
        
        times_to_target = []
        
        for path in paths:
            # Find first time target is reached
            target_reached = np.where(path >= target_value)[0]
            if len(target_reached) > 0:
                times_to_target.append(target_reached[0])
            else:
                times_to_target.append(len(path))  # Never reached
        
        times_to_target = np.array(times_to_target)
        
        # Convert to years (assuming daily steps)
        times_to_target_years = times_to_target / 252
        
        # Only consider paths that actually reached the target
        successful_times = times_to_target_years[times_to_target < len(paths[0])]
        
        result = {
            'probability_reach_target': len(successful_times) / len(times_to_target),
            'mean_time_to_target': np.mean(successful_times) if len(successful_times) > 0 else np.inf,
            'median_time_to_target': np.median(successful_times) if len(successful_times) > 0 else np.inf,
            'min_time_to_target': np.min(successful_times) if len(successful_times) > 0 else np.inf,
            'max_time_to_target': np.max(successful_times) if len(successful_times) > 0 else np.inf
        }
        
        return result
    
    def _calculate_rebalancing_costs(
        self,
        paths: np.ndarray,
        weights: np.ndarray,
        parameters: SimulationParameters
    ) -> float:
        """Calculate estimated rebalancing costs"""
        
        if parameters.rebalancing_frequency <= 0:
            return 0.0
        
        # Simplified rebalancing cost calculation
        n_rebalances = parameters.time_horizon // parameters.rebalancing_frequency
        
        # Assume average turnover of 20% per rebalancing
        avg_turnover = 0.20
        
        # Total rebalancing cost
        total_cost = n_rebalances * avg_turnover * parameters.transaction_costs
        
        return total_cost
    
    def run_scenario_analysis(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        scenarios: Dict[str, Dict[str, Any]],
        initial_value: float = 100000,
        parameters: Optional[SimulationParameters] = None
    ) -> Dict[str, SimulationResult]:
        """
        Run scenario analysis with different market conditions
        
        Args:
            returns: Historical asset returns
            weights: Portfolio weights
            scenarios: Dictionary of scenarios with parameters
            initial_value: Initial portfolio value
            parameters: Base simulation parameters
            
        Returns:
            Dictionary of scenario results
        """
        try:
            if parameters is None:
                parameters = SimulationParameters()
            
            scenario_results = {}
            
            for scenario_name, scenario_params in scenarios.items():
                # Modify parameters for scenario
                scenario_parameters = SimulationParameters(
                    n_simulations=scenario_params.get('n_simulations', parameters.n_simulations),
                    time_horizon=scenario_params.get('time_horizon', parameters.time_horizon),
                    method=scenario_params.get('method', parameters.method),
                    distribution=scenario_params.get('distribution', parameters.distribution)
                )
                
                # Apply scenario-specific modifications to returns
                modified_returns = returns.copy()
                
                if 'return_adjustment' in scenario_params:
                    # Adjust expected returns
                    adjustment = scenario_params['return_adjustment']
                    if isinstance(adjustment, dict):
                        # Asset-specific adjustments
                        for asset, adj in adjustment.items():
                            if asset in modified_returns.columns:
                                modified_returns[asset] = modified_returns[asset] + adj/252  # Daily adjustment
                    else:
                        # Universal adjustment
                        modified_returns = modified_returns + adjustment/252
                
                if 'volatility_multiplier' in scenario_params:
                    # Adjust volatility
                    multiplier = scenario_params['volatility_multiplier']
                    means = modified_returns.mean()
                    modified_returns = (modified_returns - means) * multiplier + means
                
                # Run simulation for scenario
                result = self.simulate_portfolio(
                    modified_returns,
                    weights,
                    initial_value,
                    scenario_parameters
                )
                
                scenario_results[scenario_name] = result
            
            return scenario_results
            
        except Exception as e:
            logger.error(f"Scenario analysis error: {str(e)}")
            return {}
    
    def optimize_simulation_parameters(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        target_accuracy: float = 0.01,
        max_simulations: int = 50000
    ) -> SimulationParameters:
        """
        Optimize simulation parameters for desired accuracy
        
        Args:
            returns: Historical asset returns
            weights: Portfolio weights
            target_accuracy: Target accuracy for mean estimate
            max_simulations: Maximum number of simulations
            
        Returns:
            Optimized SimulationParameters
        """
        try:
            # Start with small number of simulations
            n_sims = 1000
            
            while n_sims <= max_simulations:
                # Run simulation
                params = SimulationParameters(n_simulations=n_sims)
                result = self.simulate_portfolio(returns, weights, 100000, params)
                
                # Calculate standard error of mean
                std_error = result.statistics['std_final_value'] / np.sqrt(n_sims)
                relative_error = std_error / result.statistics['mean_final_value']
                
                if relative_error <= target_accuracy:
                    break
                
                # Increase simulations
                n_sims = min(int(n_sims * 1.5), max_simulations)
            
            return SimulationParameters(n_simulations=n_sims)
            
        except Exception as e:
            logger.error(f"Parameter optimization error: {str(e)}")
            return SimulationParameters()
    
    def generate_confidence_intervals(
        self,
        simulation_result: SimulationResult,
        confidence_level: float = 0.95
    ) -> Dict[str, Tuple[float, float]]:
        """Generate confidence intervals for simulation results"""
        
        try:
            alpha = 1 - confidence_level
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100
            
            intervals = {}
            
            # Final value confidence interval
            final_values = simulation_result.final_values
            intervals['final_value'] = (
                np.percentile(final_values, lower_percentile),
                np.percentile(final_values, upper_percentile)
            )
            
            # Return confidence interval
            initial_value = simulation_result.paths[0, 0]  # Assuming first value is initial
            returns = (final_values - initial_value) / initial_value
            intervals['return'] = (
                np.percentile(returns, lower_percentile),
                np.percentile(returns, upper_percentile)
            )
            
            return intervals
            
        except Exception as e:
            logger.error(f"Confidence interval calculation error: {str(e)}")
            return {}

