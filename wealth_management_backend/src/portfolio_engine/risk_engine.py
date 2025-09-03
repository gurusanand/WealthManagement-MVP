import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class RiskMeasure(Enum):
    """Risk measurement types"""
    VALUE_AT_RISK = "var"
    CONDITIONAL_VAR = "cvar"
    EXPECTED_SHORTFALL = "expected_shortfall"
    MAXIMUM_DRAWDOWN = "max_drawdown"
    VOLATILITY = "volatility"
    DOWNSIDE_DEVIATION = "downside_deviation"
    BETA = "beta"
    TRACKING_ERROR = "tracking_error"
    INFORMATION_RATIO = "information_ratio"

class StressTestScenario(Enum):
    """Predefined stress test scenarios"""
    MARKET_CRASH = "market_crash"
    INTEREST_RATE_SHOCK = "interest_rate_shock"
    CREDIT_CRISIS = "credit_crisis"
    INFLATION_SPIKE = "inflation_spike"
    CURRENCY_CRISIS = "currency_crisis"
    LIQUIDITY_CRISIS = "liquidity_crisis"
    GEOPOLITICAL_SHOCK = "geopolitical_shock"
    PANDEMIC_SCENARIO = "pandemic_scenario"
    TECH_BUBBLE_BURST = "tech_bubble_burst"
    ENERGY_CRISIS = "energy_crisis"

@dataclass
class RiskMetrics:
    """Comprehensive risk metrics"""
    var_95: float
    var_99: float
    cvar_95: float
    cvar_99: float
    expected_shortfall: float
    volatility: float
    downside_deviation: float
    maximum_drawdown: float
    beta: Optional[float] = None
    tracking_error: Optional[float] = None
    information_ratio: Optional[float] = None
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    risk_adjusted_return: float = 0.0

@dataclass
class StressTestResult:
    """Stress test scenario result"""
    scenario_name: str
    portfolio_return: float
    portfolio_loss: float
    worst_asset: str
    worst_asset_loss: float
    best_asset: str
    best_asset_gain: float
    sector_impacts: Dict[str, float]
    correlation_breakdown: bool
    liquidity_impact: float
    recovery_time_estimate: int  # days

@dataclass
class FactorExposure:
    """Factor model exposure"""
    factor_name: str
    exposure: float
    contribution_to_risk: float
    t_statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]

class RiskEngine:
    """
    Comprehensive Risk Analytics Engine
    
    Features:
    - Value at Risk (VaR) and Conditional VaR
    - Stress testing and scenario analysis
    - Factor model risk attribution
    - Drawdown and tail risk analysis
    - Correlation and dependency analysis
    - Risk budgeting and decomposition
    - Monte Carlo risk simulation
    """
    
    def __init__(self, confidence_levels: List[float] = [0.95, 0.99]):
        self.confidence_levels = confidence_levels
        self.risk_free_rate = 0.02  # 2% annual risk-free rate
        
        # Risk model parameters
        self.var_methods = ['historical', 'parametric', 'monte_carlo']
        self.stress_scenarios = self._initialize_stress_scenarios()
        self.factor_models = ['fama_french_3', 'fama_french_5', 'custom']
        
        # Calculation cache
        self._cache = {}
    
    def calculate_portfolio_risk(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        benchmark_returns: Optional[pd.Series] = None,
        factor_returns: Optional[pd.DataFrame] = None,
        sector_mapping: Optional[Dict[str, str]] = None,
        market_cap_data: Optional[Dict[str, float]] = None
    ) -> RiskMetrics:
        """
        Calculate comprehensive risk metrics for a portfolio
        
        Args:
            returns: Asset returns DataFrame
            weights: Portfolio weights
            benchmark_returns: Benchmark returns for relative metrics
            factor_returns: Factor returns for factor model analysis
            sector_mapping: Asset to sector mapping
            market_cap_data: Market capitalization data
            
        Returns:
            RiskMetrics with comprehensive risk analysis
        """
        try:
            # Calculate portfolio returns
            portfolio_returns = (returns * weights).sum(axis=1)
            
            # Basic risk metrics
            volatility = portfolio_returns.std() * np.sqrt(252)  # Annualized
            mean_return = portfolio_returns.mean() * 252  # Annualized
            
            # VaR calculations
            var_95 = self._calculate_var(portfolio_returns, 0.95, method='historical')
            var_99 = self._calculate_var(portfolio_returns, 0.99, method='historical')
            
            # Conditional VaR (Expected Shortfall)
            cvar_95 = self._calculate_cvar(portfolio_returns, 0.95)
            cvar_99 = self._calculate_cvar(portfolio_returns, 0.99)
            expected_shortfall = cvar_95
            
            # Downside risk metrics
            downside_deviation = self._calculate_downside_deviation(portfolio_returns)
            maximum_drawdown = self._calculate_maximum_drawdown(portfolio_returns)
            
            # Risk-adjusted returns
            sharpe_ratio = (mean_return - self.risk_free_rate) / volatility if volatility > 0 else 0
            sortino_ratio = (mean_return - self.risk_free_rate) / downside_deviation if downside_deviation > 0 else 0
            calmar_ratio = mean_return / abs(maximum_drawdown) if maximum_drawdown != 0 else 0
            
            # Benchmark-relative metrics
            beta = None
            tracking_error = None
            information_ratio = None
            
            if benchmark_returns is not None:
                beta = self._calculate_beta(portfolio_returns, benchmark_returns)
                tracking_error = self._calculate_tracking_error(portfolio_returns, benchmark_returns)
                information_ratio = self._calculate_information_ratio(portfolio_returns, benchmark_returns)
            
            return RiskMetrics(
                var_95=var_95,
                var_99=var_99,
                cvar_95=cvar_95,
                cvar_99=cvar_99,
                expected_shortfall=expected_shortfall,
                volatility=volatility,
                downside_deviation=downside_deviation,
                maximum_drawdown=maximum_drawdown,
                beta=beta,
                tracking_error=tracking_error,
                information_ratio=information_ratio,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                risk_adjusted_return=sharpe_ratio
            )
            
        except Exception as e:
            logger.error(f"Portfolio risk calculation error: {str(e)}")
            return RiskMetrics(
                var_95=0.0, var_99=0.0, cvar_95=0.0, cvar_99=0.0,
                expected_shortfall=0.0, volatility=0.0, downside_deviation=0.0,
                maximum_drawdown=0.0
            )
    
    def run_stress_tests(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        scenarios: Optional[List[StressTestScenario]] = None,
        custom_scenarios: Optional[Dict[str, Dict[str, float]]] = None,
        sector_mapping: Optional[Dict[str, str]] = None
    ) -> List[StressTestResult]:
        """
        Run comprehensive stress tests on the portfolio
        
        Args:
            returns: Asset returns DataFrame
            weights: Portfolio weights
            scenarios: List of predefined scenarios to test
            custom_scenarios: Custom stress scenarios
            sector_mapping: Asset to sector mapping
            
        Returns:
            List of StressTestResult objects
        """
        try:
            if scenarios is None:
                scenarios = [
                    StressTestScenario.MARKET_CRASH,
                    StressTestScenario.INTEREST_RATE_SHOCK,
                    StressTestScenario.CREDIT_CRISIS,
                    StressTestScenario.INFLATION_SPIKE
                ]
            
            stress_results = []
            assets = returns.columns.tolist()
            
            # Run predefined scenarios
            for scenario in scenarios:
                scenario_shocks = self.stress_scenarios.get(scenario.value, {})
                result = self._apply_stress_scenario(
                    returns, weights, assets, scenario.value, scenario_shocks, sector_mapping
                )
                stress_results.append(result)
            
            # Run custom scenarios
            if custom_scenarios:
                for scenario_name, shocks in custom_scenarios.items():
                    result = self._apply_stress_scenario(
                        returns, weights, assets, scenario_name, shocks, sector_mapping
                    )
                    stress_results.append(result)
            
            return stress_results
            
        except Exception as e:
            logger.error(f"Stress testing error: {str(e)}")
            return []
    
    def calculate_factor_exposures(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        factor_returns: pd.DataFrame,
        model_type: str = 'fama_french_3'
    ) -> List[FactorExposure]:
        """
        Calculate factor model exposures and risk attribution
        
        Args:
            returns: Asset returns DataFrame
            weights: Portfolio weights
            factor_returns: Factor returns DataFrame
            model_type: Type of factor model to use
            
        Returns:
            List of FactorExposure objects
        """
        try:
            # Calculate portfolio returns
            portfolio_returns = (returns * weights).sum(axis=1)
            
            # Align dates
            common_dates = portfolio_returns.index.intersection(factor_returns.index)
            portfolio_returns = portfolio_returns.loc[common_dates]
            factor_returns = factor_returns.loc[common_dates]
            
            factor_exposures = []
            
            for factor_name in factor_returns.columns:
                factor_data = factor_returns[factor_name]
                
                # Run regression: portfolio_returns = alpha + beta * factor + error
                X = np.column_stack([np.ones(len(factor_data)), factor_data.values])
                y = portfolio_returns.values
                
                # OLS regression
                try:
                    beta_coef = np.linalg.lstsq(X, y, rcond=None)[0]
                    alpha, beta = beta_coef[0], beta_coef[1]
                    
                    # Calculate statistics
                    y_pred = X @ beta_coef
                    residuals = y - y_pred
                    mse = np.mean(residuals ** 2)
                    
                    # Standard error of beta
                    X_var = np.linalg.inv(X.T @ X)
                    se_beta = np.sqrt(mse * X_var[1, 1])
                    
                    # t-statistic and p-value
                    t_stat = beta / se_beta if se_beta > 0 else 0
                    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), len(y) - 2))
                    
                    # Confidence interval (95%)
                    t_critical = stats.t.ppf(0.975, len(y) - 2)
                    ci_lower = beta - t_critical * se_beta
                    ci_upper = beta + t_critical * se_beta
                    
                    # Risk contribution (simplified)
                    factor_var = factor_data.var()
                    contribution_to_risk = (beta ** 2) * factor_var
                    
                    factor_exposures.append(FactorExposure(
                        factor_name=factor_name,
                        exposure=beta,
                        contribution_to_risk=contribution_to_risk,
                        t_statistic=t_stat,
                        p_value=p_value,
                        confidence_interval=(ci_lower, ci_upper)
                    ))
                    
                except np.linalg.LinAlgError:
                    logger.warning(f"Could not calculate exposure for factor {factor_name}")
                    continue
            
            return factor_exposures
            
        except Exception as e:
            logger.error(f"Factor exposure calculation error: {str(e)}")
            return []
    
    def calculate_risk_decomposition(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        sector_mapping: Optional[Dict[str, str]] = None
    ) -> Dict[str, Any]:
        """
        Decompose portfolio risk by assets and sectors
        
        Args:
            returns: Asset returns DataFrame
            weights: Portfolio weights
            sector_mapping: Asset to sector mapping
            
        Returns:
            Risk decomposition analysis
        """
        try:
            assets = returns.columns.tolist()
            cov_matrix = returns.cov().values * 252  # Annualized
            
            # Portfolio variance
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            portfolio_volatility = np.sqrt(portfolio_variance)
            
            # Marginal risk contributions
            marginal_risk = np.dot(cov_matrix, weights) / portfolio_volatility
            
            # Component risk contributions
            component_risk = weights * marginal_risk
            
            # Asset-level risk decomposition
            asset_risk_contrib = {}
            for i, asset in enumerate(assets):
                asset_risk_contrib[asset] = {
                    'weight': weights[i],
                    'marginal_risk': marginal_risk[i],
                    'risk_contribution': component_risk[i],
                    'risk_contribution_pct': component_risk[i] / portfolio_variance * 100
                }
            
            # Sector-level risk decomposition
            sector_risk_contrib = {}
            if sector_mapping:
                sectors = set(sector_mapping.values())
                for sector in sectors:
                    sector_indices = [i for i, asset in enumerate(assets) if sector_mapping.get(asset) == sector]
                    sector_weight = sum(weights[i] for i in sector_indices)
                    sector_risk = sum(component_risk[i] for i in sector_indices)
                    
                    sector_risk_contrib[sector] = {
                        'weight': sector_weight,
                        'risk_contribution': sector_risk,
                        'risk_contribution_pct': sector_risk / portfolio_variance * 100
                    }
            
            return {
                'portfolio_volatility': portfolio_volatility,
                'portfolio_variance': portfolio_variance,
                'asset_contributions': asset_risk_contrib,
                'sector_contributions': sector_risk_contrib,
                'diversification_ratio': self._calculate_diversification_ratio(weights, returns)
            }
            
        except Exception as e:
            logger.error(f"Risk decomposition error: {str(e)}")
            return {}
    
    def calculate_correlation_analysis(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        rolling_window: int = 252
    ) -> Dict[str, Any]:
        """
        Analyze correlation structure and stability
        
        Args:
            returns: Asset returns DataFrame
            weights: Portfolio weights
            rolling_window: Window for rolling correlation analysis
            
        Returns:
            Correlation analysis results
        """
        try:
            # Static correlation matrix
            correlation_matrix = returns.corr()
            
            # Average correlation
            n_assets = len(returns.columns)
            avg_correlation = (correlation_matrix.sum().sum() - n_assets) / (n_assets * (n_assets - 1))
            
            # Rolling correlation analysis
            rolling_corr = {}
            assets = returns.columns.tolist()
            
            for i in range(len(assets)):
                for j in range(i + 1, len(assets)):
                    asset1, asset2 = assets[i], assets[j]
                    rolling_corr[f"{asset1}_{asset2}"] = returns[asset1].rolling(rolling_window).corr(returns[asset2])
            
            # Correlation stability metrics
            correlation_stability = {}
            for pair, corr_series in rolling_corr.items():
                corr_clean = corr_series.dropna()
                if len(corr_clean) > 0:
                    correlation_stability[pair] = {
                        'mean': corr_clean.mean(),
                        'std': corr_clean.std(),
                        'min': corr_clean.min(),
                        'max': corr_clean.max(),
                        'current': corr_clean.iloc[-1] if len(corr_clean) > 0 else 0
                    }
            
            # Portfolio correlation metrics
            weighted_avg_correlation = 0
            total_weight_pairs = 0
            
            for i in range(n_assets):
                for j in range(i + 1, n_assets):
                    weight_product = weights[i] * weights[j]
                    correlation = correlation_matrix.iloc[i, j]
                    weighted_avg_correlation += weight_product * correlation
                    total_weight_pairs += weight_product
            
            if total_weight_pairs > 0:
                weighted_avg_correlation /= total_weight_pairs
            
            return {
                'correlation_matrix': correlation_matrix.to_dict(),
                'average_correlation': avg_correlation,
                'weighted_average_correlation': weighted_avg_correlation,
                'correlation_stability': correlation_stability,
                'rolling_correlations': {k: v.to_dict() for k, v in rolling_corr.items()}
            }
            
        except Exception as e:
            logger.error(f"Correlation analysis error: {str(e)}")
            return {}
    
    def _calculate_var(self, returns: pd.Series, confidence_level: float, method: str = 'historical') -> float:
        """Calculate Value at Risk"""
        try:
            if method == 'historical':
                return np.percentile(returns, (1 - confidence_level) * 100)
            elif method == 'parametric':
                mean = returns.mean()
                std = returns.std()
                z_score = stats.norm.ppf(1 - confidence_level)
                return mean + z_score * std
            elif method == 'monte_carlo':
                # Simplified Monte Carlo
                simulated_returns = np.random.normal(returns.mean(), returns.std(), 10000)
                return np.percentile(simulated_returns, (1 - confidence_level) * 100)
            else:
                return np.percentile(returns, (1 - confidence_level) * 100)
        except Exception:
            return 0.0
    
    def _calculate_cvar(self, returns: pd.Series, confidence_level: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)"""
        try:
            var_threshold = self._calculate_var(returns, confidence_level)
            tail_losses = returns[returns <= var_threshold]
            return tail_losses.mean() if len(tail_losses) > 0 else 0.0
        except Exception:
            return 0.0
    
    def _calculate_downside_deviation(self, returns: pd.Series, target_return: float = 0.0) -> float:
        """Calculate downside deviation"""
        try:
            downside_returns = returns[returns < target_return]
            if len(downside_returns) == 0:
                return 0.0
            return np.sqrt(((downside_returns - target_return) ** 2).mean()) * np.sqrt(252)
        except Exception:
            return 0.0
    
    def _calculate_maximum_drawdown(self, returns: pd.Series) -> float:
        """Calculate maximum drawdown"""
        try:
            cumulative_returns = (1 + returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            return drawdown.min()
        except Exception:
            return 0.0
    
    def _calculate_beta(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate portfolio beta"""
        try:
            # Align series
            common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
            port_ret = portfolio_returns.loc[common_dates]
            bench_ret = benchmark_returns.loc[common_dates]
            
            covariance = np.cov(port_ret, bench_ret)[0, 1]
            benchmark_variance = np.var(bench_ret)
            
            return covariance / benchmark_variance if benchmark_variance > 0 else 0.0
        except Exception:
            return 0.0
    
    def _calculate_tracking_error(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate tracking error"""
        try:
            # Align series
            common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
            port_ret = portfolio_returns.loc[common_dates]
            bench_ret = benchmark_returns.loc[common_dates]
            
            active_returns = port_ret - bench_ret
            return active_returns.std() * np.sqrt(252)
        except Exception:
            return 0.0
    
    def _calculate_information_ratio(self, portfolio_returns: pd.Series, benchmark_returns: pd.Series) -> float:
        """Calculate information ratio"""
        try:
            # Align series
            common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
            port_ret = portfolio_returns.loc[common_dates]
            bench_ret = benchmark_returns.loc[common_dates]
            
            active_returns = port_ret - bench_ret
            active_return = active_returns.mean() * 252
            tracking_error = active_returns.std() * np.sqrt(252)
            
            return active_return / tracking_error if tracking_error > 0 else 0.0
        except Exception:
            return 0.0
    
    def _calculate_diversification_ratio(self, weights: np.ndarray, returns: pd.DataFrame) -> float:
        """Calculate diversification ratio"""
        try:
            # Weighted average volatility
            individual_vols = returns.std().values * np.sqrt(252)
            weighted_avg_vol = np.dot(weights, individual_vols)
            
            # Portfolio volatility
            portfolio_vol = np.sqrt(np.dot(weights, np.dot(returns.cov().values * 252, weights)))
            
            return weighted_avg_vol / portfolio_vol if portfolio_vol > 0 else 1.0
        except Exception:
            return 1.0
    
    def _apply_stress_scenario(
        self,
        returns: pd.DataFrame,
        weights: np.ndarray,
        assets: List[str],
        scenario_name: str,
        shocks: Dict[str, float],
        sector_mapping: Optional[Dict[str, str]] = None
    ) -> StressTestResult:
        """Apply a stress scenario to the portfolio"""
        try:
            # Apply shocks to assets
            shocked_returns = np.zeros(len(assets))
            
            for i, asset in enumerate(assets):
                if asset in shocks:
                    shocked_returns[i] = shocks[asset]
                else:
                    # Apply sector-level shock if available
                    sector = sector_mapping.get(asset) if sector_mapping else None
                    if sector and sector in shocks:
                        shocked_returns[i] = shocks[sector]
                    else:
                        # Default market shock
                        shocked_returns[i] = shocks.get('market', -0.1)
            
            # Calculate portfolio impact
            portfolio_return = np.dot(weights, shocked_returns)
            portfolio_loss = -portfolio_return if portfolio_return < 0 else 0
            
            # Find worst and best performing assets
            worst_idx = np.argmin(shocked_returns)
            best_idx = np.argmax(shocked_returns)
            
            worst_asset = assets[worst_idx]
            worst_asset_loss = -shocked_returns[worst_idx]
            best_asset = assets[best_idx]
            best_asset_gain = shocked_returns[best_idx]
            
            # Sector impacts
            sector_impacts = {}
            if sector_mapping:
                sectors = set(sector_mapping.values())
                for sector in sectors:
                    sector_assets = [i for i, asset in enumerate(assets) if sector_mapping.get(asset) == sector]
                    sector_weights = np.array([weights[i] for i in sector_assets])
                    sector_returns = np.array([shocked_returns[i] for i in sector_assets])
                    
                    if len(sector_weights) > 0:
                        sector_impact = np.dot(sector_weights, sector_returns) / sector_weights.sum()
                        sector_impacts[sector] = sector_impact
            
            # Estimate recovery time (simplified heuristic)
            severity = abs(portfolio_return)
            if severity < 0.05:
                recovery_time = 30
            elif severity < 0.15:
                recovery_time = 90
            elif severity < 0.30:
                recovery_time = 180
            else:
                recovery_time = 365
            
            return StressTestResult(
                scenario_name=scenario_name,
                portfolio_return=portfolio_return,
                portfolio_loss=portfolio_loss,
                worst_asset=worst_asset,
                worst_asset_loss=worst_asset_loss,
                best_asset=best_asset,
                best_asset_gain=best_asset_gain,
                sector_impacts=sector_impacts,
                correlation_breakdown=severity > 0.20,  # Assume correlation breakdown in severe scenarios
                liquidity_impact=min(severity * 2, 0.5),  # Liquidity impact as function of severity
                recovery_time_estimate=recovery_time
            )
            
        except Exception as e:
            logger.error(f"Stress scenario application error: {str(e)}")
            return StressTestResult(
                scenario_name=scenario_name,
                portfolio_return=0.0,
                portfolio_loss=0.0,
                worst_asset="",
                worst_asset_loss=0.0,
                best_asset="",
                best_asset_gain=0.0,
                sector_impacts={},
                correlation_breakdown=False,
                liquidity_impact=0.0,
                recovery_time_estimate=0
            )
    
    def _initialize_stress_scenarios(self) -> Dict[str, Dict[str, float]]:
        """Initialize predefined stress test scenarios"""
        return {
            'market_crash': {
                'market': -0.30,
                'Technology': -0.40,
                'Financial': -0.35,
                'Energy': -0.25,
                'Healthcare': -0.15,
                'Utilities': -0.10,
                'Consumer Staples': -0.12,
                'Consumer Discretionary': -0.35,
                'Materials': -0.28,
                'Industrials': -0.25,
                'Real Estate': -0.30,
                'Communication Services': -0.25
            },
            'interest_rate_shock': {
                'market': -0.15,
                'Financial': 0.10,  # Banks benefit from higher rates
                'Real Estate': -0.25,
                'Utilities': -0.20,
                'Technology': -0.18,
                'Consumer Discretionary': -0.15,
                'Materials': -0.10,
                'Energy': -0.05,
                'Healthcare': -0.08,
                'Consumer Staples': -0.05,
                'Industrials': -0.12,
                'Communication Services': -0.10
            },
            'credit_crisis': {
                'market': -0.25,
                'Financial': -0.45,
                'Real Estate': -0.35,
                'Energy': -0.30,
                'Materials': -0.25,
                'Industrials': -0.20,
                'Consumer Discretionary': -0.25,
                'Technology': -0.15,
                'Communication Services': -0.18,
                'Healthcare': -0.10,
                'Consumer Staples': -0.08,
                'Utilities': -0.05
            },
            'inflation_spike': {
                'market': -0.12,
                'Energy': 0.15,
                'Materials': 0.10,
                'Real Estate': 0.05,
                'Financial': -0.05,
                'Consumer Staples': -0.08,
                'Utilities': -0.10,
                'Healthcare': -0.05,
                'Technology': -0.15,
                'Consumer Discretionary': -0.18,
                'Communication Services': -0.10,
                'Industrials': -0.08
            },
            'currency_crisis': {
                'market': -0.20,
                'Financial': -0.30,
                'Technology': -0.15,  # Often export-oriented
                'Materials': -0.10,   # Commodity exporters
                'Energy': -0.05,
                'Consumer Discretionary': -0.25,
                'Consumer Staples': -0.15,
                'Healthcare': -0.10,
                'Utilities': -0.20,
                'Industrials': -0.15,
                'Communication Services': -0.18,
                'Real Estate': -0.25
            },
            'liquidity_crisis': {
                'market': -0.18,
                'Financial': -0.35,
                'Real Estate': -0.30,
                'Technology': -0.20,
                'Consumer Discretionary': -0.22,
                'Materials': -0.15,
                'Energy': -0.12,
                'Industrials': -0.18,
                'Communication Services': -0.15,
                'Healthcare': -0.08,
                'Consumer Staples': -0.05,
                'Utilities': -0.03
            },
            'geopolitical_shock': {
                'market': -0.15,
                'Energy': 0.20,
                'Materials': 0.05,
                'Financial': -0.20,
                'Technology': -0.18,
                'Consumer Discretionary': -0.15,
                'Industrials': -0.12,
                'Communication Services': -0.10,
                'Healthcare': -0.05,
                'Consumer Staples': -0.03,
                'Utilities': 0.02,
                'Real Estate': -0.10
            },
            'pandemic_scenario': {
                'market': -0.25,
                'Technology': 0.10,   # Work from home beneficiaries
                'Healthcare': 0.05,
                'Consumer Staples': 0.02,
                'Utilities': -0.02,
                'Communication Services': 0.08,
                'Energy': -0.40,
                'Financial': -0.30,
                'Consumer Discretionary': -0.35,
                'Industrials': -0.25,
                'Materials': -0.20,
                'Real Estate': -0.30
            },
            'tech_bubble_burst': {
                'market': -0.20,
                'Technology': -0.50,
                'Communication Services': -0.35,
                'Consumer Discretionary': -0.25,
                'Financial': -0.15,
                'Industrials': -0.10,
                'Materials': -0.08,
                'Energy': -0.05,
                'Healthcare': -0.05,
                'Consumer Staples': -0.03,
                'Utilities': 0.02,
                'Real Estate': -0.10
            },
            'energy_crisis': {
                'market': -0.15,
                'Energy': 0.30,
                'Materials': 0.10,
                'Utilities': 0.05,
                'Consumer Staples': -0.08,
                'Healthcare': -0.05,
                'Financial': -0.10,
                'Technology': -0.12,
                'Consumer Discretionary': -0.20,
                'Industrials': -0.15,
                'Communication Services': -0.08,
                'Real Estate': -0.12
            }
        }

