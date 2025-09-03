import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class AttributionMethod(Enum):
    """Performance attribution methods"""
    BRINSON_HOOD_BEEBOWER = "brinson_hood_beebower"
    BRINSON_FACHLER = "brinson_fachler"
    SECTOR_ATTRIBUTION = "sector_attribution"
    FACTOR_ATTRIBUTION = "factor_attribution"
    SECURITY_SELECTION = "security_selection"

class PerformancePeriod(Enum):
    """Performance measurement periods"""
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    QUARTERLY = "quarterly"
    YEARLY = "yearly"
    INCEPTION_TO_DATE = "inception_to_date"

@dataclass
class PerformanceMetrics:
    """Comprehensive performance metrics"""
    total_return: float
    annualized_return: float
    volatility: float
    sharpe_ratio: float
    sortino_ratio: float
    calmar_ratio: float
    max_drawdown: float
    var_95: float
    cvar_95: float
    beta: Optional[float] = None
    alpha: Optional[float] = None
    tracking_error: Optional[float] = None
    information_ratio: Optional[float] = None
    up_capture: Optional[float] = None
    down_capture: Optional[float] = None
    hit_rate: Optional[float] = None
    average_win: Optional[float] = None
    average_loss: Optional[float] = None
    win_loss_ratio: Optional[float] = None

@dataclass
class AttributionResult:
    """Performance attribution result"""
    total_excess_return: float
    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    sector_attribution: Dict[str, Dict[str, float]]
    security_attribution: Dict[str, float]
    factor_attribution: Optional[Dict[str, float]] = None

@dataclass
class RiskAdjustedMetrics:
    """Risk-adjusted performance metrics"""
    sharpe_ratio: float
    treynor_ratio: float
    jensen_alpha: float
    modigliani_m2: float
    information_ratio: float
    appraisal_ratio: float
    burke_ratio: float
    pain_ratio: float

class PerformanceAnalyzer:
    """
    Comprehensive Performance Analysis Engine
    
    Features:
    - Performance attribution analysis
    - Risk-adjusted return metrics
    - Benchmark comparison and tracking
    - Factor-based performance analysis
    - Rolling performance analysis
    - Drawdown and recovery analysis
    - Performance persistence analysis
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.performance_cache = {}
    
    def analyze_performance(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        sector_returns: Optional[pd.DataFrame] = None,
        factor_returns: Optional[pd.DataFrame] = None,
        portfolio_weights: Optional[pd.DataFrame] = None,
        benchmark_weights: Optional[pd.DataFrame] = None,
        period: PerformancePeriod = PerformancePeriod.MONTHLY
    ) -> PerformanceMetrics:
        """
        Comprehensive performance analysis
        
        Args:
            portfolio_returns: Portfolio returns time series
            benchmark_returns: Benchmark returns for comparison
            sector_returns: Sector returns for attribution
            factor_returns: Factor returns for factor analysis
            portfolio_weights: Portfolio weights over time
            benchmark_weights: Benchmark weights over time
            period: Analysis period frequency
            
        Returns:
            PerformanceMetrics with comprehensive analysis
        """
        try:
            # Resample returns based on period
            if period != PerformancePeriod.DAILY:
                portfolio_returns = self._resample_returns(portfolio_returns, period)
                if benchmark_returns is not None:
                    benchmark_returns = self._resample_returns(benchmark_returns, period)
            
            # Basic performance metrics
            total_return = (1 + portfolio_returns).prod() - 1
            
            # Annualized return
            n_periods = len(portfolio_returns)
            if period == PerformancePeriod.DAILY:
                periods_per_year = 252
            elif period == PerformancePeriod.WEEKLY:
                periods_per_year = 52
            elif period == PerformancePeriod.MONTHLY:
                periods_per_year = 12
            elif period == PerformancePeriod.QUARTERLY:
                periods_per_year = 4
            else:
                periods_per_year = 1
            
            annualized_return = (1 + total_return) ** (periods_per_year / n_periods) - 1
            
            # Risk metrics
            volatility = portfolio_returns.std() * np.sqrt(periods_per_year)
            
            # Risk-adjusted metrics
            excess_returns = portfolio_returns - self.risk_free_rate / periods_per_year
            sharpe_ratio = excess_returns.mean() / portfolio_returns.std() if portfolio_returns.std() > 0 else 0
            
            # Downside metrics
            downside_returns = portfolio_returns[portfolio_returns < 0]
            downside_deviation = downside_returns.std() * np.sqrt(periods_per_year) if len(downside_returns) > 0 else 0
            sortino_ratio = excess_returns.mean() / (downside_deviation / np.sqrt(periods_per_year)) if downside_deviation > 0 else 0
            
            # Drawdown analysis
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            max_drawdown = drawdown.min()
            
            # Calmar ratio
            calmar_ratio = annualized_return / abs(max_drawdown) if max_drawdown != 0 else 0
            
            # VaR and CVaR
            var_95 = np.percentile(portfolio_returns, 5)
            cvar_95 = portfolio_returns[portfolio_returns <= var_95].mean() if len(portfolio_returns[portfolio_returns <= var_95]) > 0 else 0
            
            # Benchmark-relative metrics
            beta = None
            alpha = None
            tracking_error = None
            information_ratio = None
            up_capture = None
            down_capture = None
            
            if benchmark_returns is not None:
                # Align returns
                common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
                port_aligned = portfolio_returns.loc[common_dates]
                bench_aligned = benchmark_returns.loc[common_dates]
                
                if len(common_dates) > 1:
                    # Beta calculation
                    covariance = np.cov(port_aligned, bench_aligned)[0, 1]
                    benchmark_variance = np.var(bench_aligned)
                    beta = covariance / benchmark_variance if benchmark_variance > 0 else 0
                    
                    # Alpha calculation (Jensen's alpha)
                    portfolio_excess = port_aligned - self.risk_free_rate / periods_per_year
                    benchmark_excess = bench_aligned - self.risk_free_rate / periods_per_year
                    alpha = portfolio_excess.mean() - beta * benchmark_excess.mean()
                    alpha *= periods_per_year  # Annualize
                    
                    # Tracking error
                    active_returns = port_aligned - bench_aligned
                    tracking_error = active_returns.std() * np.sqrt(periods_per_year)
                    
                    # Information ratio
                    information_ratio = active_returns.mean() / active_returns.std() if active_returns.std() > 0 else 0
                    information_ratio *= np.sqrt(periods_per_year)  # Annualize
                    
                    # Up/Down capture ratios
                    up_periods = bench_aligned > 0
                    down_periods = bench_aligned < 0
                    
                    if up_periods.sum() > 0:
                        up_capture = port_aligned[up_periods].mean() / bench_aligned[up_periods].mean()
                    
                    if down_periods.sum() > 0:
                        down_capture = port_aligned[down_periods].mean() / bench_aligned[down_periods].mean()
            
            # Hit rate and win/loss analysis
            positive_periods = portfolio_returns > 0
            hit_rate = positive_periods.mean()
            
            wins = portfolio_returns[portfolio_returns > 0]
            losses = portfolio_returns[portfolio_returns < 0]
            
            average_win = wins.mean() if len(wins) > 0 else 0
            average_loss = losses.mean() if len(losses) > 0 else 0
            win_loss_ratio = abs(average_win / average_loss) if average_loss != 0 else 0
            
            return PerformanceMetrics(
                total_return=total_return,
                annualized_return=annualized_return,
                volatility=volatility,
                sharpe_ratio=sharpe_ratio,
                sortino_ratio=sortino_ratio,
                calmar_ratio=calmar_ratio,
                max_drawdown=max_drawdown,
                var_95=var_95,
                cvar_95=cvar_95,
                beta=beta,
                alpha=alpha,
                tracking_error=tracking_error,
                information_ratio=information_ratio,
                up_capture=up_capture,
                down_capture=down_capture,
                hit_rate=hit_rate,
                average_win=average_win,
                average_loss=average_loss,
                win_loss_ratio=win_loss_ratio
            )
            
        except Exception as e:
            logger.error(f"Performance analysis error: {str(e)}")
            return PerformanceMetrics(
                total_return=0.0, annualized_return=0.0, volatility=0.0,
                sharpe_ratio=0.0, sortino_ratio=0.0, calmar_ratio=0.0,
                max_drawdown=0.0, var_95=0.0, cvar_95=0.0
            )
    
    def calculate_attribution(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        portfolio_weights: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        sector_mapping: Dict[str, str],
        method: AttributionMethod = AttributionMethod.BRINSON_HOOD_BEEBOWER
    ) -> AttributionResult:
        """
        Calculate performance attribution
        
        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            portfolio_weights: Portfolio weights by asset over time
            benchmark_weights: Benchmark weights by asset over time
            sector_mapping: Asset to sector mapping
            method: Attribution method to use
            
        Returns:
            AttributionResult with detailed attribution analysis
        """
        try:
            if method == AttributionMethod.BRINSON_HOOD_BEEBOWER:
                return self._brinson_hood_beebower_attribution(
                    portfolio_returns, benchmark_returns, portfolio_weights,
                    benchmark_weights, sector_mapping
                )
            elif method == AttributionMethod.SECTOR_ATTRIBUTION:
                return self._sector_attribution(
                    portfolio_returns, benchmark_returns, portfolio_weights,
                    benchmark_weights, sector_mapping
                )
            else:
                # Default to Brinson-Hood-Beebower
                return self._brinson_hood_beebower_attribution(
                    portfolio_returns, benchmark_returns, portfolio_weights,
                    benchmark_weights, sector_mapping
                )
                
        except Exception as e:
            logger.error(f"Attribution calculation error: {str(e)}")
            return AttributionResult(
                total_excess_return=0.0,
                allocation_effect=0.0,
                selection_effect=0.0,
                interaction_effect=0.0,
                sector_attribution={},
                security_attribution={}
            )
    
    def calculate_rolling_performance(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: Optional[pd.Series] = None,
        window: int = 252,  # 1 year for daily data
        step: int = 21  # Monthly steps
    ) -> pd.DataFrame:
        """
        Calculate rolling performance metrics
        
        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns for comparison
            window: Rolling window size
            step: Step size between calculations
            
        Returns:
            DataFrame with rolling performance metrics
        """
        try:
            results = []
            
            for start_idx in range(0, len(portfolio_returns) - window + 1, step):
                end_idx = start_idx + window
                
                # Extract window data
                window_returns = portfolio_returns.iloc[start_idx:end_idx]
                window_bench = benchmark_returns.iloc[start_idx:end_idx] if benchmark_returns is not None else None
                
                # Calculate metrics for window
                metrics = self.analyze_performance(window_returns, window_bench)
                
                # Store results
                result_dict = {
                    'date': window_returns.index[-1],
                    'total_return': metrics.total_return,
                    'annualized_return': metrics.annualized_return,
                    'volatility': metrics.volatility,
                    'sharpe_ratio': metrics.sharpe_ratio,
                    'max_drawdown': metrics.max_drawdown,
                    'var_95': metrics.var_95
                }
                
                if metrics.beta is not None:
                    result_dict.update({
                        'beta': metrics.beta,
                        'alpha': metrics.alpha,
                        'tracking_error': metrics.tracking_error,
                        'information_ratio': metrics.information_ratio
                    })
                
                results.append(result_dict)
            
            return pd.DataFrame(results).set_index('date')
            
        except Exception as e:
            logger.error(f"Rolling performance calculation error: {str(e)}")
            return pd.DataFrame()
    
    def analyze_drawdowns(
        self,
        portfolio_returns: pd.Series,
        threshold: float = 0.05
    ) -> Dict[str, Any]:
        """
        Detailed drawdown analysis
        
        Args:
            portfolio_returns: Portfolio returns
            threshold: Minimum drawdown threshold for analysis
            
        Returns:
            Comprehensive drawdown analysis
        """
        try:
            # Calculate cumulative returns and drawdowns
            cumulative_returns = (1 + portfolio_returns).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            
            # Find drawdown periods
            in_drawdown = drawdown < -threshold
            
            # Identify individual drawdown periods
            drawdown_periods = []
            start_idx = None
            
            for i, is_dd in enumerate(in_drawdown):
                if is_dd and start_idx is None:
                    start_idx = i
                elif not is_dd and start_idx is not None:
                    # End of drawdown period
                    end_idx = i - 1
                    period_drawdown = drawdown.iloc[start_idx:end_idx+1]
                    max_dd = period_drawdown.min()
                    duration = end_idx - start_idx + 1
                    
                    # Recovery analysis
                    recovery_idx = None
                    for j in range(end_idx + 1, len(cumulative_returns)):
                        if cumulative_returns.iloc[j] >= running_max.iloc[start_idx]:
                            recovery_idx = j
                            break
                    
                    recovery_time = recovery_idx - end_idx if recovery_idx else None
                    
                    drawdown_periods.append({
                        'start_date': portfolio_returns.index[start_idx],
                        'end_date': portfolio_returns.index[end_idx],
                        'duration': duration,
                        'max_drawdown': max_dd,
                        'recovery_time': recovery_time,
                        'recovered': recovery_idx is not None
                    })
                    
                    start_idx = None
            
            # Handle ongoing drawdown
            if start_idx is not None:
                period_drawdown = drawdown.iloc[start_idx:]
                max_dd = period_drawdown.min()
                duration = len(period_drawdown)
                
                drawdown_periods.append({
                    'start_date': portfolio_returns.index[start_idx],
                    'end_date': portfolio_returns.index[-1],
                    'duration': duration,
                    'max_drawdown': max_dd,
                    'recovery_time': None,
                    'recovered': False
                })
            
            # Summary statistics
            if drawdown_periods:
                max_drawdown = min([dd['max_drawdown'] for dd in drawdown_periods])
                avg_drawdown = np.mean([dd['max_drawdown'] for dd in drawdown_periods])
                avg_duration = np.mean([dd['duration'] for dd in drawdown_periods])
                
                recovered_periods = [dd for dd in drawdown_periods if dd['recovered']]
                avg_recovery_time = np.mean([dd['recovery_time'] for dd in recovered_periods]) if recovered_periods else None
            else:
                max_drawdown = 0.0
                avg_drawdown = 0.0
                avg_duration = 0.0
                avg_recovery_time = None
            
            return {
                'max_drawdown': max_drawdown,
                'average_drawdown': avg_drawdown,
                'number_of_drawdowns': len(drawdown_periods),
                'average_duration': avg_duration,
                'average_recovery_time': avg_recovery_time,
                'drawdown_periods': drawdown_periods,
                'current_drawdown': drawdown.iloc[-1],
                'time_in_drawdown': in_drawdown.mean()
            }
            
        except Exception as e:
            logger.error(f"Drawdown analysis error: {str(e)}")
            return {}
    
    def calculate_risk_adjusted_metrics(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        market_returns: Optional[pd.Series] = None
    ) -> RiskAdjustedMetrics:
        """
        Calculate comprehensive risk-adjusted performance metrics
        
        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            market_returns: Market returns (for beta calculation)
            
        Returns:
            RiskAdjustedMetrics with comprehensive risk-adjusted measures
        """
        try:
            # Align returns
            common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
            port_ret = portfolio_returns.loc[common_dates]
            bench_ret = benchmark_returns.loc[common_dates]
            
            if market_returns is not None:
                market_ret = market_returns.loc[common_dates]
            else:
                market_ret = bench_ret  # Use benchmark as market proxy
            
            # Basic metrics
            port_excess = port_ret - self.risk_free_rate / 252
            bench_excess = bench_ret - self.risk_free_rate / 252
            market_excess = market_ret - self.risk_free_rate / 252
            
            # Sharpe ratio
            sharpe_ratio = port_excess.mean() / port_ret.std() * np.sqrt(252) if port_ret.std() > 0 else 0
            
            # Beta calculation
            beta = np.cov(port_ret, market_ret)[0, 1] / np.var(market_ret) if np.var(market_ret) > 0 else 0
            
            # Treynor ratio
            treynor_ratio = port_excess.mean() * 252 / beta if beta != 0 else 0
            
            # Jensen's alpha
            jensen_alpha = port_excess.mean() - beta * market_excess.mean()
            jensen_alpha *= 252  # Annualize
            
            # Modigliani M²
            benchmark_vol = bench_ret.std() * np.sqrt(252)
            portfolio_vol = port_ret.std() * np.sqrt(252)
            if portfolio_vol > 0:
                adjusted_return = (port_excess.mean() * 252) * (benchmark_vol / portfolio_vol)
                modigliani_m2 = adjusted_return - (bench_excess.mean() * 252)
            else:
                modigliani_m2 = 0
            
            # Information ratio
            active_returns = port_ret - bench_ret
            information_ratio = active_returns.mean() / active_returns.std() * np.sqrt(252) if active_returns.std() > 0 else 0
            
            # Appraisal ratio (similar to information ratio but uses systematic risk)
            appraisal_ratio = jensen_alpha / (port_ret.std() * np.sqrt(252) * np.sqrt(1 - beta**2)) if beta != 1 else 0
            
            # Burke ratio (return to drawdown)
            cumulative_returns = (1 + port_ret).cumprod()
            running_max = cumulative_returns.expanding().max()
            drawdown = (cumulative_returns - running_max) / running_max
            drawdown_squared_sum = (drawdown ** 2).sum()
            burke_ratio = port_ret.mean() * 252 / np.sqrt(drawdown_squared_sum) if drawdown_squared_sum > 0 else 0
            
            # Pain ratio (return to average drawdown)
            avg_drawdown = abs(drawdown.mean())
            pain_ratio = port_ret.mean() * 252 / avg_drawdown if avg_drawdown > 0 else 0
            
            return RiskAdjustedMetrics(
                sharpe_ratio=sharpe_ratio,
                treynor_ratio=treynor_ratio,
                jensen_alpha=jensen_alpha,
                modigliani_m2=modigliani_m2,
                information_ratio=information_ratio,
                appraisal_ratio=appraisal_ratio,
                burke_ratio=burke_ratio,
                pain_ratio=pain_ratio
            )
            
        except Exception as e:
            logger.error(f"Risk-adjusted metrics calculation error: {str(e)}")
            return RiskAdjustedMetrics(
                sharpe_ratio=0.0, treynor_ratio=0.0, jensen_alpha=0.0,
                modigliani_m2=0.0, information_ratio=0.0, appraisal_ratio=0.0,
                burke_ratio=0.0, pain_ratio=0.0
            )
    
    def _resample_returns(self, returns: pd.Series, period: PerformancePeriod) -> pd.Series:
        """Resample returns to specified frequency"""
        
        if period == PerformancePeriod.WEEKLY:
            return (1 + returns).resample('W').prod() - 1
        elif period == PerformancePeriod.MONTHLY:
            return (1 + returns).resample('M').prod() - 1
        elif period == PerformancePeriod.QUARTERLY:
            return (1 + returns).resample('Q').prod() - 1
        elif period == PerformancePeriod.YEARLY:
            return (1 + returns).resample('Y').prod() - 1
        else:
            return returns
    
    def _brinson_hood_beebower_attribution(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        portfolio_weights: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        sector_mapping: Dict[str, str]
    ) -> AttributionResult:
        """Brinson-Hood-Beebower attribution analysis"""
        
        try:
            # Align data
            common_dates = portfolio_returns.index.intersection(benchmark_returns.index)
            common_assets = portfolio_weights.columns.intersection(benchmark_weights.columns)
            
            # Calculate sector returns and weights
            sectors = set(sector_mapping.values())
            sector_attribution = {}
            
            total_allocation_effect = 0
            total_selection_effect = 0
            total_interaction_effect = 0
            
            for sector in sectors:
                sector_assets = [asset for asset in common_assets if sector_mapping.get(asset) == sector]
                
                if not sector_assets:
                    continue
                
                # Calculate sector weights and returns
                sector_port_weights = portfolio_weights[sector_assets].sum(axis=1)
                sector_bench_weights = benchmark_weights[sector_assets].sum(axis=1)
                
                # Simplified sector return calculation (would need asset returns)
                # For now, use approximation based on portfolio/benchmark returns
                sector_port_return = portfolio_returns.mean()  # Simplified
                sector_bench_return = benchmark_returns.mean()  # Simplified
                
                # Attribution effects
                avg_port_weight = sector_port_weights.mean()
                avg_bench_weight = sector_bench_weights.mean()
                
                allocation_effect = (avg_port_weight - avg_bench_weight) * sector_bench_return
                selection_effect = avg_bench_weight * (sector_port_return - sector_bench_return)
                interaction_effect = (avg_port_weight - avg_bench_weight) * (sector_port_return - sector_bench_return)
                
                sector_attribution[sector] = {
                    'allocation_effect': allocation_effect,
                    'selection_effect': selection_effect,
                    'interaction_effect': interaction_effect,
                    'total_effect': allocation_effect + selection_effect + interaction_effect
                }
                
                total_allocation_effect += allocation_effect
                total_selection_effect += selection_effect
                total_interaction_effect += interaction_effect
            
            # Security-level attribution (simplified)
            security_attribution = {}
            for asset in common_assets:
                # Simplified security attribution
                avg_port_weight = portfolio_weights[asset].mean()
                avg_bench_weight = benchmark_weights[asset].mean()
                
                # Would need individual asset returns for proper calculation
                security_attribution[asset] = (avg_port_weight - avg_bench_weight) * 0.01  # Placeholder
            
            total_excess_return = portfolio_returns.mean() - benchmark_returns.mean()
            
            return AttributionResult(
                total_excess_return=total_excess_return,
                allocation_effect=total_allocation_effect,
                selection_effect=total_selection_effect,
                interaction_effect=total_interaction_effect,
                sector_attribution=sector_attribution,
                security_attribution=security_attribution
            )
            
        except Exception as e:
            logger.error(f"Brinson-Hood-Beebower attribution error: {str(e)}")
            return AttributionResult(
                total_excess_return=0.0,
                allocation_effect=0.0,
                selection_effect=0.0,
                interaction_effect=0.0,
                sector_attribution={},
                security_attribution={}
            )
    
    def _sector_attribution(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        portfolio_weights: pd.DataFrame,
        benchmark_weights: pd.DataFrame,
        sector_mapping: Dict[str, str]
    ) -> AttributionResult:
        """Sector-based attribution analysis"""
        
        # Simplified sector attribution
        # In practice, this would require detailed sector return calculations
        return self._brinson_hood_beebower_attribution(
            portfolio_returns, benchmark_returns, portfolio_weights,
            benchmark_weights, sector_mapping
        )
    
    def calculate_performance_persistence(
        self,
        portfolio_returns: pd.Series,
        benchmark_returns: pd.Series,
        window: int = 252,
        step: int = 63  # Quarterly
    ) -> Dict[str, Any]:
        """
        Analyze performance persistence over time
        
        Args:
            portfolio_returns: Portfolio returns
            benchmark_returns: Benchmark returns
            window: Analysis window size
            step: Step size between analyses
            
        Returns:
            Performance persistence analysis
        """
        try:
            rolling_metrics = self.calculate_rolling_performance(
                portfolio_returns, benchmark_returns, window, step
            )
            
            if rolling_metrics.empty:
                return {}
            
            # Calculate persistence metrics
            outperformance = rolling_metrics['information_ratio'] > 0
            
            # Consecutive periods of outperformance
            consecutive_periods = []
            current_streak = 0
            
            for outperf in outperformance:
                if outperf:
                    current_streak += 1
                else:
                    if current_streak > 0:
                        consecutive_periods.append(current_streak)
                    current_streak = 0
            
            if current_streak > 0:
                consecutive_periods.append(current_streak)
            
            # Persistence statistics
            hit_rate = outperformance.mean()
            max_consecutive_outperformance = max(consecutive_periods) if consecutive_periods else 0
            avg_consecutive_outperformance = np.mean(consecutive_periods) if consecutive_periods else 0
            
            # Rank correlation (Spearman) between consecutive periods
            if len(rolling_metrics) > 1:
                rank_correlation = rolling_metrics['information_ratio'].corr(
                    rolling_metrics['information_ratio'].shift(1), method='spearman'
                )
            else:
                rank_correlation = 0
            
            return {
                'hit_rate': hit_rate,
                'max_consecutive_outperformance': max_consecutive_outperformance,
                'average_consecutive_outperformance': avg_consecutive_outperformance,
                'rank_correlation': rank_correlation,
                'number_of_streaks': len(consecutive_periods),
                'rolling_metrics': rolling_metrics
            }
            
        except Exception as e:
            logger.error(f"Performance persistence analysis error: {str(e)}")
            return {}

