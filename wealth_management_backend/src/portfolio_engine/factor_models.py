import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA, FactorAnalysis
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class FactorModelType(Enum):
    """Types of factor models"""
    FAMA_FRENCH_3 = "fama_french_3"
    FAMA_FRENCH_5 = "fama_french_5"
    CARHART_4 = "carhart_4"
    CAPM = "capm"
    ARBITRAGE_PRICING = "arbitrage_pricing"
    PRINCIPAL_COMPONENT = "principal_component"
    STATISTICAL_FACTOR = "statistical_factor"
    CUSTOM = "custom"

class RiskFactorType(Enum):
    """Types of risk factors"""
    MARKET = "market"
    SIZE = "size"
    VALUE = "value"
    PROFITABILITY = "profitability"
    INVESTMENT = "investment"
    MOMENTUM = "momentum"
    QUALITY = "quality"
    VOLATILITY = "volatility"
    LIQUIDITY = "liquidity"
    CREDIT = "credit"
    TERM_STRUCTURE = "term_structure"
    CURRENCY = "currency"
    COMMODITY = "commodity"

@dataclass
class FactorExposure:
    """Factor exposure result"""
    factor_name: str
    exposure: float
    t_statistic: float
    p_value: float
    confidence_interval: Tuple[float, float]
    r_squared_contribution: float
    volatility_contribution: float

@dataclass
class FactorModelResult:
    """Factor model analysis result"""
    model_type: str
    alpha: float
    alpha_t_stat: float
    alpha_p_value: float
    r_squared: float
    adjusted_r_squared: float
    factor_exposures: List[FactorExposure]
    residual_volatility: float
    total_volatility: float
    systematic_risk: float
    idiosyncratic_risk: float
    information_coefficient: float
    tracking_error: float

@dataclass
class RiskDecomposition:
    """Risk decomposition by factors"""
    total_risk: float
    systematic_risk: float
    idiosyncratic_risk: float
    factor_contributions: Dict[str, float]
    factor_correlations: Dict[str, Dict[str, float]]
    diversification_ratio: float

class FactorModel:
    """
    Advanced Factor Model Analysis Engine
    
    Features:
    - Multiple factor model implementations
    - Statistical factor extraction (PCA, Factor Analysis)
    - Risk decomposition and attribution
    - Factor timing and selection analysis
    - Custom factor model construction
    - Factor performance analysis
    - Risk budgeting by factors
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.factor_cache = {}
        
        # Predefined factor definitions
        self.factor_definitions = self._initialize_factor_definitions()
    
    def fit_factor_model(
        self,
        returns: pd.Series,
        factor_returns: pd.DataFrame,
        model_type: FactorModelType = FactorModelType.FAMA_FRENCH_3,
        regularization: Optional[str] = None,
        alpha: float = 0.01
    ) -> FactorModelResult:
        """
        Fit factor model to return series
        
        Args:
            returns: Asset or portfolio returns
            factor_returns: Factor returns DataFrame
            model_type: Type of factor model to fit
            regularization: Regularization method ('ridge', 'lasso', None)
            alpha: Regularization strength
            
        Returns:
            FactorModelResult with comprehensive analysis
        """
        try:
            # Align data
            common_dates = returns.index.intersection(factor_returns.index)
            returns_aligned = returns.loc[common_dates]
            factors_aligned = factor_returns.loc[common_dates]
            
            if len(common_dates) < 30:  # Minimum observations
                raise ValueError("Insufficient data for factor model fitting")
            
            # Select factors based on model type
            selected_factors = self._select_factors(factors_aligned, model_type)
            
            if selected_factors.empty:
                raise ValueError(f"No factors available for model type {model_type.value}")
            
            # Prepare regression data
            y = returns_aligned.values
            X = selected_factors.values
            
            # Add intercept
            X_with_intercept = np.column_stack([np.ones(len(X)), X])
            
            # Fit model with optional regularization
            if regularization == 'ridge':
                model = Ridge(alpha=alpha, fit_intercept=False)
                model.fit(X_with_intercept, y)
                coefficients = model.coef_
            elif regularization == 'lasso':
                model = Lasso(alpha=alpha, fit_intercept=False)
                model.fit(X_with_intercept, y)
                coefficients = model.coef_
            else:
                # OLS regression
                coefficients = np.linalg.lstsq(X_with_intercept, y, rcond=None)[0]
            
            # Extract alpha and factor loadings
            alpha_coef = coefficients[0]
            factor_loadings = coefficients[1:]
            
            # Calculate predictions and residuals
            y_pred = X_with_intercept @ coefficients
            residuals = y - y_pred
            
            # Calculate statistics
            n_obs = len(y)
            n_factors = len(factor_loadings)
            
            # R-squared
            ss_res = np.sum(residuals ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
            adjusted_r_squared = 1 - (1 - r_squared) * (n_obs - 1) / (n_obs - n_factors - 1)
            
            # Standard errors and t-statistics
            mse = ss_res / (n_obs - n_factors - 1)
            var_covar_matrix = mse * np.linalg.inv(X_with_intercept.T @ X_with_intercept)
            std_errors = np.sqrt(np.diag(var_covar_matrix))
            
            t_statistics = coefficients / std_errors
            p_values = 2 * (1 - stats.t.cdf(np.abs(t_statistics), n_obs - n_factors - 1))
            
            # Alpha statistics
            alpha_t_stat = t_statistics[0]
            alpha_p_value = p_values[0]
            
            # Factor exposures
            factor_exposures = []
            factor_names = selected_factors.columns.tolist()
            
            for i, factor_name in enumerate(factor_names):
                # Confidence interval (95%)
                t_critical = stats.t.ppf(0.975, n_obs - n_factors - 1)
                ci_lower = factor_loadings[i] - t_critical * std_errors[i + 1]
                ci_upper = factor_loadings[i] + t_critical * std_errors[i + 1]
                
                # Risk contribution
                factor_var = selected_factors[factor_name].var()
                volatility_contribution = (factor_loadings[i] ** 2) * factor_var
                
                # R-squared contribution (partial R-squared)
                factor_only_model = LinearRegression().fit(
                    selected_factors[[factor_name]].values, returns_aligned.values
                )
                factor_r_squared = factor_only_model.score(
                    selected_factors[[factor_name]].values, returns_aligned.values
                )
                
                factor_exposures.append(FactorExposure(
                    factor_name=factor_name,
                    exposure=factor_loadings[i],
                    t_statistic=t_statistics[i + 1],
                    p_value=p_values[i + 1],
                    confidence_interval=(ci_lower, ci_upper),
                    r_squared_contribution=factor_r_squared,
                    volatility_contribution=volatility_contribution
                ))
            
            # Risk decomposition
            total_volatility = returns_aligned.std() * np.sqrt(252)  # Annualized
            residual_volatility = np.std(residuals) * np.sqrt(252)  # Annualized
            
            # Systematic vs idiosyncratic risk
            systematic_variance = np.var(y_pred)
            idiosyncratic_variance = np.var(residuals)
            
            systematic_risk = np.sqrt(systematic_variance) * np.sqrt(252)
            idiosyncratic_risk = np.sqrt(idiosyncratic_variance) * np.sqrt(252)
            
            # Information coefficient (correlation between predicted and actual returns)
            information_coefficient = np.corrcoef(y_pred, y)[0, 1] if len(y) > 1 else 0
            
            # Tracking error (if this is active return)
            tracking_error = residual_volatility
            
            return FactorModelResult(
                model_type=model_type.value,
                alpha=alpha_coef * 252,  # Annualized
                alpha_t_stat=alpha_t_stat,
                alpha_p_value=alpha_p_value,
                r_squared=r_squared,
                adjusted_r_squared=adjusted_r_squared,
                factor_exposures=factor_exposures,
                residual_volatility=residual_volatility,
                total_volatility=total_volatility,
                systematic_risk=systematic_risk,
                idiosyncratic_risk=idiosyncratic_risk,
                information_coefficient=information_coefficient,
                tracking_error=tracking_error
            )
            
        except Exception as e:
            logger.error(f"Factor model fitting error: {str(e)}")
            return FactorModelResult(
                model_type=model_type.value,
                alpha=0.0, alpha_t_stat=0.0, alpha_p_value=1.0,
                r_squared=0.0, adjusted_r_squared=0.0,
                factor_exposures=[], residual_volatility=0.0,
                total_volatility=0.0, systematic_risk=0.0,
                idiosyncratic_risk=0.0, information_coefficient=0.0,
                tracking_error=0.0
            )
    
    def extract_statistical_factors(
        self,
        returns: pd.DataFrame,
        n_factors: int = 5,
        method: str = 'pca'
    ) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
        """
        Extract statistical factors from return data
        
        Args:
            returns: Asset returns DataFrame
            n_factors: Number of factors to extract
            method: Extraction method ('pca' or 'factor_analysis')
            
        Returns:
            Tuple of (factor_returns, factor_loadings, explained_variance)
        """
        try:
            returns_clean = returns.dropna()
            
            if method == 'pca':
                # Principal Component Analysis
                pca = PCA(n_components=n_factors)
                factor_scores = pca.fit_transform(returns_clean)
                factor_loadings = pca.components_.T
                explained_variance = pca.explained_variance_ratio_
                
                # Create factor returns DataFrame
                factor_names = [f'PC{i+1}' for i in range(n_factors)]
                factor_returns = pd.DataFrame(
                    factor_scores,
                    index=returns_clean.index,
                    columns=factor_names
                )
                
                # Create factor loadings DataFrame
                factor_loadings_df = pd.DataFrame(
                    factor_loadings,
                    index=returns_clean.columns,
                    columns=factor_names
                )
                
            elif method == 'factor_analysis':
                # Factor Analysis
                fa = FactorAnalysis(n_components=n_factors, random_state=42)
                factor_scores = fa.fit_transform(returns_clean)
                factor_loadings = fa.components_.T
                
                # Explained variance approximation for Factor Analysis
                explained_variance = np.var(factor_scores, axis=0) / np.sum(np.var(factor_scores, axis=0))
                
                # Create factor returns DataFrame
                factor_names = [f'Factor{i+1}' for i in range(n_factors)]
                factor_returns = pd.DataFrame(
                    factor_scores,
                    index=returns_clean.index,
                    columns=factor_names
                )
                
                # Create factor loadings DataFrame
                factor_loadings_df = pd.DataFrame(
                    factor_loadings,
                    index=returns_clean.columns,
                    columns=factor_names
                )
                
            else:
                raise ValueError(f"Unknown method: {method}")
            
            return factor_returns, factor_loadings_df, explained_variance
            
        except Exception as e:
            logger.error(f"Statistical factor extraction error: {str(e)}")
            return pd.DataFrame(), pd.DataFrame(), np.array([])
    
    def decompose_portfolio_risk(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
        portfolio_weights: Optional[np.ndarray] = None,
        asset_factor_exposures: Optional[pd.DataFrame] = None
    ) -> RiskDecomposition:
        """
        Decompose portfolio risk by factors
        
        Args:
            portfolio_returns: Portfolio returns
            factor_returns: Factor returns
            portfolio_weights: Portfolio weights (if decomposing from holdings)
            asset_factor_exposures: Asset-level factor exposures
            
        Returns:
            RiskDecomposition with detailed risk attribution
        """
        try:
            # Fit factor model to portfolio
            factor_model = self.fit_factor_model(portfolio_returns, factor_returns)
            
            total_risk = factor_model.total_volatility
            systematic_risk = factor_model.systematic_risk
            idiosyncratic_risk = factor_model.idiosyncratic_risk
            
            # Factor contributions to risk
            factor_contributions = {}
            for exposure in factor_model.factor_exposures:
                factor_contributions[exposure.factor_name] = exposure.volatility_contribution
            
            # Factor correlations
            factor_correlations = {}
            for i, factor1 in enumerate(factor_returns.columns):
                factor_correlations[factor1] = {}
                for j, factor2 in enumerate(factor_returns.columns):
                    correlation = factor_returns[factor1].corr(factor_returns[factor2])
                    factor_correlations[factor1][factor2] = correlation
            
            # Diversification ratio
            if portfolio_weights is not None and asset_factor_exposures is not None:
                # Calculate weighted average individual risks
                individual_risks = []
                for asset in asset_factor_exposures.index:
                    asset_exposures = asset_factor_exposures.loc[asset]
                    asset_risk = np.sqrt(np.sum(asset_exposures ** 2))
                    individual_risks.append(asset_risk)
                
                weighted_avg_risk = np.dot(portfolio_weights, individual_risks)
                diversification_ratio = weighted_avg_risk / total_risk if total_risk > 0 else 1.0
            else:
                diversification_ratio = 1.0
            
            return RiskDecomposition(
                total_risk=total_risk,
                systematic_risk=systematic_risk,
                idiosyncratic_risk=idiosyncratic_risk,
                factor_contributions=factor_contributions,
                factor_correlations=factor_correlations,
                diversification_ratio=diversification_ratio
            )
            
        except Exception as e:
            logger.error(f"Risk decomposition error: {str(e)}")
            return RiskDecomposition(
                total_risk=0.0, systematic_risk=0.0, idiosyncratic_risk=0.0,
                factor_contributions={}, factor_correlations={},
                diversification_ratio=1.0
            )
    
    def analyze_factor_timing(
        self,
        portfolio_returns: pd.Series,
        factor_returns: pd.DataFrame,
        window: int = 252,
        step: int = 21
    ) -> pd.DataFrame:
        """
        Analyze factor timing ability over rolling windows
        
        Args:
            portfolio_returns: Portfolio returns
            factor_returns: Factor returns
            window: Rolling window size
            step: Step size between windows
            
        Returns:
            DataFrame with rolling factor exposures and timing metrics
        """
        try:
            results = []
            
            for start_idx in range(0, len(portfolio_returns) - window + 1, step):
                end_idx = start_idx + window
                
                # Extract window data
                window_returns = portfolio_returns.iloc[start_idx:end_idx]
                window_factors = factor_returns.iloc[start_idx:end_idx]
                
                # Fit factor model for window
                factor_model = self.fit_factor_model(window_returns, window_factors)
                
                # Store results
                result_dict = {
                    'date': window_returns.index[-1],
                    'alpha': factor_model.alpha,
                    'alpha_t_stat': factor_model.alpha_t_stat,
                    'r_squared': factor_model.r_squared
                }
                
                # Add factor exposures
                for exposure in factor_model.factor_exposures:
                    result_dict[f'{exposure.factor_name}_exposure'] = exposure.exposure
                    result_dict[f'{exposure.factor_name}_t_stat'] = exposure.t_statistic
                
                results.append(result_dict)
            
            results_df = pd.DataFrame(results).set_index('date')
            
            # Calculate timing metrics
            for exposure in factor_model.factor_exposures:
                factor_name = exposure.factor_name
                exposure_col = f'{factor_name}_exposure'
                
                if exposure_col in results_df.columns:
                    # Factor timing correlation
                    factor_performance = factor_returns[factor_name].rolling(window).mean()
                    factor_performance_aligned = factor_performance.loc[results_df.index]
                    
                    timing_correlation = results_df[exposure_col].corr(factor_performance_aligned)
                    results_df[f'{factor_name}_timing_correlation'] = timing_correlation
            
            return results_df
            
        except Exception as e:
            logger.error(f"Factor timing analysis error: {str(e)}")
            return pd.DataFrame()
    
    def build_custom_factor_model(
        self,
        factor_definitions: Dict[str, Dict[str, Any]],
        asset_data: pd.DataFrame,
        return_data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Build custom factor model from asset characteristics
        
        Args:
            factor_definitions: Dictionary defining custom factors
            asset_data: Asset characteristics data
            return_data: Asset returns data
            
        Returns:
            DataFrame with custom factor returns
        """
        try:
            factor_returns = pd.DataFrame(index=return_data.index)
            
            for factor_name, definition in factor_definitions.items():
                factor_type = definition.get('type', 'characteristic')
                
                if factor_type == 'characteristic':
                    # Build factor from asset characteristics
                    characteristic = definition.get('characteristic')
                    method = definition.get('method', 'long_short')
                    
                    if characteristic in asset_data.columns:
                        # Create factor portfolio
                        if method == 'long_short':
                            # Long-short portfolio based on characteristic
                            factor_portfolio = self._create_long_short_portfolio(
                                asset_data[characteristic], return_data
                            )
                        elif method == 'ranking':
                            # Ranking-based portfolio
                            factor_portfolio = self._create_ranking_portfolio(
                                asset_data[characteristic], return_data
                            )
                        else:
                            factor_portfolio = pd.Series(0, index=return_data.index)
                        
                        factor_returns[factor_name] = factor_portfolio
                
                elif factor_type == 'momentum':
                    # Momentum factor
                    lookback = definition.get('lookback', 252)
                    factor_returns[factor_name] = self._create_momentum_factor(
                        return_data, lookback
                    )
                
                elif factor_type == 'mean_reversion':
                    # Mean reversion factor
                    lookback = definition.get('lookback', 21)
                    factor_returns[factor_name] = self._create_mean_reversion_factor(
                        return_data, lookback
                    )
            
            return factor_returns
            
        except Exception as e:
            logger.error(f"Custom factor model building error: {str(e)}")
            return pd.DataFrame()
    
    def _select_factors(
        self,
        factor_returns: pd.DataFrame,
        model_type: FactorModelType
    ) -> pd.DataFrame:
        """Select factors based on model type"""
        
        available_factors = factor_returns.columns.tolist()
        
        if model_type == FactorModelType.CAPM:
            # Market factor only
            market_factors = ['Market', 'Mkt-RF', 'market', 'MKT']
            selected = [f for f in market_factors if f in available_factors]
            
        elif model_type == FactorModelType.FAMA_FRENCH_3:
            # Market, Size, Value
            ff3_factors = ['Mkt-RF', 'SMB', 'HML', 'Market', 'Size', 'Value']
            selected = [f for f in ff3_factors if f in available_factors]
            
        elif model_type == FactorModelType.FAMA_FRENCH_5:
            # Market, Size, Value, Profitability, Investment
            ff5_factors = ['Mkt-RF', 'SMB', 'HML', 'RMW', 'CMA', 'Market', 'Size', 'Value', 'Profitability', 'Investment']
            selected = [f for f in ff5_factors if f in available_factors]
            
        elif model_type == FactorModelType.CARHART_4:
            # Fama-French 3 + Momentum
            carhart_factors = ['Mkt-RF', 'SMB', 'HML', 'MOM', 'Market', 'Size', 'Value', 'Momentum']
            selected = [f for f in carhart_factors if f in available_factors]
            
        else:
            # Use all available factors
            selected = available_factors
        
        return factor_returns[selected] if selected else pd.DataFrame()
    
    def _create_long_short_portfolio(
        self,
        characteristic: pd.Series,
        returns: pd.DataFrame
    ) -> pd.Series:
        """Create long-short portfolio based on characteristic"""
        
        try:
            factor_returns = pd.Series(0.0, index=returns.index)
            
            for date in returns.index:
                if date in returns.index:
                    # Get returns for this date
                    date_returns = returns.loc[date]
                    
                    # Align with characteristics
                    common_assets = characteristic.index.intersection(date_returns.index)
                    
                    if len(common_assets) > 10:  # Minimum assets
                        char_values = characteristic.loc[common_assets]
                        asset_returns = date_returns.loc[common_assets]
                        
                        # Sort by characteristic
                        sorted_indices = char_values.argsort()
                        
                        # Long top 30%, short bottom 30%
                        n_assets = len(sorted_indices)
                        long_cutoff = int(0.7 * n_assets)
                        short_cutoff = int(0.3 * n_assets)
                        
                        long_assets = sorted_indices[long_cutoff:]
                        short_assets = sorted_indices[:short_cutoff]
                        
                        # Calculate factor return
                        long_return = asset_returns.iloc[long_assets].mean()
                        short_return = asset_returns.iloc[short_assets].mean()
                        
                        factor_returns.loc[date] = long_return - short_return
            
            return factor_returns
            
        except Exception as e:
            logger.error(f"Long-short portfolio creation error: {str(e)}")
            return pd.Series(0.0, index=returns.index)
    
    def _create_ranking_portfolio(
        self,
        characteristic: pd.Series,
        returns: pd.DataFrame
    ) -> pd.Series:
        """Create ranking-based portfolio"""
        
        # Simplified ranking portfolio (equal to long-short for now)
        return self._create_long_short_portfolio(characteristic, returns)
    
    def _create_momentum_factor(
        self,
        returns: pd.DataFrame,
        lookback: int = 252
    ) -> pd.Series:
        """Create momentum factor"""
        
        try:
            # Calculate past returns for each asset
            past_returns = returns.rolling(lookback).mean()
            
            # Create long-short momentum portfolio
            factor_returns = pd.Series(0.0, index=returns.index)
            
            for date in returns.index[lookback:]:
                if date in past_returns.index:
                    momentum_scores = past_returns.loc[date].dropna()
                    current_returns = returns.loc[date, momentum_scores.index]
                    
                    if len(momentum_scores) > 10:
                        # Sort by momentum
                        sorted_indices = momentum_scores.argsort()
                        
                        # Long top 30%, short bottom 30%
                        n_assets = len(sorted_indices)
                        long_cutoff = int(0.7 * n_assets)
                        short_cutoff = int(0.3 * n_assets)
                        
                        long_assets = sorted_indices[long_cutoff:]
                        short_assets = sorted_indices[:short_cutoff]
                        
                        # Calculate factor return
                        long_return = current_returns.iloc[long_assets].mean()
                        short_return = current_returns.iloc[short_assets].mean()
                        
                        factor_returns.loc[date] = long_return - short_return
            
            return factor_returns
            
        except Exception as e:
            logger.error(f"Momentum factor creation error: {str(e)}")
            return pd.Series(0.0, index=returns.index)
    
    def _create_mean_reversion_factor(
        self,
        returns: pd.DataFrame,
        lookback: int = 21
    ) -> pd.Series:
        """Create mean reversion factor"""
        
        try:
            # Calculate short-term returns
            short_returns = returns.rolling(lookback).mean()
            
            # Mean reversion is opposite of momentum
            factor_returns = pd.Series(0.0, index=returns.index)
            
            for date in returns.index[lookback:]:
                if date in short_returns.index:
                    reversion_scores = -short_returns.loc[date].dropna()  # Negative for mean reversion
                    current_returns = returns.loc[date, reversion_scores.index]
                    
                    if len(reversion_scores) > 10:
                        # Sort by mean reversion score
                        sorted_indices = reversion_scores.argsort()
                        
                        # Long top 30%, short bottom 30%
                        n_assets = len(sorted_indices)
                        long_cutoff = int(0.7 * n_assets)
                        short_cutoff = int(0.3 * n_assets)
                        
                        long_assets = sorted_indices[long_cutoff:]
                        short_assets = sorted_indices[:short_cutoff]
                        
                        # Calculate factor return
                        long_return = current_returns.iloc[long_assets].mean()
                        short_return = current_returns.iloc[short_assets].mean()
                        
                        factor_returns.loc[date] = long_return - short_return
            
            return factor_returns
            
        except Exception as e:
            logger.error(f"Mean reversion factor creation error: {str(e)}")
            return pd.Series(0.0, index=returns.index)
    
    def _initialize_factor_definitions(self) -> Dict[str, Dict[str, Any]]:
        """Initialize predefined factor definitions"""
        
        return {
            'market': {
                'description': 'Market risk factor',
                'type': 'market',
                'aliases': ['Market', 'Mkt-RF', 'MKT']
            },
            'size': {
                'description': 'Size factor (Small minus Big)',
                'type': 'characteristic',
                'aliases': ['SMB', 'Size']
            },
            'value': {
                'description': 'Value factor (High minus Low)',
                'type': 'characteristic',
                'aliases': ['HML', 'Value']
            },
            'profitability': {
                'description': 'Profitability factor (Robust minus Weak)',
                'type': 'characteristic',
                'aliases': ['RMW', 'Profitability']
            },
            'investment': {
                'description': 'Investment factor (Conservative minus Aggressive)',
                'type': 'characteristic',
                'aliases': ['CMA', 'Investment']
            },
            'momentum': {
                'description': 'Momentum factor',
                'type': 'momentum',
                'aliases': ['MOM', 'Momentum']
            },
            'quality': {
                'description': 'Quality factor',
                'type': 'characteristic',
                'aliases': ['Quality', 'QMJ']
            },
            'volatility': {
                'description': 'Volatility factor',
                'type': 'characteristic',
                'aliases': ['Volatility', 'Vol']
            }
        }

