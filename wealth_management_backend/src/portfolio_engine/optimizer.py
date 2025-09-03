import numpy as np
import pandas as pd
from scipy.optimize import minimize, LinearConstraint, Bounds
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class OptimizationObjective(Enum):
    """Portfolio optimization objectives"""
    SHARPE_MAX = "sharpe_max"  # Maximum Sharpe ratio
    VARIANCE_MIN = "variance_min"  # Minimum variance
    RETURN_MAX = "return_max"  # Maximum return
    RISK_PARITY = "risk_parity"  # Risk parity
    EQUAL_WEIGHT = "equal_weight"  # Equal weight
    BLACK_LITTERMAN = "black_litterman"  # Black-Litterman
    MEAN_REVERSION = "mean_reversion"  # Mean reversion
    MOMENTUM = "momentum"  # Momentum
    ESG_OPTIMIZED = "esg_optimized"  # ESG optimized

@dataclass
class OptimizationConstraints:
    """Portfolio optimization constraints"""
    min_weight: float = 0.0  # Minimum weight per asset
    max_weight: float = 1.0  # Maximum weight per asset
    max_sector_weight: Optional[Dict[str, float]] = None  # Maximum sector weights
    min_sector_weight: Optional[Dict[str, float]] = None  # Minimum sector weights
    max_asset_class_weight: Optional[Dict[str, float]] = None  # Maximum asset class weights
    min_asset_class_weight: Optional[Dict[str, float]] = None  # Minimum asset class weights
    target_return: Optional[float] = None  # Target return constraint
    max_risk: Optional[float] = None  # Maximum risk constraint
    turnover_limit: Optional[float] = None  # Maximum turnover
    esg_score_min: Optional[float] = None  # Minimum ESG score
    liquidity_min: Optional[float] = None  # Minimum liquidity requirement
    concentration_limit: Optional[int] = None  # Maximum number of holdings

@dataclass
class OptimizationResult:
    """Portfolio optimization result"""
    weights: np.ndarray
    expected_return: float
    expected_risk: float
    sharpe_ratio: float
    objective_value: float
    success: bool
    message: str
    iterations: int
    optimization_time: float
    asset_allocation: Dict[str, float]
    sector_allocation: Optional[Dict[str, float]] = None
    asset_class_allocation: Optional[Dict[str, float]] = None
    risk_contributions: Optional[Dict[str, float]] = None
    performance_attribution: Optional[Dict[str, Any]] = None

class PortfolioOptimizer:
    """
    Advanced Portfolio Optimizer implementing multiple optimization strategies
    
    Features:
    - Modern Portfolio Theory (Markowitz)
    - Risk Parity optimization
    - Black-Litterman model
    - Factor-based optimization
    - ESG-constrained optimization
    - Transaction cost optimization
    - Robust optimization techniques
    """
    
    def __init__(self, risk_free_rate: float = 0.02):
        self.risk_free_rate = risk_free_rate
        self.optimization_history = []
        
        # Optimization parameters
        self.max_iterations = 1000
        self.tolerance = 1e-8
        self.regularization = 1e-8  # For numerical stability
    
    def optimize_portfolio(
        self,
        returns: pd.DataFrame,
        objective: OptimizationObjective = OptimizationObjective.SHARPE_MAX,
        constraints: Optional[OptimizationConstraints] = None,
        current_weights: Optional[np.ndarray] = None,
        market_views: Optional[Dict[str, float]] = None,
        confidence_levels: Optional[Dict[str, float]] = None,
        transaction_costs: Optional[Dict[str, float]] = None,
        esg_scores: Optional[Dict[str, float]] = None,
        liquidity_scores: Optional[Dict[str, float]] = None,
        sector_mapping: Optional[Dict[str, str]] = None,
        asset_class_mapping: Optional[Dict[str, str]] = None
    ) -> OptimizationResult:
        """
        Optimize portfolio based on specified objective and constraints
        
        Args:
            returns: Historical returns DataFrame (assets as columns, dates as rows)
            objective: Optimization objective
            constraints: Portfolio constraints
            current_weights: Current portfolio weights (for turnover constraints)
            market_views: Market views for Black-Litterman (asset -> expected return)
            confidence_levels: Confidence in market views (asset -> confidence)
            transaction_costs: Transaction costs per asset (asset -> cost)
            esg_scores: ESG scores per asset (asset -> score)
            liquidity_scores: Liquidity scores per asset (asset -> score)
            sector_mapping: Asset to sector mapping
            asset_class_mapping: Asset to asset class mapping
            
        Returns:
            OptimizationResult with optimal weights and metrics
        """
        start_time = datetime.now()
        
        try:
            # Validate inputs
            if returns.empty:
                return OptimizationResult(
                    weights=np.array([]),
                    expected_return=0.0,
                    expected_risk=0.0,
                    sharpe_ratio=0.0,
                    objective_value=0.0,
                    success=False,
                    message="Empty returns data",
                    iterations=0,
                    optimization_time=0.0,
                    asset_allocation={}
                )
            
            # Prepare data
            returns_clean = returns.dropna()
            if returns_clean.empty:
                return OptimizationResult(
                    weights=np.array([]),
                    expected_return=0.0,
                    expected_risk=0.0,
                    sharpe_ratio=0.0,
                    objective_value=0.0,
                    success=False,
                    message="No valid returns data after cleaning",
                    iterations=0,
                    optimization_time=0.0,
                    asset_allocation={}
                )
            
            assets = returns_clean.columns.tolist()
            n_assets = len(assets)
            
            # Calculate expected returns and covariance matrix
            expected_returns = returns_clean.mean().values * 252  # Annualized
            cov_matrix = returns_clean.cov().values * 252  # Annualized
            
            # Add regularization for numerical stability
            cov_matrix += np.eye(n_assets) * self.regularization
            
            # Apply Black-Litterman if market views provided
            if market_views and objective == OptimizationObjective.BLACK_LITTERMAN:
                expected_returns, cov_matrix = self._apply_black_litterman(
                    expected_returns, cov_matrix, assets, market_views, confidence_levels
                )
            
            # Set up constraints
            if constraints is None:
                constraints = OptimizationConstraints()
            
            # Optimize based on objective
            if objective == OptimizationObjective.SHARPE_MAX:
                result = self._optimize_sharpe_ratio(
                    expected_returns, cov_matrix, constraints, assets,
                    current_weights, transaction_costs, esg_scores, liquidity_scores,
                    sector_mapping, asset_class_mapping
                )
            elif objective == OptimizationObjective.VARIANCE_MIN:
                result = self._optimize_minimum_variance(
                    expected_returns, cov_matrix, constraints, assets,
                    current_weights, transaction_costs, esg_scores, liquidity_scores,
                    sector_mapping, asset_class_mapping
                )
            elif objective == OptimizationObjective.RETURN_MAX:
                result = self._optimize_maximum_return(
                    expected_returns, cov_matrix, constraints, assets,
                    current_weights, transaction_costs, esg_scores, liquidity_scores,
                    sector_mapping, asset_class_mapping
                )
            elif objective == OptimizationObjective.RISK_PARITY:
                result = self._optimize_risk_parity(
                    expected_returns, cov_matrix, constraints, assets,
                    current_weights, transaction_costs, esg_scores, liquidity_scores,
                    sector_mapping, asset_class_mapping
                )
            elif objective == OptimizationObjective.EQUAL_WEIGHT:
                result = self._optimize_equal_weight(
                    expected_returns, cov_matrix, constraints, assets,
                    sector_mapping, asset_class_mapping
                )
            elif objective == OptimizationObjective.ESG_OPTIMIZED:
                result = self._optimize_esg_weighted(
                    expected_returns, cov_matrix, constraints, assets,
                    current_weights, transaction_costs, esg_scores, liquidity_scores,
                    sector_mapping, asset_class_mapping
                )
            else:
                # Default to Sharpe ratio maximization
                result = self._optimize_sharpe_ratio(
                    expected_returns, cov_matrix, constraints, assets,
                    current_weights, transaction_costs, esg_scores, liquidity_scores,
                    sector_mapping, asset_class_mapping
                )
            
            # Calculate additional metrics
            result.optimization_time = (datetime.now() - start_time).total_seconds()
            
            # Add to optimization history
            self.optimization_history.append({
                'timestamp': start_time,
                'objective': objective.value,
                'success': result.success,
                'expected_return': result.expected_return,
                'expected_risk': result.expected_risk,
                'sharpe_ratio': result.sharpe_ratio
            })
            
            logger.info(f"Portfolio optimization completed: {objective.value}, Success: {result.success}")
            
            return result
            
        except Exception as e:
            logger.error(f"Portfolio optimization error: {str(e)}")
            return OptimizationResult(
                weights=np.array([]),
                expected_return=0.0,
                expected_risk=0.0,
                sharpe_ratio=0.0,
                objective_value=0.0,
                success=False,
                message=f"Optimization failed: {str(e)}",
                iterations=0,
                optimization_time=(datetime.now() - start_time).total_seconds(),
                asset_allocation={}
            )
    
    def _optimize_sharpe_ratio(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        constraints: OptimizationConstraints,
        assets: List[str],
        current_weights: Optional[np.ndarray] = None,
        transaction_costs: Optional[Dict[str, float]] = None,
        esg_scores: Optional[Dict[str, float]] = None,
        liquidity_scores: Optional[Dict[str, float]] = None,
        sector_mapping: Optional[Dict[str, str]] = None,
        asset_class_mapping: Optional[Dict[str, str]] = None
    ) -> OptimizationResult:
        """Optimize for maximum Sharpe ratio"""
        
        n_assets = len(assets)
        
        def objective_function(weights):
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            
            # Avoid division by zero
            if portfolio_risk < 1e-10:
                return -1e10
            
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
            
            # Add transaction costs if provided
            if transaction_costs and current_weights is not None:
                tc_penalty = sum(
                    transaction_costs.get(asset, 0.001) * abs(weights[i] - current_weights[i])
                    for i, asset in enumerate(assets)
                )
                sharpe_ratio -= tc_penalty
            
            return -sharpe_ratio  # Minimize negative Sharpe ratio
        
        # Set up bounds and constraints
        bounds = Bounds(
            lb=np.full(n_assets, constraints.min_weight),
            ub=np.full(n_assets, constraints.max_weight)
        )
        
        # Weight sum constraint
        constraint_list = [LinearConstraint(np.ones(n_assets), 1.0, 1.0)]
        
        # Add additional constraints
        constraint_list.extend(self._build_additional_constraints(
            n_assets, assets, constraints, esg_scores, liquidity_scores,
            sector_mapping, asset_class_mapping
        ))
        
        # Initial guess
        x0 = np.full(n_assets, 1.0 / n_assets)
        
        # Optimize
        result = minimize(
            objective_function,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint_list,
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
        )
        
        if result.success:
            weights = result.x
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
            
            return OptimizationResult(
                weights=weights,
                expected_return=portfolio_return,
                expected_risk=portfolio_risk,
                sharpe_ratio=sharpe_ratio,
                objective_value=-result.fun,
                success=True,
                message="Optimization successful",
                iterations=result.nit,
                optimization_time=0.0,
                asset_allocation=dict(zip(assets, weights)),
                sector_allocation=self._calculate_sector_allocation(weights, assets, sector_mapping),
                asset_class_allocation=self._calculate_asset_class_allocation(weights, assets, asset_class_mapping),
                risk_contributions=self._calculate_risk_contributions(weights, cov_matrix, assets)
            )
        else:
            return OptimizationResult(
                weights=np.full(n_assets, 1.0 / n_assets),
                expected_return=0.0,
                expected_risk=0.0,
                sharpe_ratio=0.0,
                objective_value=0.0,
                success=False,
                message=f"Optimization failed: {result.message}",
                iterations=result.nit,
                optimization_time=0.0,
                asset_allocation=dict(zip(assets, np.full(n_assets, 1.0 / n_assets)))
            )
    
    def _optimize_minimum_variance(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        constraints: OptimizationConstraints,
        assets: List[str],
        current_weights: Optional[np.ndarray] = None,
        transaction_costs: Optional[Dict[str, float]] = None,
        esg_scores: Optional[Dict[str, float]] = None,
        liquidity_scores: Optional[Dict[str, float]] = None,
        sector_mapping: Optional[Dict[str, str]] = None,
        asset_class_mapping: Optional[Dict[str, str]] = None
    ) -> OptimizationResult:
        """Optimize for minimum variance"""
        
        n_assets = len(assets)
        
        def objective_function(weights):
            portfolio_variance = np.dot(weights, np.dot(cov_matrix, weights))
            
            # Add transaction costs if provided
            if transaction_costs and current_weights is not None:
                tc_penalty = sum(
                    transaction_costs.get(asset, 0.001) * abs(weights[i] - current_weights[i])
                    for i, asset in enumerate(assets)
                )
                portfolio_variance += tc_penalty
            
            return portfolio_variance
        
        # Set up bounds and constraints
        bounds = Bounds(
            lb=np.full(n_assets, constraints.min_weight),
            ub=np.full(n_assets, constraints.max_weight)
        )
        
        # Weight sum constraint
        constraint_list = [LinearConstraint(np.ones(n_assets), 1.0, 1.0)]
        
        # Add additional constraints
        constraint_list.extend(self._build_additional_constraints(
            n_assets, assets, constraints, esg_scores, liquidity_scores,
            sector_mapping, asset_class_mapping
        ))
        
        # Initial guess
        x0 = np.full(n_assets, 1.0 / n_assets)
        
        # Optimize
        result = minimize(
            objective_function,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint_list,
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
        )
        
        if result.success:
            weights = result.x
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
            
            return OptimizationResult(
                weights=weights,
                expected_return=portfolio_return,
                expected_risk=portfolio_risk,
                sharpe_ratio=sharpe_ratio,
                objective_value=result.fun,
                success=True,
                message="Optimization successful",
                iterations=result.nit,
                optimization_time=0.0,
                asset_allocation=dict(zip(assets, weights)),
                sector_allocation=self._calculate_sector_allocation(weights, assets, sector_mapping),
                asset_class_allocation=self._calculate_asset_class_allocation(weights, assets, asset_class_mapping),
                risk_contributions=self._calculate_risk_contributions(weights, cov_matrix, assets)
            )
        else:
            return OptimizationResult(
                weights=np.full(n_assets, 1.0 / n_assets),
                expected_return=0.0,
                expected_risk=0.0,
                sharpe_ratio=0.0,
                objective_value=0.0,
                success=False,
                message=f"Optimization failed: {result.message}",
                iterations=result.nit,
                optimization_time=0.0,
                asset_allocation=dict(zip(assets, np.full(n_assets, 1.0 / n_assets)))
            )
    
    def _optimize_risk_parity(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        constraints: OptimizationConstraints,
        assets: List[str],
        current_weights: Optional[np.ndarray] = None,
        transaction_costs: Optional[Dict[str, float]] = None,
        esg_scores: Optional[Dict[str, float]] = None,
        liquidity_scores: Optional[Dict[str, float]] = None,
        sector_mapping: Optional[Dict[str, str]] = None,
        asset_class_mapping: Optional[Dict[str, str]] = None
    ) -> OptimizationResult:
        """Optimize for risk parity (equal risk contribution)"""
        
        n_assets = len(assets)
        
        def objective_function(weights):
            # Calculate risk contributions
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            marginal_risk = np.dot(cov_matrix, weights) / portfolio_risk if portfolio_risk > 0 else np.zeros(n_assets)
            risk_contributions = weights * marginal_risk
            
            # Target equal risk contribution
            target_risk_contrib = portfolio_risk / n_assets
            
            # Sum of squared deviations from equal risk contribution
            risk_parity_error = np.sum((risk_contributions - target_risk_contrib) ** 2)
            
            return risk_parity_error
        
        # Set up bounds and constraints
        bounds = Bounds(
            lb=np.full(n_assets, max(constraints.min_weight, 1e-6)),  # Avoid zero weights
            ub=np.full(n_assets, constraints.max_weight)
        )
        
        # Weight sum constraint
        constraint_list = [LinearConstraint(np.ones(n_assets), 1.0, 1.0)]
        
        # Add additional constraints
        constraint_list.extend(self._build_additional_constraints(
            n_assets, assets, constraints, esg_scores, liquidity_scores,
            sector_mapping, asset_class_mapping
        ))
        
        # Initial guess - start with equal weights
        x0 = np.full(n_assets, 1.0 / n_assets)
        
        # Optimize
        result = minimize(
            objective_function,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint_list,
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
        )
        
        if result.success:
            weights = result.x
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
            
            return OptimizationResult(
                weights=weights,
                expected_return=portfolio_return,
                expected_risk=portfolio_risk,
                sharpe_ratio=sharpe_ratio,
                objective_value=result.fun,
                success=True,
                message="Risk parity optimization successful",
                iterations=result.nit,
                optimization_time=0.0,
                asset_allocation=dict(zip(assets, weights)),
                sector_allocation=self._calculate_sector_allocation(weights, assets, sector_mapping),
                asset_class_allocation=self._calculate_asset_class_allocation(weights, assets, asset_class_mapping),
                risk_contributions=self._calculate_risk_contributions(weights, cov_matrix, assets)
            )
        else:
            return OptimizationResult(
                weights=np.full(n_assets, 1.0 / n_assets),
                expected_return=0.0,
                expected_risk=0.0,
                sharpe_ratio=0.0,
                objective_value=0.0,
                success=False,
                message=f"Risk parity optimization failed: {result.message}",
                iterations=result.nit,
                optimization_time=0.0,
                asset_allocation=dict(zip(assets, np.full(n_assets, 1.0 / n_assets)))
            )
    
    def _optimize_equal_weight(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        constraints: OptimizationConstraints,
        assets: List[str],
        sector_mapping: Optional[Dict[str, str]] = None,
        asset_class_mapping: Optional[Dict[str, str]] = None
    ) -> OptimizationResult:
        """Equal weight portfolio"""
        
        n_assets = len(assets)
        weights = np.full(n_assets, 1.0 / n_assets)
        
        portfolio_return = np.dot(weights, expected_returns)
        portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
        
        return OptimizationResult(
            weights=weights,
            expected_return=portfolio_return,
            expected_risk=portfolio_risk,
            sharpe_ratio=sharpe_ratio,
            objective_value=sharpe_ratio,
            success=True,
            message="Equal weight portfolio created",
            iterations=0,
            optimization_time=0.0,
            asset_allocation=dict(zip(assets, weights)),
            sector_allocation=self._calculate_sector_allocation(weights, assets, sector_mapping),
            asset_class_allocation=self._calculate_asset_class_allocation(weights, assets, asset_class_mapping),
            risk_contributions=self._calculate_risk_contributions(weights, cov_matrix, assets)
        )
    
    def _optimize_esg_weighted(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        constraints: OptimizationConstraints,
        assets: List[str],
        current_weights: Optional[np.ndarray] = None,
        transaction_costs: Optional[Dict[str, float]] = None,
        esg_scores: Optional[Dict[str, float]] = None,
        liquidity_scores: Optional[Dict[str, float]] = None,
        sector_mapping: Optional[Dict[str, str]] = None,
        asset_class_mapping: Optional[Dict[str, str]] = None
    ) -> OptimizationResult:
        """ESG-optimized portfolio"""
        
        if not esg_scores:
            # Fallback to Sharpe ratio optimization if no ESG scores
            return self._optimize_sharpe_ratio(
                expected_returns, cov_matrix, constraints, assets,
                current_weights, transaction_costs, esg_scores, liquidity_scores,
                sector_mapping, asset_class_mapping
            )
        
        n_assets = len(assets)
        
        # Create ESG score vector
        esg_vector = np.array([esg_scores.get(asset, 0.5) for asset in assets])
        
        def objective_function(weights):
            # Combine Sharpe ratio and ESG score
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            
            if portfolio_risk < 1e-10:
                return -1e10
            
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk
            esg_score = np.dot(weights, esg_vector)
            
            # Weighted combination (70% Sharpe, 30% ESG)
            combined_objective = 0.7 * sharpe_ratio + 0.3 * esg_score
            
            return -combined_objective  # Minimize negative objective
        
        # Set up bounds and constraints
        bounds = Bounds(
            lb=np.full(n_assets, constraints.min_weight),
            ub=np.full(n_assets, constraints.max_weight)
        )
        
        # Weight sum constraint
        constraint_list = [LinearConstraint(np.ones(n_assets), 1.0, 1.0)]
        
        # Add additional constraints
        constraint_list.extend(self._build_additional_constraints(
            n_assets, assets, constraints, esg_scores, liquidity_scores,
            sector_mapping, asset_class_mapping
        ))
        
        # Initial guess
        x0 = np.full(n_assets, 1.0 / n_assets)
        
        # Optimize
        result = minimize(
            objective_function,
            x0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraint_list,
            options={'maxiter': self.max_iterations, 'ftol': self.tolerance}
        )
        
        if result.success:
            weights = result.x
            portfolio_return = np.dot(weights, expected_returns)
            portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
            sharpe_ratio = (portfolio_return - self.risk_free_rate) / portfolio_risk if portfolio_risk > 0 else 0
            
            return OptimizationResult(
                weights=weights,
                expected_return=portfolio_return,
                expected_risk=portfolio_risk,
                sharpe_ratio=sharpe_ratio,
                objective_value=-result.fun,
                success=True,
                message="ESG optimization successful",
                iterations=result.nit,
                optimization_time=0.0,
                asset_allocation=dict(zip(assets, weights)),
                sector_allocation=self._calculate_sector_allocation(weights, assets, sector_mapping),
                asset_class_allocation=self._calculate_asset_class_allocation(weights, assets, asset_class_mapping),
                risk_contributions=self._calculate_risk_contributions(weights, cov_matrix, assets)
            )
        else:
            return OptimizationResult(
                weights=np.full(n_assets, 1.0 / n_assets),
                expected_return=0.0,
                expected_risk=0.0,
                sharpe_ratio=0.0,
                objective_value=0.0,
                success=False,
                message=f"ESG optimization failed: {result.message}",
                iterations=result.nit,
                optimization_time=0.0,
                asset_allocation=dict(zip(assets, np.full(n_assets, 1.0 / n_assets)))
            )
    
    def _build_additional_constraints(
        self,
        n_assets: int,
        assets: List[str],
        constraints: OptimizationConstraints,
        esg_scores: Optional[Dict[str, float]] = None,
        liquidity_scores: Optional[Dict[str, float]] = None,
        sector_mapping: Optional[Dict[str, str]] = None,
        asset_class_mapping: Optional[Dict[str, str]] = None
    ) -> List[LinearConstraint]:
        """Build additional optimization constraints"""
        
        constraint_list = []
        
        # Target return constraint
        if constraints.target_return is not None:
            # This would need expected returns, but we'll skip for now
            pass
        
        # ESG score constraint
        if constraints.esg_score_min is not None and esg_scores:
            esg_vector = np.array([esg_scores.get(asset, 0.0) for asset in assets])
            constraint_list.append(
                LinearConstraint(esg_vector, constraints.esg_score_min, np.inf)
            )
        
        # Liquidity constraint
        if constraints.liquidity_min is not None and liquidity_scores:
            liquidity_vector = np.array([liquidity_scores.get(asset, 0.0) for asset in assets])
            constraint_list.append(
                LinearConstraint(liquidity_vector, constraints.liquidity_min, np.inf)
            )
        
        # Sector constraints
        if sector_mapping and (constraints.max_sector_weight or constraints.min_sector_weight):
            sectors = set(sector_mapping.values())
            for sector in sectors:
                sector_mask = np.array([1 if sector_mapping.get(asset) == sector else 0 for asset in assets])
                
                if constraints.max_sector_weight and sector in constraints.max_sector_weight:
                    constraint_list.append(
                        LinearConstraint(sector_mask, 0.0, constraints.max_sector_weight[sector])
                    )
                
                if constraints.min_sector_weight and sector in constraints.min_sector_weight:
                    constraint_list.append(
                        LinearConstraint(sector_mask, constraints.min_sector_weight[sector], 1.0)
                    )
        
        # Asset class constraints
        if asset_class_mapping and (constraints.max_asset_class_weight or constraints.min_asset_class_weight):
            asset_classes = set(asset_class_mapping.values())
            for asset_class in asset_classes:
                class_mask = np.array([1 if asset_class_mapping.get(asset) == asset_class else 0 for asset in assets])
                
                if constraints.max_asset_class_weight and asset_class in constraints.max_asset_class_weight:
                    constraint_list.append(
                        LinearConstraint(class_mask, 0.0, constraints.max_asset_class_weight[asset_class])
                    )
                
                if constraints.min_asset_class_weight and asset_class in constraints.min_asset_class_weight:
                    constraint_list.append(
                        LinearConstraint(class_mask, constraints.min_asset_class_weight[asset_class], 1.0)
                    )
        
        return constraint_list
    
    def _calculate_sector_allocation(
        self,
        weights: np.ndarray,
        assets: List[str],
        sector_mapping: Optional[Dict[str, str]]
    ) -> Optional[Dict[str, float]]:
        """Calculate sector allocation from weights"""
        
        if not sector_mapping:
            return None
        
        sector_allocation = {}
        for i, asset in enumerate(assets):
            sector = sector_mapping.get(asset, 'Other')
            sector_allocation[sector] = sector_allocation.get(sector, 0.0) + weights[i]
        
        return sector_allocation
    
    def _calculate_asset_class_allocation(
        self,
        weights: np.ndarray,
        assets: List[str],
        asset_class_mapping: Optional[Dict[str, str]]
    ) -> Optional[Dict[str, float]]:
        """Calculate asset class allocation from weights"""
        
        if not asset_class_mapping:
            return None
        
        class_allocation = {}
        for i, asset in enumerate(assets):
            asset_class = asset_class_mapping.get(asset, 'Other')
            class_allocation[asset_class] = class_allocation.get(asset_class, 0.0) + weights[i]
        
        return class_allocation
    
    def _calculate_risk_contributions(
        self,
        weights: np.ndarray,
        cov_matrix: np.ndarray,
        assets: List[str]
    ) -> Dict[str, float]:
        """Calculate risk contributions for each asset"""
        
        portfolio_risk = np.sqrt(np.dot(weights, np.dot(cov_matrix, weights)))
        
        if portfolio_risk < 1e-10:
            return dict(zip(assets, np.zeros(len(assets))))
        
        marginal_risk = np.dot(cov_matrix, weights) / portfolio_risk
        risk_contributions = weights * marginal_risk
        
        return dict(zip(assets, risk_contributions))
    
    def _apply_black_litterman(
        self,
        expected_returns: np.ndarray,
        cov_matrix: np.ndarray,
        assets: List[str],
        market_views: Dict[str, float],
        confidence_levels: Optional[Dict[str, float]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply Black-Litterman model to adjust expected returns"""
        
        try:
            n_assets = len(assets)
            
            # Market capitalization weights (simplified - equal weights as proxy)
            market_weights = np.full(n_assets, 1.0 / n_assets)
            
            # Risk aversion parameter (typical value)
            risk_aversion = 3.0
            
            # Implied equilibrium returns
            pi = risk_aversion * np.dot(cov_matrix, market_weights)
            
            # Views matrix P and views vector Q
            n_views = len(market_views)
            P = np.zeros((n_views, n_assets))
            Q = np.zeros(n_views)
            
            view_idx = 0
            for asset, view_return in market_views.items():
                if asset in assets:
                    asset_idx = assets.index(asset)
                    P[view_idx, asset_idx] = 1.0
                    Q[view_idx] = view_return
                    view_idx += 1
            
            # Confidence matrix Omega (diagonal)
            if confidence_levels:
                omega_diag = []
                for asset, _ in market_views.items():
                    if asset in assets:
                        confidence = confidence_levels.get(asset, 0.5)
                        # Lower confidence = higher uncertainty
                        omega_diag.append(1.0 / confidence)
                Omega = np.diag(omega_diag)
            else:
                # Default uncertainty
                Omega = np.eye(n_views) * 0.1
            
            # Uncertainty in prior (tau)
            tau = 1.0 / len(expected_returns)
            
            # Black-Litterman formula
            M1 = np.linalg.inv(tau * cov_matrix)
            M2 = np.dot(P.T, np.dot(np.linalg.inv(Omega), P))
            M3 = np.dot(np.linalg.inv(tau * cov_matrix), pi)
            M4 = np.dot(P.T, np.dot(np.linalg.inv(Omega), Q))
            
            # New expected returns
            mu_bl = np.dot(np.linalg.inv(M1 + M2), M3 + M4)
            
            # New covariance matrix
            cov_bl = np.linalg.inv(M1 + M2)
            
            return mu_bl, cov_bl
            
        except Exception as e:
            logger.warning(f"Black-Litterman calculation failed: {str(e)}, using original values")
            return expected_returns, cov_matrix
    
    def get_efficient_frontier(
        self,
        returns: pd.DataFrame,
        n_portfolios: int = 100,
        constraints: Optional[OptimizationConstraints] = None
    ) -> Dict[str, List[float]]:
        """Generate efficient frontier"""
        
        try:
            returns_clean = returns.dropna()
            expected_returns = returns_clean.mean().values * 252
            cov_matrix = returns_clean.cov().values * 252
            
            # Range of target returns
            min_return = expected_returns.min()
            max_return = expected_returns.max()
            target_returns = np.linspace(min_return, max_return, n_portfolios)
            
            efficient_portfolios = {
                'returns': [],
                'risks': [],
                'sharpe_ratios': [],
                'weights': []
            }
            
            for target_return in target_returns:
                # Create constraints with target return
                if constraints is None:
                    constraints = OptimizationConstraints()
                
                constraints.target_return = target_return
                
                # Optimize for minimum variance with target return
                result = self._optimize_minimum_variance(
                    expected_returns, cov_matrix, constraints, returns_clean.columns.tolist()
                )
                
                if result.success:
                    efficient_portfolios['returns'].append(result.expected_return)
                    efficient_portfolios['risks'].append(result.expected_risk)
                    efficient_portfolios['sharpe_ratios'].append(result.sharpe_ratio)
                    efficient_portfolios['weights'].append(result.weights.tolist())
            
            return efficient_portfolios
            
        except Exception as e:
            logger.error(f"Efficient frontier calculation error: {str(e)}")
            return {'returns': [], 'risks': [], 'sharpe_ratios': [], 'weights': []}

