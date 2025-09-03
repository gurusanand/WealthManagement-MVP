import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)

class InvestmentObjective(Enum):
    """Investment objectives"""
    CAPITAL_PRESERVATION = "capital_preservation"
    INCOME_GENERATION = "income_generation"
    BALANCED_GROWTH = "balanced_growth"
    CAPITAL_APPRECIATION = "capital_appreciation"
    AGGRESSIVE_GROWTH = "aggressive_growth"
    SPECULATION = "speculation"
    RETIREMENT_PLANNING = "retirement_planning"
    EDUCATION_FUNDING = "education_funding"
    TAX_OPTIMIZATION = "tax_optimization"

class RiskTolerance(Enum):
    """Risk tolerance levels"""
    VERY_CONSERVATIVE = "very_conservative"
    CONSERVATIVE = "conservative"
    MODERATE_CONSERVATIVE = "moderate_conservative"
    MODERATE = "moderate"
    MODERATE_AGGRESSIVE = "moderate_aggressive"
    AGGRESSIVE = "aggressive"
    VERY_AGGRESSIVE = "very_aggressive"

class InvestmentExperience(Enum):
    """Investment experience levels"""
    NONE = "none"
    LIMITED = "limited"
    MODERATE = "moderate"
    EXTENSIVE = "extensive"
    PROFESSIONAL = "professional"

class LiquidityNeed(Enum):
    """Liquidity requirement levels"""
    IMMEDIATE = "immediate"
    SHORT_TERM = "short_term"  # < 1 year
    MEDIUM_TERM = "medium_term"  # 1-5 years
    LONG_TERM = "long_term"  # > 5 years
    PERMANENT = "permanent"  # No liquidity needed

class InvestmentSuitability(Enum):
    """Investment suitability levels"""
    HIGHLY_SUITABLE = "highly_suitable"
    SUITABLE = "suitable"
    MARGINALLY_SUITABLE = "marginally_suitable"
    UNSUITABLE = "unsuitable"
    PROHIBITED = "prohibited"

@dataclass
class ClientProfile:
    """Comprehensive client profile for suitability assessment"""
    client_id: str
    age: int
    annual_income: float
    net_worth: float
    liquid_net_worth: float
    investment_experience: InvestmentExperience
    risk_tolerance: RiskTolerance
    investment_objectives: List[InvestmentObjective]
    time_horizon: int  # years
    liquidity_needs: LiquidityNeed
    tax_bracket: float
    employment_status: str
    dependents: int
    debt_obligations: float
    emergency_fund_months: int
    investment_knowledge_score: float  # 0-1 scale
    previous_investment_losses: bool
    regulatory_restrictions: List[str]
    esg_preferences: bool
    concentration_limits: Dict[str, float]

@dataclass
class InvestmentProduct:
    """Investment product characteristics"""
    product_id: str
    product_name: str
    product_type: str
    asset_class: str
    risk_level: int  # 1-10 scale
    minimum_investment: float
    liquidity_timeframe: int  # days
    complexity_score: float  # 0-1 scale
    volatility: float
    expected_return: float
    fees_percentage: float
    tax_efficiency: float  # 0-1 scale
    esg_rating: Optional[float]
    regulatory_requirements: List[str]
    target_investor_profile: Dict[str, Any]
    concentration_category: str
    leverage_factor: float
    currency_exposure: List[str]

@dataclass
class SuitabilityResult:
    """Suitability assessment result"""
    client_id: str
    product_id: str
    overall_suitability: InvestmentSuitability
    suitability_score: float  # 0-1 scale
    assessment_date: datetime
    risk_alignment_score: float
    objective_alignment_score: float
    experience_alignment_score: float
    liquidity_alignment_score: float
    financial_capacity_score: float
    regulatory_compliance_score: float
    concentration_impact_score: float
    detailed_analysis: Dict[str, Any]
    recommendations: List[str]
    warnings: List[str]
    required_disclosures: List[str]
    approval_required: bool
    approved_by: Optional[str] = None
    approval_date: Optional[datetime] = None
    review_date: datetime = None
    conditions: List[str] = None

class SuitabilityEngine:
    """
    Comprehensive Investment Suitability Assessment Engine
    
    Features:
    - Multi-dimensional suitability analysis
    - Risk tolerance alignment assessment
    - Investment objective matching
    - Financial capacity evaluation
    - Regulatory compliance checking
    - Concentration risk analysis
    - Experience-based recommendations
    - Automated approval workflows
    """
    
    def __init__(self):
        # Suitability scoring weights
        self.suitability_weights = {
            'risk_alignment': 0.25,
            'objective_alignment': 0.20,
            'experience_alignment': 0.15,
            'liquidity_alignment': 0.15,
            'financial_capacity': 0.15,
            'regulatory_compliance': 0.10
        }
        
        # Risk tolerance mappings
        self.risk_tolerance_scores = {
            RiskTolerance.VERY_CONSERVATIVE: 1,
            RiskTolerance.CONSERVATIVE: 2,
            RiskTolerance.MODERATE_CONSERVATIVE: 3,
            RiskTolerance.MODERATE: 4,
            RiskTolerance.MODERATE_AGGRESSIVE: 5,
            RiskTolerance.AGGRESSIVE: 6,
            RiskTolerance.VERY_AGGRESSIVE: 7
        }
        
        # Experience level scores
        self.experience_scores = {
            InvestmentExperience.NONE: 0,
            InvestmentExperience.LIMITED: 1,
            InvestmentExperience.MODERATE: 2,
            InvestmentExperience.EXTENSIVE: 3,
            InvestmentExperience.PROFESSIONAL: 4
        }
        
        # Liquidity timeframes (in days)
        self.liquidity_timeframes = {
            LiquidityNeed.IMMEDIATE: 1,
            LiquidityNeed.SHORT_TERM: 365,
            LiquidityNeed.MEDIUM_TERM: 1825,  # 5 years
            LiquidityNeed.LONG_TERM: 3650,   # 10 years
            LiquidityNeed.PERMANENT: 999999
        }
        
        # Regulatory requirements database
        self.regulatory_requirements = self._initialize_regulatory_requirements()
        
        # Product complexity thresholds
        self.complexity_thresholds = {
            'simple': 0.3,
            'moderate': 0.6,
            'complex': 0.8,
            'very_complex': 1.0
        }
    
    def assess_suitability(
        self,
        client_profile: ClientProfile,
        investment_product: InvestmentProduct,
        proposed_allocation: float = 0.0,
        existing_portfolio: Optional[Dict[str, float]] = None
    ) -> SuitabilityResult:
        """
        Perform comprehensive suitability assessment
        
        Args:
            client_profile: Client's complete profile
            investment_product: Investment product details
            proposed_allocation: Proposed allocation percentage
            existing_portfolio: Client's existing portfolio
            
        Returns:
            SuitabilityResult with detailed analysis
        """
        try:
            assessment_date = datetime.now()
            
            # Risk alignment assessment
            risk_alignment_score = self._assess_risk_alignment(
                client_profile, investment_product
            )
            
            # Investment objective alignment
            objective_alignment_score = self._assess_objective_alignment(
                client_profile, investment_product
            )
            
            # Experience alignment assessment
            experience_alignment_score = self._assess_experience_alignment(
                client_profile, investment_product
            )
            
            # Liquidity alignment assessment
            liquidity_alignment_score = self._assess_liquidity_alignment(
                client_profile, investment_product
            )
            
            # Financial capacity assessment
            financial_capacity_score = self._assess_financial_capacity(
                client_profile, investment_product, proposed_allocation
            )
            
            # Regulatory compliance assessment
            regulatory_compliance_score = self._assess_regulatory_compliance(
                client_profile, investment_product
            )
            
            # Concentration impact assessment
            concentration_impact_score = self._assess_concentration_impact(
                client_profile, investment_product, proposed_allocation, existing_portfolio
            )
            
            # Calculate overall suitability score
            overall_score = self._calculate_overall_suitability_score(
                risk_alignment_score, objective_alignment_score,
                experience_alignment_score, liquidity_alignment_score,
                financial_capacity_score, regulatory_compliance_score
            )
            
            # Determine suitability level
            overall_suitability = self._determine_suitability_level(overall_score)
            
            # Generate detailed analysis
            detailed_analysis = self._generate_detailed_analysis(
                client_profile, investment_product, {
                    'risk_alignment': risk_alignment_score,
                    'objective_alignment': objective_alignment_score,
                    'experience_alignment': experience_alignment_score,
                    'liquidity_alignment': liquidity_alignment_score,
                    'financial_capacity': financial_capacity_score,
                    'regulatory_compliance': regulatory_compliance_score,
                    'concentration_impact': concentration_impact_score
                }
            )
            
            # Generate recommendations
            recommendations = self._generate_recommendations(
                client_profile, investment_product, overall_score, detailed_analysis
            )
            
            # Generate warnings
            warnings = self._generate_warnings(
                client_profile, investment_product, detailed_analysis
            )
            
            # Required disclosures
            required_disclosures = self._generate_required_disclosures(
                client_profile, investment_product
            )
            
            # Check if approval is required
            approval_required = self._requires_approval(
                overall_suitability, detailed_analysis
            )
            
            # Calculate review date
            review_date = self._calculate_review_date(
                client_profile, investment_product, overall_suitability
            )
            
            # Generate conditions
            conditions = self._generate_conditions(
                client_profile, investment_product, overall_suitability
            )
            
            return SuitabilityResult(
                client_id=client_profile.client_id,
                product_id=investment_product.product_id,
                overall_suitability=overall_suitability,
                suitability_score=overall_score,
                assessment_date=assessment_date,
                risk_alignment_score=risk_alignment_score,
                objective_alignment_score=objective_alignment_score,
                experience_alignment_score=experience_alignment_score,
                liquidity_alignment_score=liquidity_alignment_score,
                financial_capacity_score=financial_capacity_score,
                regulatory_compliance_score=regulatory_compliance_score,
                concentration_impact_score=concentration_impact_score,
                detailed_analysis=detailed_analysis,
                recommendations=recommendations,
                warnings=warnings,
                required_disclosures=required_disclosures,
                approval_required=approval_required,
                review_date=review_date,
                conditions=conditions
            )
            
        except Exception as e:
            logger.error(f"Suitability assessment error: {str(e)}")
            return SuitabilityResult(
                client_id=client_profile.client_id,
                product_id=investment_product.product_id,
                overall_suitability=InvestmentSuitability.UNSUITABLE,
                suitability_score=0.0,
                assessment_date=datetime.now(),
                risk_alignment_score=0.0,
                objective_alignment_score=0.0,
                experience_alignment_score=0.0,
                liquidity_alignment_score=0.0,
                financial_capacity_score=0.0,
                regulatory_compliance_score=0.0,
                concentration_impact_score=0.0,
                detailed_analysis={},
                recommendations=["Manual review required due to assessment error"],
                warnings=["Assessment error occurred - manual review required"],
                required_disclosures=[],
                approval_required=True,
                review_date=datetime.now() + timedelta(days=30)
            )
    
    def assess_portfolio_suitability(
        self,
        client_profile: ClientProfile,
        proposed_portfolio: Dict[str, Tuple[InvestmentProduct, float]],
        existing_portfolio: Optional[Dict[str, float]] = None
    ) -> Dict[str, SuitabilityResult]:
        """
        Assess suitability of entire portfolio
        
        Args:
            client_profile: Client's profile
            proposed_portfolio: Dictionary of {product_id: (product, allocation)}
            existing_portfolio: Existing portfolio allocations
            
        Returns:
            Dictionary of suitability results for each product
        """
        try:
            portfolio_results = {}
            
            for product_id, (product, allocation) in proposed_portfolio.items():
                result = self.assess_suitability(
                    client_profile, product, allocation, existing_portfolio
                )
                portfolio_results[product_id] = result
            
            return portfolio_results
            
        except Exception as e:
            logger.error(f"Portfolio suitability assessment error: {str(e)}")
            return {}
    
    def _assess_risk_alignment(
        self,
        client_profile: ClientProfile,
        investment_product: InvestmentProduct
    ) -> float:
        """Assess alignment between client risk tolerance and product risk"""
        
        try:
            client_risk_score = self.risk_tolerance_scores[client_profile.risk_tolerance]
            product_risk_level = investment_product.risk_level
            
            # Normalize product risk to 1-7 scale to match client risk tolerance
            normalized_product_risk = min(max(product_risk_level * 0.7, 1), 7)
            
            # Calculate alignment score
            risk_difference = abs(client_risk_score - normalized_product_risk)
            
            # Perfect alignment = 1.0, maximum misalignment = 0.0
            alignment_score = max(0.0, 1.0 - (risk_difference / 6.0))
            
            # Apply penalties for conservative clients with high-risk products
            if (client_risk_score <= 3 and normalized_product_risk >= 6):
                alignment_score *= 0.5  # Severe penalty
            elif (client_risk_score <= 2 and normalized_product_risk >= 5):
                alignment_score *= 0.3  # Very severe penalty
            
            # Bonus for perfect matches
            if risk_difference <= 1:
                alignment_score = min(1.0, alignment_score * 1.1)
            
            return alignment_score
            
        except Exception as e:
            logger.error(f"Risk alignment assessment error: {str(e)}")
            return 0.0
    
    def _assess_objective_alignment(
        self,
        client_profile: ClientProfile,
        investment_product: InvestmentProduct
    ) -> float:
        """Assess alignment between client objectives and product characteristics"""
        
        try:
            alignment_scores = []
            
            for objective in client_profile.investment_objectives:
                objective_score = self._calculate_objective_product_match(
                    objective, investment_product
                )
                alignment_scores.append(objective_score)
            
            # Use the best matching objective
            best_alignment = max(alignment_scores) if alignment_scores else 0.0
            
            # Consider time horizon alignment
            time_horizon_score = self._assess_time_horizon_alignment(
                client_profile.time_horizon, investment_product
            )
            
            # Weighted combination
            overall_alignment = 0.7 * best_alignment + 0.3 * time_horizon_score
            
            return min(max(overall_alignment, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Objective alignment assessment error: {str(e)}")
            return 0.0
    
    def _assess_experience_alignment(
        self,
        client_profile: ClientProfile,
        investment_product: InvestmentProduct
    ) -> float:
        """Assess alignment between client experience and product complexity"""
        
        try:
            client_experience_score = self.experience_scores[client_profile.investment_experience]
            product_complexity = investment_product.complexity_score
            
            # Determine required experience level for product
            if product_complexity <= self.complexity_thresholds['simple']:
                required_experience = 0  # No experience required
            elif product_complexity <= self.complexity_thresholds['moderate']:
                required_experience = 1  # Limited experience required
            elif product_complexity <= self.complexity_thresholds['complex']:
                required_experience = 2  # Moderate experience required
            else:
                required_experience = 3  # Extensive experience required
            
            # Calculate alignment
            if client_experience_score >= required_experience:
                # Client has sufficient experience
                alignment_score = 1.0
                
                # Bonus for overqualified clients with simple products
                if (client_experience_score >= 3 and 
                    product_complexity <= self.complexity_thresholds['simple']):
                    alignment_score = 1.0  # No penalty for overqualification
                    
            else:
                # Client lacks sufficient experience
                experience_gap = required_experience - client_experience_score
                alignment_score = max(0.0, 1.0 - (experience_gap * 0.3))
                
                # Severe penalty for complex products with inexperienced clients
                if (client_experience_score == 0 and 
                    product_complexity >= self.complexity_thresholds['complex']):
                    alignment_score = 0.0
            
            # Consider investment knowledge score
            knowledge_factor = client_profile.investment_knowledge_score
            final_score = alignment_score * (0.7 + 0.3 * knowledge_factor)
            
            return min(max(final_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Experience alignment assessment error: {str(e)}")
            return 0.0
    
    def _assess_liquidity_alignment(
        self,
        client_profile: ClientProfile,
        investment_product: InvestmentProduct
    ) -> float:
        """Assess alignment between client liquidity needs and product liquidity"""
        
        try:
            client_liquidity_days = self.liquidity_timeframes[client_profile.liquidity_needs]
            product_liquidity_days = investment_product.liquidity_timeframe
            
            # Perfect alignment when product liquidity meets or exceeds client needs
            if product_liquidity_days <= client_liquidity_days:
                # Product is more liquid than needed - good alignment
                alignment_score = 1.0
            else:
                # Product is less liquid than needed - calculate penalty
                liquidity_gap = product_liquidity_days - client_liquidity_days
                
                # Normalize gap (assume maximum acceptable gap is 2 years = 730 days)
                normalized_gap = min(liquidity_gap / 730, 2.0)
                alignment_score = max(0.0, 1.0 - (normalized_gap * 0.5))
                
                # Severe penalty for immediate liquidity needs with illiquid products
                if (client_profile.liquidity_needs == LiquidityNeed.IMMEDIATE and
                    product_liquidity_days > 30):
                    alignment_score = 0.0
            
            # Consider emergency fund adequacy
            if client_profile.emergency_fund_months < 3:
                # Client lacks adequate emergency fund - penalize illiquid investments
                if product_liquidity_days > 90:
                    alignment_score *= 0.7
            
            return alignment_score
            
        except Exception as e:
            logger.error(f"Liquidity alignment assessment error: {str(e)}")
            return 0.0
    
    def _assess_financial_capacity(
        self,
        client_profile: ClientProfile,
        investment_product: InvestmentProduct,
        proposed_allocation: float
    ) -> float:
        """Assess client's financial capacity for the investment"""
        
        try:
            # Check minimum investment requirement
            min_investment = investment_product.minimum_investment
            proposed_amount = client_profile.liquid_net_worth * (proposed_allocation / 100)
            
            if proposed_amount < min_investment:
                return 0.0  # Cannot meet minimum investment
            
            # Assess affordability relative to income and net worth
            annual_income = client_profile.annual_income
            liquid_net_worth = client_profile.liquid_net_worth
            
            # Calculate affordability scores
            income_affordability = 1.0
            if annual_income > 0:
                income_ratio = proposed_amount / annual_income
                if income_ratio > 0.5:  # More than 50% of annual income
                    income_affordability = max(0.0, 1.0 - (income_ratio - 0.5))
            
            net_worth_affordability = 1.0
            if liquid_net_worth > 0:
                net_worth_ratio = proposed_amount / liquid_net_worth
                if net_worth_ratio > 0.3:  # More than 30% of liquid net worth
                    net_worth_affordability = max(0.0, 1.0 - (net_worth_ratio - 0.3) * 2)
            
            # Consider debt obligations
            debt_factor = 1.0
            if client_profile.debt_obligations > 0:
                debt_to_income = client_profile.debt_obligations / max(annual_income, 1)
                if debt_to_income > 0.4:  # High debt burden
                    debt_factor = max(0.5, 1.0 - (debt_to_income - 0.4))
            
            # Consider dependents
            dependent_factor = 1.0
            if client_profile.dependents > 0:
                # Reduce capacity based on number of dependents
                dependent_factor = max(0.7, 1.0 - (client_profile.dependents * 0.1))
            
            # Overall financial capacity
            capacity_score = (
                income_affordability * 0.3 +
                net_worth_affordability * 0.4 +
                debt_factor * 0.2 +
                dependent_factor * 0.1
            )
            
            return min(max(capacity_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Financial capacity assessment error: {str(e)}")
            return 0.0
    
    def _assess_regulatory_compliance(
        self,
        client_profile: ClientProfile,
        investment_product: InvestmentProduct
    ) -> float:
        """Assess regulatory compliance for the investment"""
        
        try:
            compliance_score = 1.0
            
            # Check client regulatory restrictions
            for restriction in client_profile.regulatory_restrictions:
                if restriction in investment_product.regulatory_requirements:
                    compliance_score = 0.0  # Hard restriction violation
                    break
            
            # Check product regulatory requirements
            for requirement in investment_product.regulatory_requirements:
                if not self._client_meets_requirement(client_profile, requirement):
                    compliance_score *= 0.5  # Partial compliance issue
            
            # Age-based restrictions
            if client_profile.age < 18:
                compliance_score = 0.0  # Minors cannot invest
            elif client_profile.age < 21 and investment_product.complexity_score > 0.7:
                compliance_score *= 0.7  # Young investors with complex products
            
            # Employment-based restrictions
            if (client_profile.employment_status == 'financial_services' and
                investment_product.product_type in ['individual_stocks', 'options']):
                compliance_score *= 0.8  # Industry restrictions
            
            return compliance_score
            
        except Exception as e:
            logger.error(f"Regulatory compliance assessment error: {str(e)}")
            return 0.0
    
    def _assess_concentration_impact(
        self,
        client_profile: ClientProfile,
        investment_product: InvestmentProduct,
        proposed_allocation: float,
        existing_portfolio: Optional[Dict[str, float]]
    ) -> float:
        """Assess concentration risk impact"""
        
        try:
            concentration_category = investment_product.concentration_category
            
            # Get concentration limit for this category
            concentration_limit = client_profile.concentration_limits.get(
                concentration_category, 0.2  # Default 20% limit
            )
            
            # Calculate current concentration
            current_concentration = 0.0
            if existing_portfolio:
                # This would require product category mapping in practice
                current_concentration = existing_portfolio.get(concentration_category, 0.0)
            
            # Calculate new concentration after proposed investment
            new_concentration = current_concentration + (proposed_allocation / 100)
            
            # Assess concentration impact
            if new_concentration <= concentration_limit:
                concentration_score = 1.0
            else:
                # Penalty for exceeding concentration limits
                excess_concentration = new_concentration - concentration_limit
                concentration_score = max(0.0, 1.0 - (excess_concentration * 2))
            
            # Additional penalty for very high concentrations
            if new_concentration > 0.5:  # More than 50% in any category
                concentration_score *= 0.5
            
            return concentration_score
            
        except Exception as e:
            logger.error(f"Concentration impact assessment error: {str(e)}")
            return 1.0
    
    def _calculate_overall_suitability_score(
        self,
        risk_alignment: float,
        objective_alignment: float,
        experience_alignment: float,
        liquidity_alignment: float,
        financial_capacity: float,
        regulatory_compliance: float
    ) -> float:
        """Calculate weighted overall suitability score"""
        
        try:
            overall_score = (
                self.suitability_weights['risk_alignment'] * risk_alignment +
                self.suitability_weights['objective_alignment'] * objective_alignment +
                self.suitability_weights['experience_alignment'] * experience_alignment +
                self.suitability_weights['liquidity_alignment'] * liquidity_alignment +
                self.suitability_weights['financial_capacity'] * financial_capacity +
                self.suitability_weights['regulatory_compliance'] * regulatory_compliance
            )
            
            return min(max(overall_score, 0.0), 1.0)
            
        except Exception as e:
            logger.error(f"Overall suitability score calculation error: {str(e)}")
            return 0.0
    
    def _determine_suitability_level(self, overall_score: float) -> InvestmentSuitability:
        """Determine suitability level based on overall score"""
        
        if overall_score >= 0.8:
            return InvestmentSuitability.HIGHLY_SUITABLE
        elif overall_score >= 0.6:
            return InvestmentSuitability.SUITABLE
        elif overall_score >= 0.4:
            return InvestmentSuitability.MARGINALLY_SUITABLE
        elif overall_score > 0.0:
            return InvestmentSuitability.UNSUITABLE
        else:
            return InvestmentSuitability.PROHIBITED
    
    def _calculate_objective_product_match(
        self,
        objective: InvestmentObjective,
        product: InvestmentProduct
    ) -> float:
        """Calculate how well a product matches an investment objective"""
        
        # Simplified objective matching logic
        objective_product_scores = {
            InvestmentObjective.CAPITAL_PRESERVATION: {
                'bonds': 0.9, 'money_market': 1.0, 'stocks': 0.3, 'alternatives': 0.2
            },
            InvestmentObjective.INCOME_GENERATION: {
                'bonds': 0.9, 'dividend_stocks': 0.8, 'reits': 0.9, 'stocks': 0.4
            },
            InvestmentObjective.BALANCED_GROWTH: {
                'balanced_funds': 1.0, 'stocks': 0.7, 'bonds': 0.6, 'etfs': 0.8
            },
            InvestmentObjective.CAPITAL_APPRECIATION: {
                'stocks': 0.9, 'growth_funds': 1.0, 'etfs': 0.8, 'bonds': 0.3
            },
            InvestmentObjective.AGGRESSIVE_GROWTH: {
                'growth_stocks': 1.0, 'small_cap': 0.9, 'emerging_markets': 0.8, 'alternatives': 0.7
            }
        }
        
        product_type = product.product_type.lower()
        
        if objective in objective_product_scores:
            return objective_product_scores[objective].get(product_type, 0.5)
        
        return 0.5  # Default neutral score
    
    def _assess_time_horizon_alignment(self, time_horizon: int, product: InvestmentProduct) -> float:
        """Assess alignment between time horizon and product characteristics"""
        
        # Simple time horizon alignment
        if time_horizon >= 10:  # Long-term
            return 1.0 if product.asset_class in ['stocks', 'alternatives'] else 0.7
        elif time_horizon >= 5:  # Medium-term
            return 1.0 if product.asset_class in ['balanced', 'bonds', 'stocks'] else 0.6
        else:  # Short-term
            return 1.0 if product.asset_class in ['money_market', 'bonds'] else 0.4
    
    def _client_meets_requirement(self, client_profile: ClientProfile, requirement: str) -> bool:
        """Check if client meets specific regulatory requirement"""
        
        # Simplified requirement checking
        requirement_checks = {
            'accredited_investor': client_profile.net_worth >= 1000000 or client_profile.annual_income >= 200000,
            'qualified_purchaser': client_profile.net_worth >= 5000000,
            'professional_investor': client_profile.investment_experience == InvestmentExperience.PROFESSIONAL,
            'high_net_worth': client_profile.net_worth >= 1000000
        }
        
        return requirement_checks.get(requirement, True)
    
    def _generate_detailed_analysis(
        self,
        client_profile: ClientProfile,
        product: InvestmentProduct,
        scores: Dict[str, float]
    ) -> Dict[str, Any]:
        """Generate detailed suitability analysis"""
        
        return {
            'client_summary': {
                'age': client_profile.age,
                'risk_tolerance': client_profile.risk_tolerance.value,
                'investment_experience': client_profile.investment_experience.value,
                'primary_objective': client_profile.investment_objectives[0].value if client_profile.investment_objectives else None,
                'time_horizon': client_profile.time_horizon,
                'liquidity_needs': client_profile.liquidity_needs.value
            },
            'product_summary': {
                'name': product.product_name,
                'type': product.product_type,
                'risk_level': product.risk_level,
                'complexity_score': product.complexity_score,
                'minimum_investment': product.minimum_investment,
                'liquidity_timeframe': product.liquidity_timeframe
            },
            'score_breakdown': scores,
            'key_factors': self._identify_key_factors(scores),
            'risk_factors': self._identify_risk_factors(client_profile, product, scores)
        }
    
    def _generate_recommendations(
        self,
        client_profile: ClientProfile,
        product: InvestmentProduct,
        overall_score: float,
        analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate suitability recommendations"""
        
        recommendations = []
        
        if overall_score >= 0.8:
            recommendations.append("This investment is highly suitable for the client's profile")
        elif overall_score >= 0.6:
            recommendations.append("This investment is suitable for the client with standard monitoring")
        elif overall_score >= 0.4:
            recommendations.append("This investment is marginally suitable - consider alternatives")
            recommendations.append("Enhanced monitoring and regular reviews recommended")
        else:
            recommendations.append("This investment is not suitable for the client")
            recommendations.append("Consider alternative investments that better match client profile")
        
        # Specific recommendations based on scores
        scores = analysis.get('score_breakdown', {})
        
        if scores.get('risk_alignment', 0) < 0.5:
            recommendations.append("Consider products with risk level more aligned to client tolerance")
        
        if scores.get('experience_alignment', 0) < 0.5:
            recommendations.append("Client may benefit from investment education before proceeding")
        
        if scores.get('liquidity_alignment', 0) < 0.5:
            recommendations.append("Ensure client has adequate liquid reserves before investing")
        
        return recommendations
    
    def _generate_warnings(
        self,
        client_profile: ClientProfile,
        product: InvestmentProduct,
        analysis: Dict[str, Any]
    ) -> List[str]:
        """Generate suitability warnings"""
        
        warnings = []
        scores = analysis.get('score_breakdown', {})
        
        if scores.get('regulatory_compliance', 1) < 1.0:
            warnings.append("Regulatory compliance issues identified - manual review required")
        
        if scores.get('financial_capacity', 1) < 0.5:
            warnings.append("Investment may represent excessive portion of client's assets")
        
        if product.complexity_score > 0.7 and client_profile.investment_experience in [InvestmentExperience.NONE, InvestmentExperience.LIMITED]:
            warnings.append("Complex product may not be appropriate for client's experience level")
        
        if client_profile.previous_investment_losses and product.risk_level >= 7:
            warnings.append("Client has history of investment losses - high-risk product requires careful consideration")
        
        return warnings
    
    def _generate_required_disclosures(
        self,
        client_profile: ClientProfile,
        product: InvestmentProduct
    ) -> List[str]:
        """Generate required regulatory disclosures"""
        
        disclosures = []
        
        if product.risk_level >= 6:
            disclosures.append("High-risk investment - principal loss possible")
        
        if product.complexity_score > 0.6:
            disclosures.append("Complex investment product - ensure understanding before investing")
        
        if product.liquidity_timeframe > 365:
            disclosures.append("Illiquid investment - funds may not be accessible for extended period")
        
        if product.leverage_factor > 1.0:
            disclosures.append("Leveraged product - losses may exceed initial investment")
        
        return disclosures
    
    def _requires_approval(
        self,
        suitability: InvestmentSuitability,
        analysis: Dict[str, Any]
    ) -> bool:
        """Determine if manual approval is required"""
        
        if suitability in [InvestmentSuitability.UNSUITABLE, InvestmentSuitability.PROHIBITED]:
            return True
        
        if suitability == InvestmentSuitability.MARGINALLY_SUITABLE:
            return True
        
        scores = analysis.get('score_breakdown', {})
        if scores.get('regulatory_compliance', 1) < 1.0:
            return True
        
        return False
    
    def _calculate_review_date(
        self,
        client_profile: ClientProfile,
        product: InvestmentProduct,
        suitability: InvestmentSuitability
    ) -> datetime:
        """Calculate next review date"""
        
        if suitability == InvestmentSuitability.HIGHLY_SUITABLE:
            return datetime.now() + timedelta(days=365)
        elif suitability == InvestmentSuitability.SUITABLE:
            return datetime.now() + timedelta(days=180)
        else:
            return datetime.now() + timedelta(days=90)
    
    def _generate_conditions(
        self,
        client_profile: ClientProfile,
        product: InvestmentProduct,
        suitability: InvestmentSuitability
    ) -> List[str]:
        """Generate investment conditions"""
        
        conditions = []
        
        if suitability == InvestmentSuitability.MARGINALLY_SUITABLE:
            conditions.append("Regular portfolio reviews required")
            conditions.append("Investment education recommended")
        
        if product.complexity_score > 0.7:
            conditions.append("Client must acknowledge understanding of product complexity")
        
        if client_profile.emergency_fund_months < 6:
            conditions.append("Maintain adequate emergency fund before investing")
        
        return conditions
    
    def _identify_key_factors(self, scores: Dict[str, float]) -> List[str]:
        """Identify key factors affecting suitability"""
        
        factors = []
        
        for factor, score in scores.items():
            if score >= 0.8:
                factors.append(f"Strong {factor.replace('_', ' ')}")
            elif score <= 0.3:
                factors.append(f"Weak {factor.replace('_', ' ')}")
        
        return factors
    
    def _identify_risk_factors(
        self,
        client_profile: ClientProfile,
        product: InvestmentProduct,
        scores: Dict[str, float]
    ) -> List[str]:
        """Identify specific risk factors"""
        
        risk_factors = []
        
        if scores.get('risk_alignment', 1) < 0.5:
            risk_factors.append("Risk tolerance mismatch")
        
        if scores.get('experience_alignment', 1) < 0.5:
            risk_factors.append("Insufficient investment experience")
        
        if scores.get('liquidity_alignment', 1) < 0.5:
            risk_factors.append("Liquidity mismatch")
        
        return risk_factors
    
    def _initialize_regulatory_requirements(self) -> Dict[str, List[str]]:
        """Initialize regulatory requirements database"""
        
        return {
            'hedge_funds': ['accredited_investor', 'qualified_purchaser'],
            'private_equity': ['accredited_investor', 'qualified_purchaser'],
            'structured_products': ['professional_investor'],
            'options': ['options_approval'],
            'futures': ['futures_approval'],
            'forex': ['forex_approval']
        }

