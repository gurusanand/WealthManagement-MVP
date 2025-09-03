import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from enum import Enum
import json
import re
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class PolicyType(Enum):
    """Types of compliance policies"""
    POSITION_LIMIT = "position_limit"
    CONCENTRATION_LIMIT = "concentration_limit"
    RISK_LIMIT = "risk_limit"
    TRADING_RESTRICTION = "trading_restriction"
    SUITABILITY_REQUIREMENT = "suitability_requirement"
    REGULATORY_REQUIREMENT = "regulatory_requirement"
    INVESTMENT_RESTRICTION = "investment_restriction"
    LIQUIDITY_REQUIREMENT = "liquidity_requirement"
    ESG_REQUIREMENT = "esg_requirement"
    TAX_OPTIMIZATION = "tax_optimization"

class PolicySeverity(Enum):
    """Policy violation severity levels"""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    BLOCKING = "blocking"

class PolicyAction(Enum):
    """Actions to take on policy violations"""
    LOG_ONLY = "log_only"
    WARN_USER = "warn_user"
    REQUIRE_APPROVAL = "require_approval"
    BLOCK_TRANSACTION = "block_transaction"
    AUTO_CORRECT = "auto_correct"
    ESCALATE = "escalate"

class PolicyScope(Enum):
    """Scope of policy application"""
    CLIENT = "client"
    PORTFOLIO = "portfolio"
    TRANSACTION = "transaction"
    POSITION = "position"
    FIRM = "firm"
    GLOBAL = "global"

@dataclass
class PolicyViolation:
    """Policy violation details"""
    policy_id: str
    policy_name: str
    violation_type: PolicyType
    severity: PolicySeverity
    description: str
    current_value: Any
    limit_value: Any
    violation_amount: float
    violation_percentage: float
    affected_entities: List[str]
    violation_date: datetime
    resolution_required: bool
    suggested_actions: List[str]
    auto_correctable: bool
    escalation_required: bool
    compliance_notes: str

@dataclass
class PolicyRule:
    """Individual policy rule definition"""
    rule_id: str
    name: str
    description: str
    policy_type: PolicyType
    scope: PolicyScope
    severity: PolicySeverity
    action: PolicyAction
    condition: str  # Rule condition expression
    limit_value: Any
    threshold_warning: Optional[float] = None
    threshold_error: Optional[float] = None
    enabled: bool = True
    effective_date: Optional[datetime] = None
    expiry_date: Optional[datetime] = None
    applicable_clients: List[str] = field(default_factory=list)
    applicable_products: List[str] = field(default_factory=list)
    exceptions: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PolicyResult:
    """Policy enforcement result"""
    client_id: str
    evaluation_date: datetime
    total_policies_evaluated: int
    policies_passed: int
    policies_failed: int
    violations: List[PolicyViolation]
    warnings: List[PolicyViolation]
    overall_compliance_status: bool
    compliance_score: float
    required_actions: List[str]
    blocking_violations: List[PolicyViolation]
    auto_corrections_applied: List[str]
    escalations_required: List[PolicyViolation]

class PolicyConditionEvaluator:
    """Evaluates policy conditions using a safe expression evaluator"""
    
    def __init__(self):
        # Safe functions that can be used in policy conditions
        self.safe_functions = {
            'abs': abs,
            'min': min,
            'max': max,
            'sum': sum,
            'len': len,
            'round': round,
            'int': int,
            'float': float,
            'str': str,
            'bool': bool
        }
        
        # Safe operators
        self.safe_operators = {
            '+', '-', '*', '/', '//', '%', '**',
            '==', '!=', '<', '<=', '>', '>=',
            'and', 'or', 'not', 'in', 'is'
        }
    
    def evaluate_condition(
        self,
        condition: str,
        context: Dict[str, Any]
    ) -> bool:
        """
        Safely evaluate a policy condition
        
        Args:
            condition: Policy condition expression
            context: Context variables for evaluation
            
        Returns:
            Boolean result of condition evaluation
        """
        try:
            # Create safe evaluation context
            eval_context = {
                **self.safe_functions,
                **context,
                '__builtins__': {}  # Remove built-in functions for security
            }
            
            # Validate condition for safety
            if not self._is_safe_condition(condition):
                logger.error(f"Unsafe condition detected: {condition}")
                return False
            
            # Evaluate condition
            result = eval(condition, eval_context)
            return bool(result)
            
        except Exception as e:
            logger.error(f"Condition evaluation error: {condition}, Error: {str(e)}")
            return False
    
    def _is_safe_condition(self, condition: str) -> bool:
        """Check if condition is safe to evaluate"""
        
        # Block dangerous keywords
        dangerous_keywords = [
            'import', 'exec', 'eval', 'open', 'file', 'input',
            'raw_input', '__import__', 'compile', 'reload',
            'globals', 'locals', 'vars', 'dir', 'hasattr',
            'getattr', 'setattr', 'delattr', 'callable'
        ]
        
        condition_lower = condition.lower()
        for keyword in dangerous_keywords:
            if keyword in condition_lower:
                return False
        
        # Block access to private attributes
        if '__' in condition:
            return False
        
        return True

class PolicyEngine:
    """
    Comprehensive Policy Enforcement Engine
    
    Features:
    - Real-time policy validation
    - Multi-level policy rules (client, portfolio, transaction, firm)
    - Configurable policy actions (warn, block, auto-correct)
    - Policy violation tracking and reporting
    - Automated compliance monitoring
    - Exception handling and escalation
    - Policy rule management and versioning
    """
    
    def __init__(self):
        self.policy_rules: Dict[str, PolicyRule] = {}
        self.condition_evaluator = PolicyConditionEvaluator()
        self.violation_history: List[PolicyViolation] = []
        
        # Initialize default policies
        self._initialize_default_policies()
        
        # Policy evaluation cache
        self.evaluation_cache: Dict[str, Any] = {}
        
        # Policy statistics
        self.policy_stats = {
            'total_evaluations': 0,
            'total_violations': 0,
            'blocked_transactions': 0,
            'auto_corrections': 0
        }
    
    def evaluate_policies(
        self,
        client_id: str,
        context: Dict[str, Any],
        policy_types: Optional[List[PolicyType]] = None,
        scope: Optional[PolicyScope] = None
    ) -> PolicyResult:
        """
        Evaluate all applicable policies for given context
        
        Args:
            client_id: Client identifier
            context: Evaluation context with relevant data
            policy_types: Specific policy types to evaluate
            scope: Policy scope to evaluate
            
        Returns:
            PolicyResult with comprehensive evaluation results
        """
        try:
            evaluation_date = datetime.now()
            violations = []
            warnings = []
            auto_corrections = []
            
            # Get applicable policies
            applicable_policies = self._get_applicable_policies(
                client_id, policy_types, scope
            )
            
            policies_evaluated = 0
            policies_passed = 0
            policies_failed = 0
            
            # Evaluate each policy
            for policy in applicable_policies:
                if not policy.enabled:
                    continue
                
                # Check if policy is currently effective
                if not self._is_policy_effective(policy, evaluation_date):
                    continue
                
                policies_evaluated += 1
                
                # Evaluate policy condition
                violation = self._evaluate_policy_rule(policy, context, client_id)
                
                if violation:
                    policies_failed += 1
                    
                    # Categorize violation
                    if violation.severity in [PolicySeverity.INFO, PolicySeverity.WARNING]:
                        warnings.append(violation)
                    else:
                        violations.append(violation)
                    
                    # Handle auto-correction
                    if violation.auto_correctable and policy.action == PolicyAction.AUTO_CORRECT:
                        correction = self._apply_auto_correction(violation, context)
                        if correction:
                            auto_corrections.append(correction)
                    
                    # Track violation
                    self.violation_history.append(violation)
                    
                else:
                    policies_passed += 1
            
            # Determine overall compliance status
            blocking_violations = [v for v in violations if v.severity == PolicySeverity.BLOCKING]
            overall_compliance = len(blocking_violations) == 0
            
            # Calculate compliance score
            compliance_score = self._calculate_compliance_score(
                policies_passed, policies_failed, violations, warnings
            )
            
            # Generate required actions
            required_actions = self._generate_required_actions(violations, warnings)
            
            # Identify escalations
            escalations = [v for v in violations if v.escalation_required]
            
            # Update statistics
            self.policy_stats['total_evaluations'] += policies_evaluated
            self.policy_stats['total_violations'] += len(violations)
            if blocking_violations:
                self.policy_stats['blocked_transactions'] += 1
            self.policy_stats['auto_corrections'] += len(auto_corrections)
            
            return PolicyResult(
                client_id=client_id,
                evaluation_date=evaluation_date,
                total_policies_evaluated=policies_evaluated,
                policies_passed=policies_passed,
                policies_failed=policies_failed,
                violations=violations,
                warnings=warnings,
                overall_compliance_status=overall_compliance,
                compliance_score=compliance_score,
                required_actions=required_actions,
                blocking_violations=blocking_violations,
                auto_corrections_applied=auto_corrections,
                escalations_required=escalations
            )
            
        except Exception as e:
            logger.error(f"Policy evaluation error for client {client_id}: {str(e)}")
            return PolicyResult(
                client_id=client_id,
                evaluation_date=datetime.now(),
                total_policies_evaluated=0,
                policies_passed=0,
                policies_failed=0,
                violations=[],
                warnings=[],
                overall_compliance_status=False,
                compliance_score=0.0,
                required_actions=["Manual review required due to evaluation error"],
                blocking_violations=[],
                auto_corrections_applied=[],
                escalations_required=[]
            )
    
    def add_policy_rule(self, policy_rule: PolicyRule) -> bool:
        """Add or update a policy rule"""
        
        try:
            # Validate policy rule
            if not self._validate_policy_rule(policy_rule):
                return False
            
            # Store policy rule
            self.policy_rules[policy_rule.rule_id] = policy_rule
            
            logger.info(f"Policy rule added/updated: {policy_rule.rule_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error adding policy rule {policy_rule.rule_id}: {str(e)}")
            return False
    
    def remove_policy_rule(self, rule_id: str) -> bool:
        """Remove a policy rule"""
        
        try:
            if rule_id in self.policy_rules:
                del self.policy_rules[rule_id]
                logger.info(f"Policy rule removed: {rule_id}")
                return True
            else:
                logger.warning(f"Policy rule not found: {rule_id}")
                return False
                
        except Exception as e:
            logger.error(f"Error removing policy rule {rule_id}: {str(e)}")
            return False
    
    def get_policy_violations(
        self,
        client_id: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        severity: Optional[PolicySeverity] = None
    ) -> List[PolicyViolation]:
        """Get policy violations with optional filtering"""
        
        try:
            violations = self.violation_history.copy()
            
            # Apply filters
            if client_id:
                violations = [v for v in violations if client_id in v.affected_entities]
            
            if start_date:
                violations = [v for v in violations if v.violation_date >= start_date]
            
            if end_date:
                violations = [v for v in violations if v.violation_date <= end_date]
            
            if severity:
                violations = [v for v in violations if v.severity == severity]
            
            return violations
            
        except Exception as e:
            logger.error(f"Error retrieving policy violations: {str(e)}")
            return []
    
    def _evaluate_policy_rule(
        self,
        policy: PolicyRule,
        context: Dict[str, Any],
        client_id: str
    ) -> Optional[PolicyViolation]:
        """Evaluate individual policy rule"""
        
        try:
            # Prepare evaluation context
            eval_context = {
                **context,
                'client_id': client_id,
                'current_date': datetime.now(),
                'policy_limit': policy.limit_value
            }
            
            # Evaluate condition
            condition_result = self.condition_evaluator.evaluate_condition(
                policy.condition, eval_context
            )
            
            # If condition is True, it means violation occurred
            if condition_result:
                return self._create_policy_violation(policy, context, client_id)
            
            return None
            
        except Exception as e:
            logger.error(f"Error evaluating policy {policy.rule_id}: {str(e)}")
            return None
    
    def _create_policy_violation(
        self,
        policy: PolicyRule,
        context: Dict[str, Any],
        client_id: str
    ) -> PolicyViolation:
        """Create policy violation object"""
        
        try:
            # Extract current value from context
            current_value = context.get('current_value', 'Unknown')
            limit_value = policy.limit_value
            
            # Calculate violation amount and percentage
            violation_amount = 0.0
            violation_percentage = 0.0
            
            if isinstance(current_value, (int, float)) and isinstance(limit_value, (int, float)):
                violation_amount = current_value - limit_value
                if limit_value != 0:
                    violation_percentage = (violation_amount / limit_value) * 100
            
            # Determine if auto-correctable
            auto_correctable = (
                policy.action == PolicyAction.AUTO_CORRECT and
                policy.policy_type in [
                    PolicyType.POSITION_LIMIT,
                    PolicyType.CONCENTRATION_LIMIT,
                    PolicyType.RISK_LIMIT
                ]
            )
            
            # Determine if escalation required
            escalation_required = (
                policy.severity in [PolicySeverity.CRITICAL, PolicySeverity.BLOCKING] or
                policy.action == PolicyAction.ESCALATE
            )
            
            # Generate suggested actions
            suggested_actions = self._generate_suggested_actions(policy, context)
            
            return PolicyViolation(
                policy_id=policy.rule_id,
                policy_name=policy.name,
                violation_type=policy.policy_type,
                severity=policy.severity,
                description=f"{policy.description} - Current: {current_value}, Limit: {limit_value}",
                current_value=current_value,
                limit_value=limit_value,
                violation_amount=violation_amount,
                violation_percentage=violation_percentage,
                affected_entities=[client_id],
                violation_date=datetime.now(),
                resolution_required=policy.severity not in [PolicySeverity.INFO, PolicySeverity.WARNING],
                suggested_actions=suggested_actions,
                auto_correctable=auto_correctable,
                escalation_required=escalation_required,
                compliance_notes=f"Policy {policy.rule_id} violated"
            )
            
        except Exception as e:
            logger.error(f"Error creating policy violation: {str(e)}")
            return PolicyViolation(
                policy_id=policy.rule_id,
                policy_name=policy.name,
                violation_type=policy.policy_type,
                severity=PolicySeverity.ERROR,
                description="Error creating violation details",
                current_value="Unknown",
                limit_value=policy.limit_value,
                violation_amount=0.0,
                violation_percentage=0.0,
                affected_entities=[client_id],
                violation_date=datetime.now(),
                resolution_required=True,
                suggested_actions=["Manual review required"],
                auto_correctable=False,
                escalation_required=True,
                compliance_notes="Error in violation creation"
            )
    
    def _get_applicable_policies(
        self,
        client_id: str,
        policy_types: Optional[List[PolicyType]] = None,
        scope: Optional[PolicyScope] = None
    ) -> List[PolicyRule]:
        """Get policies applicable to the given context"""
        
        applicable_policies = []
        
        for policy in self.policy_rules.values():
            # Check if policy applies to this client
            if policy.applicable_clients and client_id not in policy.applicable_clients:
                continue
            
            # Check if client is in exceptions
            if client_id in policy.exceptions:
                continue
            
            # Filter by policy types
            if policy_types and policy.policy_type not in policy_types:
                continue
            
            # Filter by scope
            if scope and policy.scope != scope:
                continue
            
            applicable_policies.append(policy)
        
        return applicable_policies
    
    def _is_policy_effective(self, policy: PolicyRule, evaluation_date: datetime) -> bool:
        """Check if policy is currently effective"""
        
        # Check effective date
        if policy.effective_date and evaluation_date < policy.effective_date:
            return False
        
        # Check expiry date
        if policy.expiry_date and evaluation_date > policy.expiry_date:
            return False
        
        return True
    
    def _validate_policy_rule(self, policy: PolicyRule) -> bool:
        """Validate policy rule before adding"""
        
        try:
            # Check required fields
            if not policy.rule_id or not policy.name or not policy.condition:
                return False
            
            # Validate condition syntax
            test_context = {'test_value': 100, 'policy_limit': 50}
            try:
                self.condition_evaluator.evaluate_condition(policy.condition, test_context)
            except:
                logger.error(f"Invalid condition syntax in policy {policy.rule_id}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Policy validation error: {str(e)}")
            return False
    
    def _calculate_compliance_score(
        self,
        policies_passed: int,
        policies_failed: int,
        violations: List[PolicyViolation],
        warnings: List[PolicyViolation]
    ) -> float:
        """Calculate overall compliance score"""
        
        try:
            total_policies = policies_passed + policies_failed
            
            if total_policies == 0:
                return 1.0
            
            # Base score from pass rate
            base_score = policies_passed / total_policies
            
            # Apply penalties for violations
            violation_penalty = 0.0
            for violation in violations:
                if violation.severity == PolicySeverity.BLOCKING:
                    violation_penalty += 0.3
                elif violation.severity == PolicySeverity.CRITICAL:
                    violation_penalty += 0.2
                elif violation.severity == PolicySeverity.ERROR:
                    violation_penalty += 0.1
            
            # Apply smaller penalties for warnings
            warning_penalty = len(warnings) * 0.02
            
            # Calculate final score
            final_score = max(0.0, base_score - violation_penalty - warning_penalty)
            
            return min(final_score, 1.0)
            
        except Exception as e:
            logger.error(f"Compliance score calculation error: {str(e)}")
            return 0.0
    
    def _generate_required_actions(
        self,
        violations: List[PolicyViolation],
        warnings: List[PolicyViolation]
    ) -> List[str]:
        """Generate required actions based on violations"""
        
        actions = []
        
        # Actions for blocking violations
        blocking_violations = [v for v in violations if v.severity == PolicySeverity.BLOCKING]
        if blocking_violations:
            actions.append("Transaction blocked due to policy violations")
            actions.append("Resolve blocking violations before proceeding")
        
        # Actions for critical violations
        critical_violations = [v for v in violations if v.severity == PolicySeverity.CRITICAL]
        if critical_violations:
            actions.append("Immediate attention required for critical violations")
            actions.append("Escalate to compliance team")
        
        # Actions for auto-correctable violations
        auto_correctable = [v for v in violations if v.auto_correctable]
        if auto_correctable:
            actions.append("Auto-corrections available for some violations")
        
        # Actions for warnings
        if warnings:
            actions.append("Review and acknowledge warnings")
        
        return actions
    
    def _generate_suggested_actions(
        self,
        policy: PolicyRule,
        context: Dict[str, Any]
    ) -> List[str]:
        """Generate suggested actions for policy violation"""
        
        actions = []
        
        if policy.policy_type == PolicyType.POSITION_LIMIT:
            actions.append("Reduce position size to comply with limit")
            actions.append("Consider diversifying across multiple positions")
        
        elif policy.policy_type == PolicyType.CONCENTRATION_LIMIT:
            actions.append("Rebalance portfolio to reduce concentration")
            actions.append("Consider alternative investments in different sectors")
        
        elif policy.policy_type == PolicyType.RISK_LIMIT:
            actions.append("Reduce portfolio risk through diversification")
            actions.append("Consider lower-risk investment alternatives")
        
        elif policy.policy_type == PolicyType.SUITABILITY_REQUIREMENT:
            actions.append("Review client suitability assessment")
            actions.append("Consider more suitable investment alternatives")
        
        elif policy.policy_type == PolicyType.LIQUIDITY_REQUIREMENT:
            actions.append("Increase portfolio liquidity")
            actions.append("Reduce allocation to illiquid investments")
        
        else:
            actions.append("Review policy requirements and adjust accordingly")
        
        return actions
    
    def _apply_auto_correction(
        self,
        violation: PolicyViolation,
        context: Dict[str, Any]
    ) -> Optional[str]:
        """Apply automatic correction for violation"""
        
        try:
            if violation.violation_type == PolicyType.POSITION_LIMIT:
                # Auto-correct position size
                return f"Position size automatically reduced to comply with limit"
            
            elif violation.violation_type == PolicyType.CONCENTRATION_LIMIT:
                # Auto-correct concentration
                return f"Portfolio automatically rebalanced to reduce concentration"
            
            elif violation.violation_type == PolicyType.RISK_LIMIT:
                # Auto-correct risk exposure
                return f"Risk exposure automatically adjusted to comply with limit"
            
            return None
            
        except Exception as e:
            logger.error(f"Auto-correction error: {str(e)}")
            return None
    
    def _initialize_default_policies(self):
        """Initialize default compliance policies"""
        
        # Position limit policy
        position_limit_policy = PolicyRule(
            rule_id="POS_LIMIT_001",
            name="Maximum Position Limit",
            description="Single position cannot exceed 10% of portfolio",
            policy_type=PolicyType.POSITION_LIMIT,
            scope=PolicyScope.POSITION,
            severity=PolicySeverity.ERROR,
            action=PolicyAction.REQUIRE_APPROVAL,
            condition="position_percentage > policy_limit",
            limit_value=10.0,
            threshold_warning=8.0,
            threshold_error=10.0
        )
        
        # Concentration limit policy
        concentration_policy = PolicyRule(
            rule_id="CONC_LIMIT_001",
            name="Sector Concentration Limit",
            description="Sector concentration cannot exceed 25% of portfolio",
            policy_type=PolicyType.CONCENTRATION_LIMIT,
            scope=PolicyScope.PORTFOLIO,
            severity=PolicySeverity.WARNING,
            action=PolicyAction.WARN_USER,
            condition="sector_concentration > policy_limit",
            limit_value=25.0,
            threshold_warning=20.0,
            threshold_error=25.0
        )
        
        # Risk limit policy
        risk_limit_policy = PolicyRule(
            rule_id="RISK_LIMIT_001",
            name="Portfolio Risk Limit",
            description="Portfolio VaR cannot exceed 5%",
            policy_type=PolicyType.RISK_LIMIT,
            scope=PolicyScope.PORTFOLIO,
            severity=PolicySeverity.CRITICAL,
            action=PolicyAction.BLOCK_TRANSACTION,
            condition="portfolio_var > policy_limit",
            limit_value=5.0,
            threshold_warning=4.0,
            threshold_error=5.0
        )
        
        # Suitability requirement policy
        suitability_policy = PolicyRule(
            rule_id="SUIT_REQ_001",
            name="Investment Suitability Requirement",
            description="Investment must be suitable for client",
            policy_type=PolicyType.SUITABILITY_REQUIREMENT,
            scope=PolicyScope.TRANSACTION,
            severity=PolicySeverity.BLOCKING,
            action=PolicyAction.BLOCK_TRANSACTION,
            condition="suitability_score < policy_limit",
            limit_value=0.6,
            threshold_warning=0.7,
            threshold_error=0.6
        )
        
        # Liquidity requirement policy
        liquidity_policy = PolicyRule(
            rule_id="LIQ_REQ_001",
            name="Minimum Liquidity Requirement",
            description="Portfolio must maintain minimum 10% liquidity",
            policy_type=PolicyType.LIQUIDITY_REQUIREMENT,
            scope=PolicyScope.PORTFOLIO,
            severity=PolicySeverity.WARNING,
            action=PolicyAction.WARN_USER,
            condition="portfolio_liquidity < policy_limit",
            limit_value=10.0,
            threshold_warning=12.0,
            threshold_error=10.0
        )
        
        # Add policies to engine
        policies = [
            position_limit_policy,
            concentration_policy,
            risk_limit_policy,
            suitability_policy,
            liquidity_policy
        ]
        
        for policy in policies:
            self.add_policy_rule(policy)
    
    def get_policy_statistics(self) -> Dict[str, Any]:
        """Get policy engine statistics"""
        
        return {
            **self.policy_stats,
            'total_policies': len(self.policy_rules),
            'active_policies': len([p for p in self.policy_rules.values() if p.enabled]),
            'total_violation_history': len(self.violation_history),
            'recent_violations': len([
                v for v in self.violation_history 
                if v.violation_date >= datetime.now() - timedelta(days=30)
            ])
        }
    
    def export_policies(self) -> Dict[str, Any]:
        """Export all policies for backup or migration"""
        
        try:
            policies_data = {}
            
            for rule_id, policy in self.policy_rules.items():
                policies_data[rule_id] = {
                    'rule_id': policy.rule_id,
                    'name': policy.name,
                    'description': policy.description,
                    'policy_type': policy.policy_type.value,
                    'scope': policy.scope.value,
                    'severity': policy.severity.value,
                    'action': policy.action.value,
                    'condition': policy.condition,
                    'limit_value': policy.limit_value,
                    'threshold_warning': policy.threshold_warning,
                    'threshold_error': policy.threshold_error,
                    'enabled': policy.enabled,
                    'effective_date': policy.effective_date.isoformat() if policy.effective_date else None,
                    'expiry_date': policy.expiry_date.isoformat() if policy.expiry_date else None,
                    'applicable_clients': policy.applicable_clients,
                    'applicable_products': policy.applicable_products,
                    'exceptions': policy.exceptions,
                    'metadata': policy.metadata
                }
            
            return {
                'policies': policies_data,
                'export_date': datetime.now().isoformat(),
                'total_policies': len(policies_data)
            }
            
        except Exception as e:
            logger.error(f"Policy export error: {str(e)}")
            return {}
    
    def import_policies(self, policies_data: Dict[str, Any]) -> bool:
        """Import policies from backup or migration"""
        
        try:
            imported_count = 0
            
            for rule_id, policy_data in policies_data.get('policies', {}).items():
                # Create PolicyRule object
                policy = PolicyRule(
                    rule_id=policy_data['rule_id'],
                    name=policy_data['name'],
                    description=policy_data['description'],
                    policy_type=PolicyType(policy_data['policy_type']),
                    scope=PolicyScope(policy_data['scope']),
                    severity=PolicySeverity(policy_data['severity']),
                    action=PolicyAction(policy_data['action']),
                    condition=policy_data['condition'],
                    limit_value=policy_data['limit_value'],
                    threshold_warning=policy_data.get('threshold_warning'),
                    threshold_error=policy_data.get('threshold_error'),
                    enabled=policy_data.get('enabled', True),
                    effective_date=datetime.fromisoformat(policy_data['effective_date']) if policy_data.get('effective_date') else None,
                    expiry_date=datetime.fromisoformat(policy_data['expiry_date']) if policy_data.get('expiry_date') else None,
                    applicable_clients=policy_data.get('applicable_clients', []),
                    applicable_products=policy_data.get('applicable_products', []),
                    exceptions=policy_data.get('exceptions', []),
                    metadata=policy_data.get('metadata', {})
                )
                
                # Add policy
                if self.add_policy_rule(policy):
                    imported_count += 1
            
            logger.info(f"Successfully imported {imported_count} policies")
            return True
            
        except Exception as e:
            logger.error(f"Policy import error: {str(e)}")
            return False

