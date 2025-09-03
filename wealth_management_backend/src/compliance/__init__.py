"""
Compliance and Risk Management System

This package provides comprehensive compliance monitoring, regulatory adherence,
and risk management capabilities for the wealth management system.

Components:
- KYC/AML Engine: Know Your Customer and Anti-Money Laundering compliance
- Suitability Assessment: Investment suitability analysis and validation
- Cross-Border Compliance: Multi-jurisdiction regulatory compliance
- Policy Enforcement: Real-time compliance rule enforcement
- Audit Trail: Comprehensive audit logging and reporting
- Regulatory Reporting: Automated regulatory report generation
"""

from .kyc_aml_engine import KYCAMLEngine, KYCResult, AMLResult, ComplianceStatus
from .suitability_engine import SuitabilityEngine, SuitabilityResult, InvestmentSuitability
from .policy_engine import PolicyEngine, PolicyResult, PolicyViolation
from .audit_engine import AuditEngine, AuditEvent, AuditTrail
from .regulatory_engine import RegulatoryEngine, RegulatoryRequirement, ComplianceReport

__all__ = [
    'KYCAMLEngine', 'KYCResult', 'AMLResult', 'ComplianceStatus',
    'SuitabilityEngine', 'SuitabilityResult', 'InvestmentSuitability',
    'PolicyEngine', 'PolicyResult', 'PolicyViolation',
    'AuditEngine', 'AuditEvent', 'AuditTrail',
    'RegulatoryEngine', 'RegulatoryRequirement', 'ComplianceReport'
]

