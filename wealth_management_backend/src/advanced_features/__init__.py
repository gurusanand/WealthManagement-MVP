# Advanced Features Package
# This package contains sophisticated AI-powered features for the wealth management system

from .lifegraph_twin import LifeGraphTwin, LifeEvent, LifeEventType
from .advanced_analytics import AdvancedAnalyticsEngine, MarketRegimeDetector
from .ml_models import ClientBehaviorPredictor, RiskFactorModel, PerformanceAttributor
from .event_processor import IntelligentEventProcessor, EventImpactAnalyzer
from .compliance_intelligence import ComplianceIntelligence, PredictiveComplianceRisk
from .data_fusion import DataFusionEngine, AlternativeDataProcessor

__all__ = [
    'LifeGraphTwin',
    'LifeEvent', 
    'LifeEventType',
    'AdvancedAnalyticsEngine',
    'MarketRegimeDetector',
    'ClientBehaviorPredictor',
    'RiskFactorModel',
    'PerformanceAttributor',
    'IntelligentEventProcessor',
    'EventImpactAnalyzer',
    'ComplianceIntelligence',
    'PredictiveComplianceRisk',
    'DataFusionEngine',
    'AlternativeDataProcessor'
]

