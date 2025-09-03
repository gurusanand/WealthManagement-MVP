import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Union
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
import uuid
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

class AuditEventType(Enum):
    """Types of audit events"""
    USER_LOGIN = "user_login"
    USER_LOGOUT = "user_logout"
    TRANSACTION_CREATED = "transaction_created"
    TRANSACTION_EXECUTED = "transaction_executed"
    TRANSACTION_CANCELLED = "transaction_cancelled"
    PORTFOLIO_REBALANCED = "portfolio_rebalanced"
    POLICY_VIOLATION = "policy_violation"
    COMPLIANCE_CHECK = "compliance_check"
    KYC_VERIFICATION = "kyc_verification"
    AML_SCREENING = "aml_screening"
    SUITABILITY_ASSESSMENT = "suitability_assessment"
    DOCUMENT_UPLOADED = "document_uploaded"
    DOCUMENT_VERIFIED = "document_verified"
    APPROVAL_GRANTED = "approval_granted"
    APPROVAL_DENIED = "approval_denied"
    RISK_ASSESSMENT = "risk_assessment"
    CLIENT_ONBOARDED = "client_onboarded"
    CLIENT_UPDATED = "client_updated"
    PORTFOLIO_CREATED = "portfolio_created"
    PORTFOLIO_UPDATED = "portfolio_updated"
    SYSTEM_ACCESS = "system_access"
    DATA_EXPORT = "data_export"
    CONFIGURATION_CHANGE = "configuration_change"
    REGULATORY_REPORT = "regulatory_report"

class AuditSeverity(Enum):
    """Audit event severity levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AuditStatus(Enum):
    """Audit event status"""
    SUCCESS = "success"
    FAILURE = "failure"
    WARNING = "warning"
    PENDING = "pending"
    CANCELLED = "cancelled"

@dataclass
class AuditEvent:
    """Individual audit event"""
    event_id: str
    event_type: AuditEventType
    timestamp: datetime
    user_id: Optional[str]
    client_id: Optional[str]
    session_id: Optional[str]
    ip_address: Optional[str]
    user_agent: Optional[str]
    severity: AuditSeverity
    status: AuditStatus
    description: str
    details: Dict[str, Any]
    before_state: Optional[Dict[str, Any]] = None
    after_state: Optional[Dict[str, Any]] = None
    affected_entities: List[str] = field(default_factory=list)
    compliance_relevant: bool = True
    retention_period_days: int = 2555  # 7 years default
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    checksum: Optional[str] = None

@dataclass
class AuditTrail:
    """Collection of related audit events"""
    trail_id: str
    trail_name: str
    start_time: datetime
    end_time: Optional[datetime]
    events: List[AuditEvent]
    total_events: int
    success_events: int
    failure_events: int
    warning_events: int
    compliance_events: int
    trail_status: AuditStatus
    created_by: str
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AuditQuery:
    """Audit query parameters"""
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    event_types: Optional[List[AuditEventType]] = None
    user_ids: Optional[List[str]] = None
    client_ids: Optional[List[str]] = None
    severities: Optional[List[AuditSeverity]] = None
    statuses: Optional[List[AuditStatus]] = None
    compliance_relevant_only: bool = False
    tags: Optional[List[str]] = None
    search_text: Optional[str] = None
    limit: int = 1000
    offset: int = 0

@dataclass
class AuditReport:
    """Audit report summary"""
    report_id: str
    report_name: str
    generation_date: datetime
    query_parameters: AuditQuery
    total_events: int
    event_summary: Dict[AuditEventType, int]
    severity_summary: Dict[AuditSeverity, int]
    status_summary: Dict[AuditStatus, int]
    compliance_summary: Dict[str, Any]
    timeline_analysis: Dict[str, Any]
    user_activity_summary: Dict[str, int]
    client_activity_summary: Dict[str, int]
    recommendations: List[str]
    export_formats: List[str] = field(default_factory=lambda: ['json', 'csv', 'pdf'])

class AuditEngine:
    """
    Comprehensive Audit Trail and Compliance Logging System
    
    Features:
    - Complete audit trail for all system activities
    - Compliance-focused event logging
    - Tamper-evident audit records with checksums
    - Advanced audit querying and filtering
    - Automated audit report generation
    - Regulatory compliance reporting
    - Data retention management
    - Real-time audit monitoring
    - Audit trail integrity verification
    """
    
    def __init__(self):
        # Audit event storage
        self.audit_events: List[AuditEvent] = []
        self.audit_trails: Dict[str, AuditTrail] = {}
        
        # Audit configuration
        self.config = {
            'default_retention_days': 2555,  # 7 years
            'max_events_in_memory': 10000,
            'auto_archive_threshold': 50000,
            'checksum_enabled': True,
            'real_time_monitoring': True,
            'compliance_alerts_enabled': True
        }
        
        # Event type configurations
        self.event_configs = self._initialize_event_configurations()
        
        # Audit statistics
        self.audit_stats = {
            'total_events_logged': 0,
            'compliance_events_logged': 0,
            'critical_events_logged': 0,
            'failed_events_logged': 0,
            'integrity_checks_performed': 0,
            'integrity_violations_detected': 0
        }
        
        # Active audit trails
        self.active_trails: Dict[str, AuditTrail] = {}
    
    def log_event(
        self,
        event_type: AuditEventType,
        description: str,
        user_id: Optional[str] = None,
        client_id: Optional[str] = None,
        session_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        user_agent: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
        before_state: Optional[Dict[str, Any]] = None,
        after_state: Optional[Dict[str, Any]] = None,
        severity: Optional[AuditSeverity] = None,
        status: AuditStatus = AuditStatus.SUCCESS,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Log an audit event
        
        Args:
            event_type: Type of audit event
            description: Human-readable description
            user_id: User who performed the action
            client_id: Affected client (if applicable)
            session_id: User session identifier
            ip_address: Source IP address
            user_agent: User agent string
            details: Detailed event information
            before_state: State before the action
            after_state: State after the action
            severity: Event severity level
            status: Event status
            tags: Event tags for categorization
            metadata: Additional metadata
            
        Returns:
            Event ID of the logged event
        """
        try:
            # Generate unique event ID
            event_id = str(uuid.uuid4())
            
            # Determine severity if not provided
            if severity is None:
                severity = self._determine_event_severity(event_type, status)
            
            # Get event configuration
            event_config = self.event_configs.get(event_type, {})
            
            # Determine compliance relevance
            compliance_relevant = event_config.get('compliance_relevant', True)
            
            # Determine retention period
            retention_days = event_config.get('retention_days', self.config['default_retention_days'])
            
            # Create audit event
            audit_event = AuditEvent(
                event_id=event_id,
                event_type=event_type,
                timestamp=datetime.now(),
                user_id=user_id,
                client_id=client_id,
                session_id=session_id,
                ip_address=ip_address,
                user_agent=user_agent,
                severity=severity,
                status=status,
                description=description,
                details=details or {},
                before_state=before_state,
                after_state=after_state,
                affected_entities=self._extract_affected_entities(details, client_id),
                compliance_relevant=compliance_relevant,
                retention_period_days=retention_days,
                tags=tags or [],
                metadata=metadata or {}
            )
            
            # Generate checksum for integrity
            if self.config['checksum_enabled']:
                audit_event.checksum = self._generate_event_checksum(audit_event)
            
            # Store audit event
            self.audit_events.append(audit_event)
            
            # Update statistics
            self._update_audit_statistics(audit_event)
            
            # Add to active trails
            self._add_to_active_trails(audit_event)
            
            # Trigger real-time monitoring
            if self.config['real_time_monitoring']:
                self._trigger_real_time_monitoring(audit_event)
            
            # Check for compliance alerts
            if self.config['compliance_alerts_enabled'] and compliance_relevant:
                self._check_compliance_alerts(audit_event)
            
            # Auto-archive if threshold reached
            if len(self.audit_events) >= self.config['auto_archive_threshold']:
                self._auto_archive_events()
            
            logger.info(f"Audit event logged: {event_id} - {event_type.value}")
            return event_id
            
        except Exception as e:
            logger.error(f"Error logging audit event: {str(e)}")
            return ""
    
    def start_audit_trail(
        self,
        trail_name: str,
        created_by: str,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Start a new audit trail for related events
        
        Args:
            trail_name: Name of the audit trail
            created_by: User who created the trail
            tags: Trail tags
            metadata: Trail metadata
            
        Returns:
            Trail ID
        """
        try:
            trail_id = str(uuid.uuid4())
            
            audit_trail = AuditTrail(
                trail_id=trail_id,
                trail_name=trail_name,
                start_time=datetime.now(),
                end_time=None,
                events=[],
                total_events=0,
                success_events=0,
                failure_events=0,
                warning_events=0,
                compliance_events=0,
                trail_status=AuditStatus.PENDING,
                created_by=created_by,
                tags=tags or [],
                metadata=metadata or {}
            )
            
            self.active_trails[trail_id] = audit_trail
            
            # Log trail start event
            self.log_event(
                event_type=AuditEventType.SYSTEM_ACCESS,
                description=f"Audit trail started: {trail_name}",
                user_id=created_by,
                details={'trail_id': trail_id, 'trail_name': trail_name},
                tags=['audit_trail', 'system']
            )
            
            logger.info(f"Audit trail started: {trail_id} - {trail_name}")
            return trail_id
            
        except Exception as e:
            logger.error(f"Error starting audit trail: {str(e)}")
            return ""
    
    def end_audit_trail(self, trail_id: str, status: AuditStatus = AuditStatus.SUCCESS) -> bool:
        """
        End an active audit trail
        
        Args:
            trail_id: Trail identifier
            status: Final trail status
            
        Returns:
            Success status
        """
        try:
            if trail_id not in self.active_trails:
                logger.warning(f"Audit trail not found: {trail_id}")
                return False
            
            trail = self.active_trails[trail_id]
            trail.end_time = datetime.now()
            trail.trail_status = status
            
            # Move to completed trails
            self.audit_trails[trail_id] = trail
            del self.active_trails[trail_id]
            
            # Log trail end event
            self.log_event(
                event_type=AuditEventType.SYSTEM_ACCESS,
                description=f"Audit trail ended: {trail.trail_name}",
                user_id=trail.created_by,
                details={
                    'trail_id': trail_id,
                    'trail_name': trail.trail_name,
                    'total_events': trail.total_events,
                    'duration_minutes': (trail.end_time - trail.start_time).total_seconds() / 60
                },
                status=status,
                tags=['audit_trail', 'system']
            )
            
            logger.info(f"Audit trail ended: {trail_id}")
            return True
            
        except Exception as e:
            logger.error(f"Error ending audit trail {trail_id}: {str(e)}")
            return False
    
    def query_events(self, query: AuditQuery) -> List[AuditEvent]:
        """
        Query audit events with filtering
        
        Args:
            query: Audit query parameters
            
        Returns:
            List of matching audit events
        """
        try:
            events = self.audit_events.copy()
            
            # Apply filters
            if query.start_date:
                events = [e for e in events if e.timestamp >= query.start_date]
            
            if query.end_date:
                events = [e for e in events if e.timestamp <= query.end_date]
            
            if query.event_types:
                events = [e for e in events if e.event_type in query.event_types]
            
            if query.user_ids:
                events = [e for e in events if e.user_id in query.user_ids]
            
            if query.client_ids:
                events = [e for e in events if e.client_id in query.client_ids]
            
            if query.severities:
                events = [e for e in events if e.severity in query.severities]
            
            if query.statuses:
                events = [e for e in events if e.status in query.statuses]
            
            if query.compliance_relevant_only:
                events = [e for e in events if e.compliance_relevant]
            
            if query.tags:
                events = [e for e in events if any(tag in e.tags for tag in query.tags)]
            
            if query.search_text:
                search_lower = query.search_text.lower()
                events = [
                    e for e in events 
                    if search_lower in e.description.lower() or
                       search_lower in str(e.details).lower()
                ]
            
            # Sort by timestamp (newest first)
            events.sort(key=lambda x: x.timestamp, reverse=True)
            
            # Apply pagination
            start_idx = query.offset
            end_idx = start_idx + query.limit
            
            return events[start_idx:end_idx]
            
        except Exception as e:
            logger.error(f"Error querying audit events: {str(e)}")
            return []
    
    def generate_audit_report(
        self,
        query: AuditQuery,
        report_name: str,
        include_recommendations: bool = True
    ) -> AuditReport:
        """
        Generate comprehensive audit report
        
        Args:
            query: Query parameters for report
            report_name: Name of the report
            include_recommendations: Whether to include recommendations
            
        Returns:
            Comprehensive audit report
        """
        try:
            report_id = str(uuid.uuid4())
            
            # Query events
            events = self.query_events(query)
            
            # Generate event summary
            event_summary = {}
            for event_type in AuditEventType:
                count = len([e for e in events if e.event_type == event_type])
                if count > 0:
                    event_summary[event_type] = count
            
            # Generate severity summary
            severity_summary = {}
            for severity in AuditSeverity:
                count = len([e for e in events if e.severity == severity])
                if count > 0:
                    severity_summary[severity] = count
            
            # Generate status summary
            status_summary = {}
            for status in AuditStatus:
                count = len([e for e in events if e.status == status])
                if count > 0:
                    status_summary[status] = count
            
            # Generate compliance summary
            compliance_summary = self._generate_compliance_summary(events)
            
            # Generate timeline analysis
            timeline_analysis = self._generate_timeline_analysis(events)
            
            # Generate user activity summary
            user_activity = {}
            for event in events:
                if event.user_id:
                    user_activity[event.user_id] = user_activity.get(event.user_id, 0) + 1
            
            # Generate client activity summary
            client_activity = {}
            for event in events:
                if event.client_id:
                    client_activity[event.client_id] = client_activity.get(event.client_id, 0) + 1
            
            # Generate recommendations
            recommendations = []
            if include_recommendations:
                recommendations = self._generate_audit_recommendations(events, query)
            
            audit_report = AuditReport(
                report_id=report_id,
                report_name=report_name,
                generation_date=datetime.now(),
                query_parameters=query,
                total_events=len(events),
                event_summary=event_summary,
                severity_summary=severity_summary,
                status_summary=status_summary,
                compliance_summary=compliance_summary,
                timeline_analysis=timeline_analysis,
                user_activity_summary=user_activity,
                client_activity_summary=client_activity,
                recommendations=recommendations
            )
            
            # Log report generation
            self.log_event(
                event_type=AuditEventType.REGULATORY_REPORT,
                description=f"Audit report generated: {report_name}",
                details={
                    'report_id': report_id,
                    'total_events': len(events),
                    'query_parameters': query.__dict__
                },
                tags=['audit_report', 'compliance']
            )
            
            return audit_report
            
        except Exception as e:
            logger.error(f"Error generating audit report: {str(e)}")
            return AuditReport(
                report_id="error",
                report_name=report_name,
                generation_date=datetime.now(),
                query_parameters=query,
                total_events=0,
                event_summary={},
                severity_summary={},
                status_summary={},
                compliance_summary={},
                timeline_analysis={},
                user_activity_summary={},
                client_activity_summary={},
                recommendations=["Error generating report - manual review required"]
            )
    
    def verify_audit_integrity(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None
    ) -> Dict[str, Any]:
        """
        Verify audit trail integrity using checksums
        
        Args:
            start_date: Start date for verification
            end_date: End date for verification
            
        Returns:
            Integrity verification results
        """
        try:
            # Filter events for verification
            events_to_verify = self.audit_events.copy()
            
            if start_date:
                events_to_verify = [e for e in events_to_verify if e.timestamp >= start_date]
            
            if end_date:
                events_to_verify = [e for e in events_to_verify if e.timestamp <= end_date]
            
            total_events = len(events_to_verify)
            verified_events = 0
            integrity_violations = []
            
            for event in events_to_verify:
                if event.checksum:
                    # Recalculate checksum
                    calculated_checksum = self._generate_event_checksum(event)
                    
                    if calculated_checksum == event.checksum:
                        verified_events += 1
                    else:
                        integrity_violations.append({
                            'event_id': event.event_id,
                            'timestamp': event.timestamp,
                            'event_type': event.event_type.value,
                            'stored_checksum': event.checksum,
                            'calculated_checksum': calculated_checksum
                        })
                else:
                    # Event without checksum
                    integrity_violations.append({
                        'event_id': event.event_id,
                        'timestamp': event.timestamp,
                        'event_type': event.event_type.value,
                        'issue': 'Missing checksum'
                    })
            
            # Update statistics
            self.audit_stats['integrity_checks_performed'] += 1
            self.audit_stats['integrity_violations_detected'] += len(integrity_violations)
            
            integrity_result = {
                'verification_date': datetime.now(),
                'total_events_checked': total_events,
                'verified_events': verified_events,
                'integrity_violations': len(integrity_violations),
                'integrity_percentage': (verified_events / total_events * 100) if total_events > 0 else 100,
                'violations_details': integrity_violations,
                'overall_integrity': len(integrity_violations) == 0
            }
            
            # Log integrity check
            self.log_event(
                event_type=AuditEventType.SYSTEM_ACCESS,
                description="Audit integrity verification performed",
                details=integrity_result,
                severity=AuditSeverity.HIGH if integrity_violations else AuditSeverity.LOW,
                tags=['integrity_check', 'system']
            )
            
            return integrity_result
            
        except Exception as e:
            logger.error(f"Error verifying audit integrity: {str(e)}")
            return {
                'verification_date': datetime.now(),
                'total_events_checked': 0,
                'verified_events': 0,
                'integrity_violations': 0,
                'integrity_percentage': 0,
                'violations_details': [],
                'overall_integrity': False,
                'error': str(e)
            }
    
    def export_audit_data(
        self,
        query: AuditQuery,
        export_format: str = 'json',
        include_sensitive_data: bool = False
    ) -> Dict[str, Any]:
        """
        Export audit data in specified format
        
        Args:
            query: Query parameters for export
            export_format: Export format (json, csv, xml)
            include_sensitive_data: Whether to include sensitive information
            
        Returns:
            Export result with data or file path
        """
        try:
            # Query events
            events = self.query_events(query)
            
            # Prepare export data
            export_data = []
            for event in events:
                event_data = {
                    'event_id': event.event_id,
                    'event_type': event.event_type.value,
                    'timestamp': event.timestamp.isoformat(),
                    'user_id': event.user_id if include_sensitive_data else self._anonymize_user_id(event.user_id),
                    'client_id': event.client_id if include_sensitive_data else self._anonymize_client_id(event.client_id),
                    'severity': event.severity.value,
                    'status': event.status.value,
                    'description': event.description,
                    'compliance_relevant': event.compliance_relevant,
                    'tags': event.tags
                }
                
                # Include details if not sensitive or if sensitive data is allowed
                if include_sensitive_data or not self._contains_sensitive_data(event.details):
                    event_data['details'] = event.details
                
                export_data.append(event_data)
            
            # Log export event
            self.log_event(
                event_type=AuditEventType.DATA_EXPORT,
                description=f"Audit data exported in {export_format} format",
                details={
                    'total_events_exported': len(export_data),
                    'export_format': export_format,
                    'include_sensitive_data': include_sensitive_data,
                    'query_parameters': query.__dict__
                },
                severity=AuditSeverity.HIGH,
                tags=['data_export', 'compliance']
            )
            
            return {
                'export_date': datetime.now().isoformat(),
                'total_events': len(export_data),
                'format': export_format,
                'data': export_data,
                'query_parameters': query.__dict__
            }
            
        except Exception as e:
            logger.error(f"Error exporting audit data: {str(e)}")
            return {
                'export_date': datetime.now().isoformat(),
                'total_events': 0,
                'format': export_format,
                'data': [],
                'error': str(e)
            }
    
    def get_audit_statistics(self) -> Dict[str, Any]:
        """Get comprehensive audit statistics"""
        
        try:
            # Calculate additional statistics
            total_events = len(self.audit_events)
            recent_events = len([
                e for e in self.audit_events 
                if e.timestamp >= datetime.now() - timedelta(days=30)
            ])
            
            compliance_events = len([e for e in self.audit_events if e.compliance_relevant])
            critical_events = len([e for e in self.audit_events if e.severity == AuditSeverity.CRITICAL])
            failed_events = len([e for e in self.audit_events if e.status == AuditStatus.FAILURE])
            
            # Event type distribution
            event_type_distribution = {}
            for event in self.audit_events:
                event_type = event.event_type.value
                event_type_distribution[event_type] = event_type_distribution.get(event_type, 0) + 1
            
            # User activity statistics
            user_activity = {}
            for event in self.audit_events:
                if event.user_id:
                    user_activity[event.user_id] = user_activity.get(event.user_id, 0) + 1
            
            return {
                **self.audit_stats,
                'total_events_stored': total_events,
                'recent_events_30_days': recent_events,
                'compliance_events_total': compliance_events,
                'critical_events_total': critical_events,
                'failed_events_total': failed_events,
                'active_trails': len(self.active_trails),
                'completed_trails': len(self.audit_trails),
                'event_type_distribution': event_type_distribution,
                'top_users_by_activity': dict(sorted(user_activity.items(), key=lambda x: x[1], reverse=True)[:10]),
                'average_events_per_day': total_events / max((datetime.now() - datetime.now().replace(day=1)).days, 1),
                'integrity_check_status': 'enabled' if self.config['checksum_enabled'] else 'disabled'
            }
            
        except Exception as e:
            logger.error(f"Error calculating audit statistics: {str(e)}")
            return self.audit_stats
    
    def _determine_event_severity(self, event_type: AuditEventType, status: AuditStatus) -> AuditSeverity:
        """Determine event severity based on type and status"""
        
        # High severity events
        high_severity_events = [
            AuditEventType.POLICY_VIOLATION,
            AuditEventType.APPROVAL_DENIED,
            AuditEventType.DATA_EXPORT,
            AuditEventType.CONFIGURATION_CHANGE
        ]
        
        # Critical severity events
        critical_severity_events = [
            AuditEventType.AML_SCREENING,
            AuditEventType.REGULATORY_REPORT
        ]
        
        if event_type in critical_severity_events:
            return AuditSeverity.CRITICAL
        elif event_type in high_severity_events or status == AuditStatus.FAILURE:
            return AuditSeverity.HIGH
        elif status == AuditStatus.WARNING:
            return AuditSeverity.MEDIUM
        else:
            return AuditSeverity.LOW
    
    def _generate_event_checksum(self, event: AuditEvent) -> str:
        """Generate checksum for audit event integrity"""
        
        try:
            # Create checksum data
            checksum_data = {
                'event_id': event.event_id,
                'event_type': event.event_type.value,
                'timestamp': event.timestamp.isoformat(),
                'user_id': event.user_id,
                'client_id': event.client_id,
                'description': event.description,
                'details': json.dumps(event.details, sort_keys=True),
                'status': event.status.value
            }
            
            # Generate SHA-256 checksum
            checksum_string = json.dumps(checksum_data, sort_keys=True)
            checksum = hashlib.sha256(checksum_string.encode()).hexdigest()
            
            return checksum
            
        except Exception as e:
            logger.error(f"Error generating event checksum: {str(e)}")
            return ""
    
    def _extract_affected_entities(self, details: Optional[Dict[str, Any]], client_id: Optional[str]) -> List[str]:
        """Extract affected entities from event details"""
        
        entities = []
        
        if client_id:
            entities.append(f"client:{client_id}")
        
        if details:
            # Extract portfolio IDs
            if 'portfolio_id' in details:
                entities.append(f"portfolio:{details['portfolio_id']}")
            
            # Extract transaction IDs
            if 'transaction_id' in details:
                entities.append(f"transaction:{details['transaction_id']}")
            
            # Extract other entity IDs
            for key, value in details.items():
                if key.endswith('_id') and isinstance(value, str):
                    entity_type = key.replace('_id', '')
                    entities.append(f"{entity_type}:{value}")
        
        return entities
    
    def _update_audit_statistics(self, event: AuditEvent):
        """Update audit statistics with new event"""
        
        self.audit_stats['total_events_logged'] += 1
        
        if event.compliance_relevant:
            self.audit_stats['compliance_events_logged'] += 1
        
        if event.severity == AuditSeverity.CRITICAL:
            self.audit_stats['critical_events_logged'] += 1
        
        if event.status == AuditStatus.FAILURE:
            self.audit_stats['failed_events_logged'] += 1
    
    def _add_to_active_trails(self, event: AuditEvent):
        """Add event to active audit trails"""
        
        for trail in self.active_trails.values():
            # Add event to trail if it matches criteria
            if self._event_matches_trail(event, trail):
                trail.events.append(event)
                trail.total_events += 1
                
                if event.status == AuditStatus.SUCCESS:
                    trail.success_events += 1
                elif event.status == AuditStatus.FAILURE:
                    trail.failure_events += 1
                elif event.status == AuditStatus.WARNING:
                    trail.warning_events += 1
                
                if event.compliance_relevant:
                    trail.compliance_events += 1
    
    def _event_matches_trail(self, event: AuditEvent, trail: AuditTrail) -> bool:
        """Check if event matches trail criteria"""
        
        # Simple matching based on user or client
        if event.user_id == trail.created_by:
            return True
        
        # Match based on tags
        if any(tag in event.tags for tag in trail.tags):
            return True
        
        return False
    
    def _trigger_real_time_monitoring(self, event: AuditEvent):
        """Trigger real-time monitoring for critical events"""
        
        if event.severity == AuditSeverity.CRITICAL or event.status == AuditStatus.FAILURE:
            # In production, this would trigger alerts, notifications, etc.
            logger.warning(f"Critical audit event detected: {event.event_id}")
    
    def _check_compliance_alerts(self, event: AuditEvent):
        """Check for compliance alerts based on event"""
        
        # Define compliance alert conditions
        alert_conditions = [
            event.event_type == AuditEventType.POLICY_VIOLATION,
            event.event_type == AuditEventType.AML_SCREENING and event.status == AuditStatus.FAILURE,
            event.event_type == AuditEventType.KYC_VERIFICATION and event.status == AuditStatus.FAILURE,
            event.severity == AuditSeverity.CRITICAL
        ]
        
        if any(alert_conditions):
            # In production, this would send compliance alerts
            logger.warning(f"Compliance alert triggered for event: {event.event_id}")
    
    def _auto_archive_events(self):
        """Auto-archive old events to maintain performance"""
        
        # Keep only recent events in memory
        cutoff_date = datetime.now() - timedelta(days=90)
        
        events_to_archive = [e for e in self.audit_events if e.timestamp < cutoff_date]
        self.audit_events = [e for e in self.audit_events if e.timestamp >= cutoff_date]
        
        # In production, archived events would be moved to long-term storage
        logger.info(f"Auto-archived {len(events_to_archive)} audit events")
    
    def _generate_compliance_summary(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """Generate compliance summary from events"""
        
        compliance_events = [e for e in events if e.compliance_relevant]
        
        return {
            'total_compliance_events': len(compliance_events),
            'compliance_violations': len([e for e in compliance_events if e.event_type == AuditEventType.POLICY_VIOLATION]),
            'kyc_events': len([e for e in compliance_events if e.event_type == AuditEventType.KYC_VERIFICATION]),
            'aml_events': len([e for e in compliance_events if e.event_type == AuditEventType.AML_SCREENING]),
            'suitability_events': len([e for e in compliance_events if e.event_type == AuditEventType.SUITABILITY_ASSESSMENT]),
            'regulatory_reports': len([e for e in compliance_events if e.event_type == AuditEventType.REGULATORY_REPORT]),
            'compliance_success_rate': len([e for e in compliance_events if e.status == AuditStatus.SUCCESS]) / max(len(compliance_events), 1) * 100
        }
    
    def _generate_timeline_analysis(self, events: List[AuditEvent]) -> Dict[str, Any]:
        """Generate timeline analysis from events"""
        
        if not events:
            return {}
        
        # Group events by day
        daily_events = {}
        for event in events:
            date_key = event.timestamp.date()
            daily_events[date_key] = daily_events.get(date_key, 0) + 1
        
        # Calculate statistics
        daily_counts = list(daily_events.values())
        
        return {
            'date_range': {
                'start': min(events, key=lambda x: x.timestamp).timestamp.date().isoformat(),
                'end': max(events, key=lambda x: x.timestamp).timestamp.date().isoformat()
            },
            'daily_average': sum(daily_counts) / len(daily_counts) if daily_counts else 0,
            'peak_day': max(daily_events.items(), key=lambda x: x[1]) if daily_events else None,
            'total_days': len(daily_events),
            'events_per_day': dict(sorted(daily_events.items()))
        }
    
    def _generate_audit_recommendations(self, events: List[AuditEvent], query: AuditQuery) -> List[str]:
        """Generate audit recommendations based on analysis"""
        
        recommendations = []
        
        # Check for high failure rates
        failed_events = [e for e in events if e.status == AuditStatus.FAILURE]
        if len(failed_events) / max(len(events), 1) > 0.1:
            recommendations.append("High failure rate detected - review system processes and user training")
        
        # Check for critical events
        critical_events = [e for e in events if e.severity == AuditSeverity.CRITICAL]
        if critical_events:
            recommendations.append("Critical events detected - immediate review and remediation required")
        
        # Check for compliance violations
        violations = [e for e in events if e.event_type == AuditEventType.POLICY_VIOLATION]
        if violations:
            recommendations.append("Policy violations detected - review compliance procedures and controls")
        
        # Check for unusual activity patterns
        if len(events) > 1000:
            recommendations.append("High volume of audit events - consider implementing additional monitoring")
        
        return recommendations
    
    def _initialize_event_configurations(self) -> Dict[AuditEventType, Dict[str, Any]]:
        """Initialize event type configurations"""
        
        return {
            AuditEventType.USER_LOGIN: {
                'compliance_relevant': False,
                'retention_days': 365
            },
            AuditEventType.TRANSACTION_EXECUTED: {
                'compliance_relevant': True,
                'retention_days': 2555  # 7 years
            },
            AuditEventType.POLICY_VIOLATION: {
                'compliance_relevant': True,
                'retention_days': 2555
            },
            AuditEventType.KYC_VERIFICATION: {
                'compliance_relevant': True,
                'retention_days': 2555
            },
            AuditEventType.AML_SCREENING: {
                'compliance_relevant': True,
                'retention_days': 2555
            },
            AuditEventType.REGULATORY_REPORT: {
                'compliance_relevant': True,
                'retention_days': 2555
            },
            AuditEventType.DATA_EXPORT: {
                'compliance_relevant': True,
                'retention_days': 2555
            }
        }
    
    def _anonymize_user_id(self, user_id: Optional[str]) -> Optional[str]:
        """Anonymize user ID for export"""
        if user_id:
            return f"user_{hashlib.md5(user_id.encode()).hexdigest()[:8]}"
        return None
    
    def _anonymize_client_id(self, client_id: Optional[str]) -> Optional[str]:
        """Anonymize client ID for export"""
        if client_id:
            return f"client_{hashlib.md5(client_id.encode()).hexdigest()[:8]}"
        return None
    
    def _contains_sensitive_data(self, details: Dict[str, Any]) -> bool:
        """Check if event details contain sensitive data"""
        
        sensitive_keys = [
            'ssn', 'social_security', 'passport', 'account_number',
            'routing_number', 'credit_card', 'password', 'pin'
        ]
        
        details_str = json.dumps(details).lower()
        return any(key in details_str for key in sensitive_keys)

