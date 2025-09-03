from flask import Blueprint, request, jsonify
from sqlalchemy.exc import IntegrityError
from src.models.event import Event, Proposal, EventCreate, EventUpdate, ProposalCreate, ProposalUpdate, db
from src.models.client import Client
from src.models.portfolio import Portfolio
from datetime import datetime
import math

event_bp = Blueprint('event', __name__)

@event_bp.route('/events', methods=['GET'])
def get_events():
    """Get all events with pagination and filtering"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 10, type=int), 100)
        client_id = request.args.get('client_id', '')
        event_type = request.args.get('event_type', '')
        status = request.args.get('status', '')
        severity = request.args.get('severity', '')
        
        query = Event.query
        
        # Apply filters
        if client_id:
            query = query.filter(Event.client_id == client_id)
        
        if event_type:
            query = query.filter(Event.event_type == event_type)
            
        if status:
            query = query.filter(Event.status == status)
            
        if severity:
            query = query.filter(Event.severity == severity)
        
        # Order by creation date (newest first)
        query = query.order_by(Event.created_at.desc())
        
        # Apply pagination
        pagination = query.paginate(
            page=page, 
            per_page=per_page, 
            error_out=False
        )
        
        events = [event.to_dict() for event in pagination.items]
        
        return jsonify({
            'events': events,
            'total': pagination.total,
            'page': page,
            'per_page': per_page,
            'pages': pagination.pages
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@event_bp.route('/events/<event_id>', methods=['GET'])
def get_event(event_id):
    """Get a specific event by ID"""
    try:
        event = Event.query.get(event_id)
        if not event:
            return jsonify({'error': 'Event not found'}), 404
        
        event_data = event.to_dict()
        event_data['proposals'] = [proposal.to_dict() for proposal in event.proposals]
        
        return jsonify(event_data), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@event_bp.route('/events', methods=['POST'])
def create_event():
    """Create a new event"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate input data using Pydantic
        try:
            event_data = EventCreate(**data)
        except Exception as e:
            return jsonify({'error': f'Validation error: {str(e)}'}), 400
        
        # Check if client exists (if client_id is provided)
        if event_data.client_id:
            client = Client.query.get(event_data.client_id)
            if not client:
                return jsonify({'error': 'Client not found'}), 404
        
        # Create new event
        event = Event(
            client_id=event_data.client_id,
            event_type=event_data.event_type,
            event_source=event_data.event_source,
            event_category=event_data.event_category,
            title=event_data.title,
            description=event_data.description,
            severity=event_data.severity,
            confidence=event_data.confidence,
            latitude=event_data.location.latitude if event_data.location else None,
            longitude=event_data.location.longitude if event_data.location else None,
            location_name=event_data.location.name if event_data.location else None,
            related_symbols=event_data.related_symbols,
            affected_sectors=event_data.affected_sectors,
            sentiment_score=event_data.sentiment.score if event_data.sentiment else None,
            sentiment_label=event_data.sentiment.label if event_data.sentiment else None,
            event_data=event_data.event_data,
            event_timestamp=datetime.fromisoformat(event_data.event_timestamp) if event_data.event_timestamp else None
        )
        
        db.session.add(event)
        db.session.commit()
        
        return jsonify(event.to_dict()), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@event_bp.route('/events/<event_id>', methods=['PUT'])
def update_event(event_id):
    """Update an existing event"""
    try:
        event = Event.query.get(event_id)
        if not event:
            return jsonify({'error': 'Event not found'}), 404
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate input data using Pydantic
        try:
            event_data = EventUpdate(**data)
        except Exception as e:
            return jsonify({'error': f'Validation error: {str(e)}'}), 400
        
        # Update event fields
        update_data = event_data.dict(exclude_unset=True)
        
        for field, value in update_data.items():
            if field == 'location' and value:
                event.latitude = value.get('latitude', event.latitude)
                event.longitude = value.get('longitude', event.longitude)
                event.location_name = value.get('name', event.location_name)
            elif field == 'sentiment' and value:
                event.sentiment_score = value.get('score', event.sentiment_score)
                event.sentiment_label = value.get('label', event.sentiment_label)
            elif field == 'processing_results' and value:
                event.processing_results = value.dict()
            else:
                setattr(event, field, value)
        
        event.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify(event.to_dict()), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@event_bp.route('/events/<event_id>/process', methods=['POST'])
def process_event(event_id):
    """Process an event through the workflow stages"""
    try:
        event = Event.query.get(event_id)
        if not event:
            return jsonify({'error': 'Event not found'}), 404
        
        data = request.get_json() or {}
        stage = data.get('stage', 'enrich')  # enrich, propose, check, approve, execute, narrate
        
        # Update event status based on stage
        status_mapping = {
            'enrich': 'ENRICHED',
            'propose': 'PROPOSED',
            'check': 'CHECKED',
            'approve': 'APPROVED',
            'execute': 'EXECUTED',
            'narrate': 'NARRATED'
        }
        
        if stage in status_mapping:
            event.status = status_mapping[stage]
            
            # Update processing results
            if not event.processing_results:
                event.processing_results = {}
            
            processing_results = event.processing_results.copy()
            
            if stage == 'enrich':
                processing_results['enriched'] = True
            elif stage == 'propose':
                processing_results['proposal_generated'] = True
            elif stage == 'check':
                processing_results['compliance_checked'] = True
            elif stage == 'approve':
                processing_results['approved'] = True
            elif stage == 'execute':
                processing_results['execution_completed'] = True
            elif stage == 'narrate':
                processing_results['narrative_generated'] = True
            
            event.processing_results = processing_results
            event.processed_at = datetime.utcnow()
            event.updated_at = datetime.utcnow()
            
            db.session.commit()
            
            return jsonify({
                'message': f'Event processed through {stage} stage',
                'event': event.to_dict()
            }), 200
        else:
            return jsonify({'error': 'Invalid processing stage'}), 400
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@event_bp.route('/proposals', methods=['GET'])
def get_proposals():
    """Get all proposals with pagination and filtering"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 10, type=int), 100)
        event_id = request.args.get('event_id', '')
        portfolio_id = request.args.get('portfolio_id', '')
        compliance_status = request.args.get('compliance_status', '')
        execution_status = request.args.get('execution_status', '')
        
        query = Proposal.query
        
        # Apply filters
        if event_id:
            query = query.filter(Proposal.event_id == event_id)
        
        if portfolio_id:
            query = query.filter(Proposal.portfolio_id == portfolio_id)
            
        if compliance_status:
            query = query.filter(Proposal.compliance_status == compliance_status)
            
        if execution_status:
            query = query.filter(Proposal.execution_status == execution_status)
        
        # Order by creation date (newest first)
        query = query.order_by(Proposal.created_at.desc())
        
        # Apply pagination
        pagination = query.paginate(
            page=page, 
            per_page=per_page, 
            error_out=False
        )
        
        proposals = [proposal.to_dict() for proposal in pagination.items]
        
        return jsonify({
            'proposals': proposals,
            'total': pagination.total,
            'page': page,
            'per_page': per_page,
            'pages': pagination.pages
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@event_bp.route('/proposals', methods=['POST'])
def create_proposal():
    """Create a new proposal"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate input data using Pydantic
        try:
            proposal_data = ProposalCreate(**data)
        except Exception as e:
            return jsonify({'error': f'Validation error: {str(e)}'}), 400
        
        # Check if event and portfolio exist
        event = Event.query.get(proposal_data.event_id)
        if not event:
            return jsonify({'error': 'Event not found'}), 404
            
        portfolio = Portfolio.query.get(proposal_data.portfolio_id)
        if not portfolio:
            return jsonify({'error': 'Portfolio not found'}), 404
        
        # Create new proposal
        proposal = Proposal(
            event_id=proposal_data.event_id,
            portfolio_id=proposal_data.portfolio_id,
            proposal_type=proposal_data.proposal_type,
            objective=proposal_data.objective,
            proposed_trades=[trade.dict() for trade in proposal_data.proposed_trades],
            expected_impact=proposal_data.expected_impact.dict() if proposal_data.expected_impact else None,
            risk_assessment=proposal_data.risk_assessment
        )
        
        db.session.add(proposal)
        db.session.commit()
        
        return jsonify(proposal.to_dict()), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@event_bp.route('/proposals/<proposal_id>/approve', methods=['POST'])
def approve_proposal(proposal_id):
    """Approve or reject a proposal"""
    try:
        proposal = Proposal.query.get(proposal_id)
        if not proposal:
            return jsonify({'error': 'Proposal not found'}), 404
        
        data = request.get_json() or {}
        approved = data.get('approved', True)
        approved_by = data.get('approved_by', 'System')
        rejection_reason = data.get('rejection_reason', '')
        
        if approved:
            proposal.approved_by = approved_by
            proposal.approval_timestamp = datetime.utcnow()
            proposal.execution_status = 'Ready'
        else:
            proposal.rejection_reason = rejection_reason
            proposal.execution_status = 'Rejected'
        
        proposal.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({
            'message': f'Proposal {"approved" if approved else "rejected"}',
            'proposal': proposal.to_dict()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@event_bp.route('/proposals/<proposal_id>/execute', methods=['POST'])
def execute_proposal(proposal_id):
    """Execute an approved proposal"""
    try:
        proposal = Proposal.query.get(proposal_id)
        if not proposal:
            return jsonify({'error': 'Proposal not found'}), 404
        
        if not proposal.approved_by:
            return jsonify({'error': 'Proposal not approved'}), 400
        
        data = request.get_json() or {}
        execution_results = data.get('execution_results', {})
        
        proposal.execution_status = 'Executed'
        proposal.executed_at = datetime.utcnow()
        proposal.execution_results = execution_results
        proposal.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        return jsonify({
            'message': 'Proposal executed successfully',
            'proposal': proposal.to_dict()
        }), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@event_bp.route('/events/stats', methods=['GET'])
def get_event_stats():
    """Get event statistics"""
    try:
        total_events = Event.query.count()
        
        # Status distribution
        status_stats = db.session.query(
            Event.status, 
            db.func.count(Event.id)
        ).group_by(Event.status).all()
        
        status_distribution = {status.value if status else 'Unknown': count for status, count in status_stats}
        
        # Event type distribution
        type_stats = db.session.query(
            Event.event_type, 
            db.func.count(Event.id)
        ).group_by(Event.event_type).all()
        
        type_distribution = {event_type.value if event_type else 'Unknown': count for event_type, count in type_stats}
        
        # Severity distribution
        severity_stats = db.session.query(
            Event.severity, 
            db.func.count(Event.id)
        ).group_by(Event.severity).all()
        
        severity_distribution = {severity.value if severity else 'Unknown': count for severity, count in severity_stats}
        
        # Recent events (last 7 days)
        from datetime import timedelta
        recent_events = Event.query.filter(
            Event.created_at >= datetime.utcnow() - timedelta(days=7)
        ).count()
        
        return jsonify({
            'total_events': total_events,
            'recent_events': recent_events,
            'status_distribution': status_distribution,
            'type_distribution': type_distribution,
            'severity_distribution': severity_distribution
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

