from flask import Blueprint, request, jsonify
from sqlalchemy.exc import IntegrityError
from src.models.client import Client, ClientCreate, ClientUpdate, ClientResponse, ClientListResponse, db
from src.models.user import db as user_db
from datetime import datetime
import math

client_bp = Blueprint('client', __name__)

@client_bp.route('/clients', methods=['GET'])
def get_clients():
    """Get all clients with pagination"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 10, type=int), 100)
        search = request.args.get('search', '')
        status = request.args.get('status', '')
        risk_tolerance = request.args.get('risk_tolerance', '')
        
        query = Client.query
        
        # Apply filters
        if search:
            query = query.filter(
                (Client.first_name.contains(search)) |
                (Client.last_name.contains(search)) |
                (Client.email.contains(search))
            )
        
        if status:
            query = query.filter(Client.status == status)
            
        if risk_tolerance:
            query = query.filter(Client.risk_tolerance == risk_tolerance)
        
        # Apply pagination
        pagination = query.paginate(
            page=page, 
            per_page=per_page, 
            error_out=False
        )
        
        clients = [client.to_dict() for client in pagination.items]
        
        return jsonify({
            'clients': clients,
            'total': pagination.total,
            'page': page,
            'per_page': per_page,
            'pages': pagination.pages
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@client_bp.route('/clients/<client_id>', methods=['GET'])
def get_client(client_id):
    """Get a specific client by ID"""
    try:
        client = Client.query.get(client_id)
        if not client:
            return jsonify({'error': 'Client not found'}), 404
        
        return jsonify(client.to_dict()), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@client_bp.route('/clients', methods=['POST'])
def create_client():
    """Create a new client"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate input data using Pydantic
        try:
            client_data = ClientCreate(**data)
        except Exception as e:
            return jsonify({'error': f'Validation error: {str(e)}'}), 400
        
        # Create new client
        client = Client(
            first_name=client_data.first_name,
            last_name=client_data.last_name,
            email=client_data.email,
            phone=client_data.phone,
            date_of_birth=datetime.fromisoformat(client_data.date_of_birth).date() if client_data.date_of_birth else None,
            risk_tolerance=client_data.risk_tolerance,
            net_worth=client_data.net_worth,
            annual_income=client_data.annual_income,
            investment_objectives=client_data.investment_objectives,
            esg_environmental=client_data.esg_preferences.environmental if client_data.esg_preferences else False,
            esg_social=client_data.esg_preferences.social if client_data.esg_preferences else False,
            esg_governance=client_data.esg_preferences.governance if client_data.esg_preferences else False,
            preferred_language=client_data.preferences.language if client_data.preferences else 'en',
            preferred_currency=client_data.preferences.currency if client_data.preferences else 'USD',
            communication_channels=client_data.preferences.communication_channels if client_data.preferences else ['email']
        )
        
        db.session.add(client)
        db.session.commit()
        
        return jsonify(client.to_dict()), 201
        
    except IntegrityError as e:
        db.session.rollback()
        return jsonify({'error': 'Email already exists'}), 409
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@client_bp.route('/clients/<client_id>', methods=['PUT'])
def update_client(client_id):
    """Update an existing client"""
    try:
        client = Client.query.get(client_id)
        if not client:
            return jsonify({'error': 'Client not found'}), 404
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate input data using Pydantic
        try:
            client_data = ClientUpdate(**data)
        except Exception as e:
            return jsonify({'error': f'Validation error: {str(e)}'}), 400
        
        # Update client fields
        update_data = client_data.dict(exclude_unset=True)
        
        for field, value in update_data.items():
            if field == 'date_of_birth' and value:
                setattr(client, field, datetime.fromisoformat(value).date())
            elif field == 'esg_preferences' and value:
                client.esg_environmental = value.get('environmental', client.esg_environmental)
                client.esg_social = value.get('social', client.esg_social)
                client.esg_governance = value.get('governance', client.esg_governance)
            elif field == 'preferences' and value:
                client.preferred_language = value.get('language', client.preferred_language)
                client.preferred_currency = value.get('currency', client.preferred_currency)
                client.communication_channels = value.get('communication_channels', client.communication_channels)
            else:
                setattr(client, field, value)
        
        client.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify(client.to_dict()), 200
        
    except IntegrityError as e:
        db.session.rollback()
        return jsonify({'error': 'Email already exists'}), 409
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@client_bp.route('/clients/<client_id>', methods=['DELETE'])
def delete_client(client_id):
    """Delete a client (soft delete by setting is_active to False)"""
    try:
        client = Client.query.get(client_id)
        if not client:
            return jsonify({'error': 'Client not found'}), 404
        
        client.is_active = False
        client.updated_at = datetime.utcnow()
        db.session.commit()
        
        return jsonify({'message': 'Client deleted successfully'}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@client_bp.route('/clients/<client_id>/portfolios', methods=['GET'])
def get_client_portfolios(client_id):
    """Get all portfolios for a specific client"""
    try:
        client = Client.query.get(client_id)
        if not client:
            return jsonify({'error': 'Client not found'}), 404
        
        portfolios = [portfolio.to_dict() for portfolio in client.portfolios]
        
        return jsonify({
            'client_id': client_id,
            'portfolios': portfolios,
            'total': len(portfolios)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@client_bp.route('/clients/<client_id>/events', methods=['GET'])
def get_client_events(client_id):
    """Get all events for a specific client"""
    try:
        client = Client.query.get(client_id)
        if not client:
            return jsonify({'error': 'Client not found'}), 404
        
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 10, type=int), 100)
        status = request.args.get('status', '')
        event_type = request.args.get('event_type', '')
        
        query = client.events
        
        # Apply filters
        if status:
            query = [event for event in query if event.status.value == status]
        if event_type:
            query = [event for event in query if event.event_type.value == event_type]
        
        # Manual pagination for relationship
        total = len(query)
        start = (page - 1) * per_page
        end = start + per_page
        events = [event.to_dict() for event in query[start:end]]
        
        return jsonify({
            'client_id': client_id,
            'events': events,
            'total': total,
            'page': page,
            'per_page': per_page,
            'pages': math.ceil(total / per_page) if total > 0 else 1
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@client_bp.route('/clients/stats', methods=['GET'])
def get_client_stats():
    """Get client statistics"""
    try:
        total_clients = Client.query.filter(Client.is_active == True).count()
        active_clients = Client.query.filter(Client.status == 'Active').count()
        
        # Risk tolerance distribution
        risk_stats = db.session.query(
            Client.risk_tolerance, 
            db.func.count(Client.id)
        ).filter(Client.is_active == True).group_by(Client.risk_tolerance).all()
        
        risk_distribution = {risk.value if risk else 'Unknown': count for risk, count in risk_stats}
        
        # KYC/AML status
        kyc_pending = Client.query.filter(
            Client.kyc_status == 'Pending',
            Client.is_active == True
        ).count()
        
        aml_pending = Client.query.filter(
            Client.aml_status == 'Pending',
            Client.is_active == True
        ).count()
        
        return jsonify({
            'total_clients': total_clients,
            'active_clients': active_clients,
            'risk_distribution': risk_distribution,
            'compliance': {
                'kyc_pending': kyc_pending,
                'aml_pending': aml_pending
            }
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

