from flask import Blueprint, request, jsonify
from sqlalchemy.exc import IntegrityError
from src.models.portfolio import Portfolio, Holding, PortfolioCreate, PortfolioUpdate, HoldingCreate, HoldingUpdate, db
from src.models.client import Client
from datetime import datetime
import math

portfolio_bp = Blueprint('portfolio', __name__)

@portfolio_bp.route('/portfolios', methods=['GET'])
def get_portfolios():
    """Get all portfolios with pagination"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 10, type=int), 100)
        client_id = request.args.get('client_id', '')
        status = request.args.get('status', '')
        
        query = Portfolio.query
        
        # Apply filters
        if client_id:
            query = query.filter(Portfolio.client_id == client_id)
        
        if status:
            query = query.filter(Portfolio.status == status)
        
        # Apply pagination
        pagination = query.paginate(
            page=page, 
            per_page=per_page, 
            error_out=False
        )
        
        portfolios = [portfolio.to_dict() for portfolio in pagination.items]
        
        return jsonify({
            'portfolios': portfolios,
            'total': pagination.total,
            'page': page,
            'per_page': per_page,
            'pages': pagination.pages
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@portfolio_bp.route('/portfolios/<portfolio_id>', methods=['GET'])
def get_portfolio(portfolio_id):
    """Get a specific portfolio by ID"""
    try:
        portfolio = Portfolio.query.get(portfolio_id)
        if not portfolio:
            return jsonify({'error': 'Portfolio not found'}), 404
        
        portfolio_data = portfolio.to_dict()
        portfolio_data['holdings'] = [holding.to_dict() for holding in portfolio.holdings]
        
        return jsonify(portfolio_data), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@portfolio_bp.route('/portfolios', methods=['POST'])
def create_portfolio():
    """Create a new portfolio"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate input data using Pydantic
        try:
            portfolio_data = PortfolioCreate(**data)
        except Exception as e:
            return jsonify({'error': f'Validation error: {str(e)}'}), 400
        
        # Check if client exists
        client = Client.query.get(portfolio_data.client_id)
        if not client:
            return jsonify({'error': 'Client not found'}), 404
        
        # Create new portfolio
        portfolio = Portfolio(
            client_id=portfolio_data.client_id,
            portfolio_name=portfolio_data.portfolio_name,
            cash_balance=portfolio_data.cash_balance,
            ips_constraints=portfolio_data.ips_constraints.dict() if portfolio_data.ips_constraints else None
        )
        
        db.session.add(portfolio)
        db.session.commit()
        
        return jsonify(portfolio.to_dict()), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@portfolio_bp.route('/portfolios/<portfolio_id>/holdings', methods=['GET'])
def get_portfolio_holdings(portfolio_id):
    """Get all holdings for a specific portfolio"""
    try:
        portfolio = Portfolio.query.get(portfolio_id)
        if not portfolio:
            return jsonify({'error': 'Portfolio not found'}), 404
        
        holdings = [holding.to_dict() for holding in portfolio.holdings]
        
        return jsonify({
            'portfolio_id': portfolio_id,
            'holdings': holdings,
            'total': len(holdings)
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@portfolio_bp.route('/portfolios/<portfolio_id>/holdings', methods=['POST'])
def add_holding(portfolio_id):
    """Add a new holding to a portfolio"""
    try:
        portfolio = Portfolio.query.get(portfolio_id)
        if not portfolio:
            return jsonify({'error': 'Portfolio not found'}), 404
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Add portfolio_id to data
        data['portfolio_id'] = portfolio_id
        
        # Validate input data using Pydantic
        try:
            holding_data = HoldingCreate(**data)
        except Exception as e:
            return jsonify({'error': f'Validation error: {str(e)}'}), 400
        
        # Create new holding
        holding = Holding(
            portfolio_id=holding_data.portfolio_id,
            symbol=holding_data.symbol,
            company_name=holding_data.company_name,
            quantity=holding_data.quantity,
            average_cost=holding_data.average_cost,
            current_price=holding_data.current_price,
            asset_class=holding_data.asset_class,
            sector=holding_data.sector,
            industry=holding_data.industry,
            country=holding_data.country,
            esg_score=holding_data.esg_scores.overall if holding_data.esg_scores else None,
            environmental_score=holding_data.esg_scores.environmental if holding_data.esg_scores else None,
            social_score=holding_data.esg_scores.social if holding_data.esg_scores else None,
            governance_score=holding_data.esg_scores.governance if holding_data.esg_scores else None,
            tax_lots=[lot.dict() for lot in holding_data.tax_lots] if holding_data.tax_lots else None
        )
        
        # Calculate market value and unrealized gain/loss
        if holding.current_price and holding.quantity:
            holding.market_value = holding.current_price * holding.quantity
            holding.unrealized_gain_loss = holding.market_value - (holding.average_cost * holding.quantity)
        
        db.session.add(holding)
        
        # Update portfolio total value
        portfolio.total_value = (portfolio.total_value or 0) + (holding.market_value or 0)
        portfolio.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        return jsonify(holding.to_dict()), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@portfolio_bp.route('/holdings/<holding_id>', methods=['PUT'])
def update_holding(holding_id):
    """Update an existing holding"""
    try:
        holding = Holding.query.get(holding_id)
        if not holding:
            return jsonify({'error': 'Holding not found'}), 404
        
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate input data using Pydantic
        try:
            holding_data = HoldingUpdate(**data)
        except Exception as e:
            return jsonify({'error': f'Validation error: {str(e)}'}), 400
        
        # Store old market value for portfolio update
        old_market_value = holding.market_value or 0
        
        # Update holding fields
        update_data = holding_data.dict(exclude_unset=True)
        
        for field, value in update_data.items():
            if field == 'esg_scores' and value:
                holding.esg_score = value.get('overall', holding.esg_score)
                holding.environmental_score = value.get('environmental', holding.environmental_score)
                holding.social_score = value.get('social', holding.social_score)
                holding.governance_score = value.get('governance', holding.governance_score)
            elif field == 'tax_lots' and value:
                holding.tax_lots = [lot.dict() for lot in value]
            else:
                setattr(holding, field, value)
        
        # Recalculate market value and unrealized gain/loss
        if holding.current_price and holding.quantity:
            holding.market_value = holding.current_price * holding.quantity
            holding.unrealized_gain_loss = holding.market_value - (holding.average_cost * holding.quantity)
        
        holding.last_updated = datetime.utcnow()
        
        # Update portfolio total value
        portfolio = Portfolio.query.get(holding.portfolio_id)
        if portfolio:
            portfolio.total_value = (portfolio.total_value or 0) - old_market_value + (holding.market_value or 0)
            portfolio.updated_at = datetime.utcnow()
        
        db.session.commit()
        
        return jsonify(holding.to_dict()), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@portfolio_bp.route('/holdings/<holding_id>', methods=['DELETE'])
def delete_holding(holding_id):
    """Delete a holding from a portfolio"""
    try:
        holding = Holding.query.get(holding_id)
        if not holding:
            return jsonify({'error': 'Holding not found'}), 404
        
        # Update portfolio total value
        portfolio = Portfolio.query.get(holding.portfolio_id)
        if portfolio:
            portfolio.total_value = (portfolio.total_value or 0) - (holding.market_value or 0)
            portfolio.updated_at = datetime.utcnow()
        
        db.session.delete(holding)
        db.session.commit()
        
        return jsonify({'message': 'Holding deleted successfully'}), 200
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@portfolio_bp.route('/portfolios/<portfolio_id>/performance', methods=['GET'])
def get_portfolio_performance(portfolio_id):
    """Get portfolio performance metrics"""
    try:
        portfolio = Portfolio.query.get(portfolio_id)
        if not portfolio:
            return jsonify({'error': 'Portfolio not found'}), 404
        
        # Calculate current performance metrics
        holdings = portfolio.holdings
        total_market_value = sum(holding.market_value or 0 for holding in holdings)
        total_cost_basis = sum((holding.average_cost or 0) * (holding.quantity or 0) for holding in holdings)
        total_unrealized_gain_loss = sum(holding.unrealized_gain_loss or 0 for holding in holdings)
        
        # Asset allocation
        asset_allocation = {}
        for holding in holdings:
            asset_class = holding.asset_class.value if holding.asset_class else 'Unknown'
            asset_allocation[asset_class] = asset_allocation.get(asset_class, 0) + (holding.market_value or 0)
        
        # Convert to percentages
        if total_market_value > 0:
            asset_allocation = {k: (v / total_market_value) * 100 for k, v in asset_allocation.items()}
        
        # Sector allocation
        sector_allocation = {}
        for holding in holdings:
            sector = holding.sector or 'Unknown'
            sector_allocation[sector] = sector_allocation.get(sector, 0) + (holding.market_value or 0)
        
        # Convert to percentages
        if total_market_value > 0:
            sector_allocation = {k: (v / total_market_value) * 100 for k, v in sector_allocation.items()}
        
        performance_data = {
            'portfolio_id': portfolio_id,
            'total_value': total_market_value,
            'cost_basis': total_cost_basis,
            'unrealized_gain_loss': total_unrealized_gain_loss,
            'unrealized_return_pct': (total_unrealized_gain_loss / total_cost_basis * 100) if total_cost_basis > 0 else 0,
            'cash_balance': float(portfolio.cash_balance or 0),
            'asset_allocation': asset_allocation,
            'sector_allocation': sector_allocation,
            'performance_metrics': {
                'total_return': float(portfolio.total_return) if portfolio.total_return else None,
                'annualized_return': float(portfolio.annualized_return) if portfolio.annualized_return else None,
                'sharpe_ratio': float(portfolio.sharpe_ratio) if portfolio.sharpe_ratio else None,
                'max_drawdown': float(portfolio.max_drawdown) if portfolio.max_drawdown else None,
                'volatility': float(portfolio.volatility) if portfolio.volatility else None
            },
            'risk_metrics': {
                'beta': float(portfolio.beta) if portfolio.beta else None,
                'var_95': float(portfolio.var_95) if portfolio.var_95 else None,
                'expected_shortfall': float(portfolio.expected_shortfall) if portfolio.expected_shortfall else None,
                'concentration_risk': float(portfolio.concentration_risk) if portfolio.concentration_risk else None
            }
        }
        
        return jsonify(performance_data), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@portfolio_bp.route('/portfolios/stats', methods=['GET'])
def get_portfolio_stats():
    """Get portfolio statistics"""
    try:
        total_portfolios = Portfolio.query.count()
        active_portfolios = Portfolio.query.filter(Portfolio.status == 'Active').count()
        
        # Total AUM (Assets Under Management)
        total_aum = db.session.query(db.func.sum(Portfolio.total_value)).scalar() or 0
        
        # Average portfolio value
        avg_portfolio_value = total_aum / total_portfolios if total_portfolios > 0 else 0
        
        # Portfolio status distribution
        status_stats = db.session.query(
            Portfolio.status, 
            db.func.count(Portfolio.id)
        ).group_by(Portfolio.status).all()
        
        status_distribution = {status.value if status else 'Unknown': count for status, count in status_stats}
        
        return jsonify({
            'total_portfolios': total_portfolios,
            'active_portfolios': active_portfolios,
            'total_aum': float(total_aum),
            'average_portfolio_value': float(avg_portfolio_value),
            'status_distribution': status_distribution
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

