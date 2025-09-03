from flask import Blueprint, request, jsonify
from sqlalchemy.exc import IntegrityError
from src.models.external_data import (
    MarketData, WeatherData, SatelliteData, NewsData,
    MarketDataCreate, WeatherDataCreate, SatelliteDataCreate,
    db
)
from datetime import datetime, date
import math

external_data_bp = Blueprint('external_data', __name__)

# Market Data Endpoints
@external_data_bp.route('/market-data', methods=['GET'])
def get_market_data():
    """Get market data with pagination and filtering"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 10, type=int), 100)
        symbol = request.args.get('symbol', '')
        start_date = request.args.get('start_date', '')
        end_date = request.args.get('end_date', '')
        data_source = request.args.get('data_source', '')
        
        query = MarketData.query
        
        # Apply filters
        if symbol:
            query = query.filter(MarketData.symbol == symbol.upper())
        
        if start_date:
            query = query.filter(MarketData.data_date >= datetime.fromisoformat(start_date).date())
            
        if end_date:
            query = query.filter(MarketData.data_date <= datetime.fromisoformat(end_date).date())
            
        if data_source:
            query = query.filter(MarketData.data_source == data_source)
        
        # Order by date (newest first)
        query = query.order_by(MarketData.data_date.desc())
        
        # Apply pagination
        pagination = query.paginate(
            page=page, 
            per_page=per_page, 
            error_out=False
        )
        
        market_data = [data.to_dict() for data in pagination.items]
        
        return jsonify({
            'market_data': market_data,
            'total': pagination.total,
            'page': page,
            'per_page': per_page,
            'pages': pagination.pages
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@external_data_bp.route('/market-data', methods=['POST'])
def create_market_data():
    """Create new market data entry"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate input data using Pydantic
        try:
            market_data_obj = MarketDataCreate(**data)
        except Exception as e:
            return jsonify({'error': f'Validation error: {str(e)}'}), 400
        
        # Create new market data entry
        market_data = MarketData(
            symbol=market_data_obj.symbol.upper(),
            data_date=datetime.fromisoformat(market_data_obj.data_date).date(),
            open_price=market_data_obj.ohlcv.open if market_data_obj.ohlcv else None,
            high_price=market_data_obj.ohlcv.high if market_data_obj.ohlcv else None,
            low_price=market_data_obj.ohlcv.low if market_data_obj.ohlcv else None,
            close_price=market_data_obj.ohlcv.close if market_data_obj.ohlcv else None,
            volume=market_data_obj.ohlcv.volume if market_data_obj.ohlcv else None,
            adjusted_close=market_data_obj.ohlcv.adjusted_close if market_data_obj.ohlcv else None,
            sma_20=market_data_obj.technical_indicators.sma_20 if market_data_obj.technical_indicators else None,
            sma_50=market_data_obj.technical_indicators.sma_50 if market_data_obj.technical_indicators else None,
            rsi=market_data_obj.technical_indicators.rsi if market_data_obj.technical_indicators else None,
            macd=market_data_obj.technical_indicators.macd.get('macd') if market_data_obj.technical_indicators and market_data_obj.technical_indicators.macd else None,
            macd_signal=market_data_obj.technical_indicators.macd.get('signal') if market_data_obj.technical_indicators and market_data_obj.technical_indicators.macd else None,
            macd_histogram=market_data_obj.technical_indicators.macd.get('histogram') if market_data_obj.technical_indicators and market_data_obj.technical_indicators.macd else None,
            pe_ratio=market_data_obj.fundamentals.pe_ratio if market_data_obj.fundamentals else None,
            pb_ratio=market_data_obj.fundamentals.pb_ratio if market_data_obj.fundamentals else None,
            dividend_yield=market_data_obj.fundamentals.dividend_yield if market_data_obj.fundamentals else None,
            market_cap=market_data_obj.fundamentals.market_cap if market_data_obj.fundamentals else None,
            data_source=market_data_obj.data_source
        )
        
        db.session.add(market_data)
        db.session.commit()
        
        return jsonify(market_data.to_dict()), 201
        
    except IntegrityError as e:
        db.session.rollback()
        return jsonify({'error': 'Market data for this symbol and date already exists'}), 409
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

@external_data_bp.route('/market-data/<symbol>/latest', methods=['GET'])
def get_latest_market_data(symbol):
    """Get the latest market data for a specific symbol"""
    try:
        market_data = MarketData.query.filter(
            MarketData.symbol == symbol.upper()
        ).order_by(MarketData.data_date.desc()).first()
        
        if not market_data:
            return jsonify({'error': 'No market data found for this symbol'}), 404
        
        return jsonify(market_data.to_dict()), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Weather Data Endpoints
@external_data_bp.route('/weather-data', methods=['GET'])
def get_weather_data():
    """Get weather data with pagination and filtering"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 10, type=int), 100)
        location_name = request.args.get('location_name', '')
        start_date = request.args.get('start_date', '')
        end_date = request.args.get('end_date', '')
        
        query = WeatherData.query
        
        # Apply filters
        if location_name:
            query = query.filter(WeatherData.location_name.contains(location_name))
        
        if start_date:
            query = query.filter(WeatherData.recorded_at >= datetime.fromisoformat(start_date))
            
        if end_date:
            query = query.filter(WeatherData.recorded_at <= datetime.fromisoformat(end_date))
        
        # Order by recorded time (newest first)
        query = query.order_by(WeatherData.recorded_at.desc())
        
        # Apply pagination
        pagination = query.paginate(
            page=page, 
            per_page=per_page, 
            error_out=False
        )
        
        weather_data = [data.to_dict() for data in pagination.items]
        
        return jsonify({
            'weather_data': weather_data,
            'total': pagination.total,
            'page': page,
            'per_page': per_page,
            'pages': pagination.pages
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@external_data_bp.route('/weather-data', methods=['POST'])
def create_weather_data():
    """Create new weather data entry"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate input data using Pydantic
        try:
            weather_data_obj = WeatherDataCreate(**data)
        except Exception as e:
            return jsonify({'error': f'Validation error: {str(e)}'}), 400
        
        # Create new weather data entry
        weather_data = WeatherData(
            latitude=weather_data_obj.location.latitude,
            longitude=weather_data_obj.location.longitude,
            location_name=weather_data_obj.location.name,
            temperature=weather_data_obj.weather.temperature,
            humidity=weather_data_obj.weather.humidity,
            pressure=weather_data_obj.weather.pressure,
            wind_speed=weather_data_obj.weather.wind_speed,
            wind_direction=weather_data_obj.weather.wind_direction,
            visibility=weather_data_obj.weather.visibility,
            uv_index=weather_data_obj.weather.uv_index,
            weather_condition=weather_data_obj.weather.condition,
            condition_description=weather_data_obj.weather.description,
            air_quality_index=weather_data_obj.environmental.air_quality_index if weather_data_obj.environmental else None,
            precipitation=weather_data_obj.environmental.precipitation if weather_data_obj.environmental else None,
            cloud_cover=weather_data_obj.environmental.cloud_cover if weather_data_obj.environmental else None,
            recorded_at=datetime.fromisoformat(weather_data_obj.recorded_at),
            data_source=weather_data_obj.data_source
        )
        
        db.session.add(weather_data)
        db.session.commit()
        
        return jsonify(weather_data.to_dict()), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# Satellite Data Endpoints
@external_data_bp.route('/satellite-data', methods=['GET'])
def get_satellite_data():
    """Get satellite data with pagination and filtering"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 10, type=int), 100)
        data_type = request.args.get('data_type', '')
        location_name = request.args.get('location_name', '')
        start_date = request.args.get('start_date', '')
        end_date = request.args.get('end_date', '')
        has_change = request.args.get('has_change', '')
        
        query = SatelliteData.query
        
        # Apply filters
        if data_type:
            query = query.filter(SatelliteData.data_type == data_type)
            
        if location_name:
            query = query.filter(SatelliteData.location_name.contains(location_name))
        
        if start_date:
            query = query.filter(SatelliteData.image_date >= datetime.fromisoformat(start_date).date())
            
        if end_date:
            query = query.filter(SatelliteData.image_date <= datetime.fromisoformat(end_date).date())
            
        if has_change:
            query = query.filter(SatelliteData.has_change == (has_change.lower() == 'true'))
        
        # Order by image date (newest first)
        query = query.order_by(SatelliteData.image_date.desc())
        
        # Apply pagination
        pagination = query.paginate(
            page=page, 
            per_page=per_page, 
            error_out=False
        )
        
        satellite_data = [data.to_dict() for data in pagination.items]
        
        return jsonify({
            'satellite_data': satellite_data,
            'total': pagination.total,
            'page': page,
            'per_page': per_page,
            'pages': pagination.pages
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@external_data_bp.route('/satellite-data', methods=['POST'])
def create_satellite_data():
    """Create new satellite data entry"""
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Validate input data using Pydantic
        try:
            satellite_data_obj = SatelliteDataCreate(**data)
        except Exception as e:
            return jsonify({'error': f'Validation error: {str(e)}'}), 400
        
        # Create new satellite data entry
        satellite_data = SatelliteData(
            latitude=satellite_data_obj.location.latitude,
            longitude=satellite_data_obj.location.longitude,
            location_name=satellite_data_obj.location.name,
            image_url=satellite_data_obj.image_metadata.url,
            image_date=datetime.fromisoformat(satellite_data_obj.image_metadata.date).date(),
            satellite_source=satellite_data_obj.image_metadata.satellite,
            resolution=satellite_data_obj.image_metadata.resolution,
            cloud_cover=satellite_data_obj.image_metadata.cloud_cover,
            data_type=satellite_data_obj.analysis.data_type,
            data_value=satellite_data_obj.analysis.value,
            has_change=satellite_data_obj.analysis.change_detection.has_change if satellite_data_obj.analysis.change_detection else False,
            change_type=satellite_data_obj.analysis.change_detection.type if satellite_data_obj.analysis.change_detection else None,
            change_confidence=satellite_data_obj.analysis.change_detection.confidence if satellite_data_obj.analysis.change_detection else None,
            related_commodities=satellite_data_obj.related_assets.commodities if satellite_data_obj.related_assets else None,
            affected_companies=satellite_data_obj.related_assets.companies if satellite_data_obj.related_assets else None,
            additional_metadata=satellite_data_obj.metadata,
            data_source=satellite_data_obj.data_source
        )
        
        db.session.add(satellite_data)
        db.session.commit()
        
        return jsonify(satellite_data.to_dict()), 201
        
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

# News Data Endpoints
@external_data_bp.route('/news-data', methods=['GET'])
def get_news_data():
    """Get news data with pagination and filtering"""
    try:
        page = request.args.get('page', 1, type=int)
        per_page = min(request.args.get('per_page', 10, type=int), 100)
        category = request.args.get('category', '')
        symbol = request.args.get('symbol', '')
        sentiment = request.args.get('sentiment', '')
        start_date = request.args.get('start_date', '')
        end_date = request.args.get('end_date', '')
        
        query = NewsData.query
        
        # Apply filters
        if category:
            query = query.filter(NewsData.category == category)
            
        if symbol:
            # Filter by related symbols (JSON contains)
            query = query.filter(NewsData.related_symbols.contains([symbol.upper()]))
            
        if sentiment:
            query = query.filter(NewsData.sentiment_label == sentiment)
        
        if start_date:
            query = query.filter(NewsData.published_at >= datetime.fromisoformat(start_date))
            
        if end_date:
            query = query.filter(NewsData.published_at <= datetime.fromisoformat(end_date))
        
        # Order by published date (newest first)
        query = query.order_by(NewsData.published_at.desc())
        
        # Apply pagination
        pagination = query.paginate(
            page=page, 
            per_page=per_page, 
            error_out=False
        )
        
        news_data = [data.to_dict() for data in pagination.items]
        
        return jsonify({
            'news_data': news_data,
            'total': pagination.total,
            'page': page,
            'per_page': per_page,
            'pages': pagination.pages
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Data Analytics Endpoints
@external_data_bp.route('/analytics/market-summary', methods=['GET'])
def get_market_summary():
    """Get market data summary and analytics"""
    try:
        symbol = request.args.get('symbol', '')
        days = request.args.get('days', 30, type=int)
        
        if not symbol:
            return jsonify({'error': 'Symbol parameter is required'}), 400
        
        # Get recent market data
        from datetime import timedelta
        start_date = datetime.utcnow().date() - timedelta(days=days)
        
        market_data = MarketData.query.filter(
            MarketData.symbol == symbol.upper(),
            MarketData.data_date >= start_date
        ).order_by(MarketData.data_date.asc()).all()
        
        if not market_data:
            return jsonify({'error': 'No market data found for this symbol'}), 404
        
        # Calculate analytics
        prices = [float(data.close_price) for data in market_data if data.close_price]
        volumes = [int(data.volume) for data in market_data if data.volume]
        
        if prices:
            current_price = prices[-1]
            price_change = prices[-1] - prices[0] if len(prices) > 1 else 0
            price_change_pct = (price_change / prices[0] * 100) if len(prices) > 1 and prices[0] > 0 else 0
            
            # Volatility calculation (standard deviation of returns)
            if len(prices) > 1:
                returns = [(prices[i] - prices[i-1]) / prices[i-1] for i in range(1, len(prices))]
                avg_return = sum(returns) / len(returns)
                volatility = (sum([(r - avg_return) ** 2 for r in returns]) / len(returns)) ** 0.5 * 100
            else:
                volatility = 0
        else:
            current_price = price_change = price_change_pct = volatility = 0
        
        avg_volume = sum(volumes) / len(volumes) if volumes else 0
        
        summary = {
            'symbol': symbol.upper(),
            'period_days': days,
            'data_points': len(market_data),
            'current_price': current_price,
            'price_change': price_change,
            'price_change_pct': price_change_pct,
            'volatility': volatility,
            'average_volume': avg_volume,
            'latest_data': market_data[-1].to_dict() if market_data else None
        }
        
        return jsonify(summary), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@external_data_bp.route('/analytics/weather-impact', methods=['GET'])
def get_weather_impact():
    """Get weather impact analysis for commodities"""
    try:
        commodity = request.args.get('commodity', '')
        location = request.args.get('location', '')
        days = request.args.get('days', 7, type=int)
        
        # Get recent weather data
        from datetime import timedelta
        start_date = datetime.utcnow() - timedelta(days=days)
        
        query = WeatherData.query.filter(WeatherData.recorded_at >= start_date)
        
        if location:
            query = query.filter(WeatherData.location_name.contains(location))
        
        weather_data = query.order_by(WeatherData.recorded_at.desc()).all()
        
        # Analyze weather patterns
        temperatures = [float(data.temperature) for data in weather_data if data.temperature]
        precipitation = [float(data.precipitation) for data in weather_data if data.precipitation]
        
        avg_temp = sum(temperatures) / len(temperatures) if temperatures else 0
        total_precipitation = sum(precipitation) if precipitation else 0
        
        # Simple impact assessment (this would be more sophisticated in production)
        impact_score = 0
        impact_factors = []
        
        if commodity.lower() in ['corn', 'wheat', 'soybeans']:
            if avg_temp > 35:  # Too hot
                impact_score -= 0.3
                impact_factors.append("High temperatures may stress crops")
            elif avg_temp < 10:  # Too cold
                impact_score -= 0.2
                impact_factors.append("Low temperatures may slow growth")
            
            if total_precipitation > 100:  # Too much rain
                impact_score -= 0.2
                impact_factors.append("Excessive rainfall may cause flooding")
            elif total_precipitation < 10:  # Too little rain
                impact_score -= 0.4
                impact_factors.append("Insufficient rainfall may cause drought stress")
        
        impact_analysis = {
            'commodity': commodity,
            'location': location,
            'period_days': days,
            'weather_summary': {
                'average_temperature': avg_temp,
                'total_precipitation': total_precipitation,
                'data_points': len(weather_data)
            },
            'impact_score': max(-1, min(1, impact_score)),  # Clamp between -1 and 1
            'impact_factors': impact_factors,
            'recommendation': 'Monitor closely' if abs(impact_score) > 0.2 else 'Normal conditions'
        }
        
        return jsonify(impact_analysis), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@external_data_bp.route('/external-data/stats', methods=['GET'])
def get_external_data_stats():
    """Get external data statistics"""
    try:
        # Market data stats
        total_market_data = MarketData.query.count()
        unique_symbols = db.session.query(MarketData.symbol).distinct().count()
        
        # Weather data stats
        total_weather_data = WeatherData.query.count()
        unique_locations = db.session.query(WeatherData.location_name).distinct().count()
        
        # Satellite data stats
        total_satellite_data = SatelliteData.query.count()
        change_detections = SatelliteData.query.filter(SatelliteData.has_change == True).count()
        
        # News data stats
        total_news_data = NewsData.query.count()
        
        # Recent data (last 24 hours)
        from datetime import timedelta
        recent_cutoff = datetime.utcnow() - timedelta(hours=24)
        
        recent_market_data = MarketData.query.filter(MarketData.created_at >= recent_cutoff).count()
        recent_weather_data = WeatherData.query.filter(WeatherData.created_at >= recent_cutoff).count()
        recent_satellite_data = SatelliteData.query.filter(SatelliteData.created_at >= recent_cutoff).count()
        recent_news_data = NewsData.query.filter(NewsData.created_at >= recent_cutoff).count()
        
        stats = {
            'market_data': {
                'total': total_market_data,
                'unique_symbols': unique_symbols,
                'recent_24h': recent_market_data
            },
            'weather_data': {
                'total': total_weather_data,
                'unique_locations': unique_locations,
                'recent_24h': recent_weather_data
            },
            'satellite_data': {
                'total': total_satellite_data,
                'change_detections': change_detections,
                'recent_24h': recent_satellite_data
            },
            'news_data': {
                'total': total_news_data,
                'recent_24h': recent_news_data
            }
        }
        
        return jsonify(stats), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

