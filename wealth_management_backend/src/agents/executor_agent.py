import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from .base_agent import BaseAgent, AgentType, AgentMessage, AgentResponse, AgentCapability
from src.models.event import Event, Proposal
from src.models.client import Client
from src.models.portfolio import Portfolio, Holding, AssetClass
from src.models.user import db
import logging

logger = logging.getLogger(__name__)

class ExecutorAgent(BaseAgent):
    """
    Executor Agent - Responsible for executing approved proposals and updating portfolios
    
    Capabilities:
    1. Execute approved trade proposals
    2. Update portfolio holdings and valuations
    3. Record trade transactions
    4. Manage order routing and execution
    5. Handle partial fills and order management
    6. Update client portfolios in real-time
    """
    
    def __init__(self, agent_id: str = "executor_001"):
        capabilities = [
            AgentCapability(
                name="trade_execution",
                description="Execute approved trades and update portfolios",
                input_schema={
                    "type": "object",
                    "properties": {
                        "proposal_id": {"type": "string"},
                        "execution_strategy": {"type": "string"}
                    }
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "execution_results": {"type": "array"},
                        "portfolio_updates": {"type": "object"},
                        "execution_summary": {"type": "object"}
                    }
                }
            ),
            AgentCapability(
                name="portfolio_rebalancing",
                description="Execute portfolio rebalancing operations",
                input_schema={
                    "type": "object",
                    "properties": {
                        "portfolio_id": {"type": "string"},
                        "target_allocation": {"type": "object"}
                    }
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "rebalancing_trades": {"type": "array"},
                        "new_allocation": {"type": "object"}
                    }
                }
            ),
            AgentCapability(
                name="order_management",
                description="Manage order lifecycle and execution monitoring",
                input_schema={
                    "type": "object",
                    "properties": {
                        "order_id": {"type": "string"},
                        "action": {"type": "string"}
                    }
                },
                output_schema={
                    "type": "object",
                    "properties": {
                        "order_status": {"type": "string"},
                        "execution_details": {"type": "object"}
                    }
                }
            )
        ]
        
        super().__init__(
            agent_id=agent_id,
            agent_type=AgentType.EXECUTOR,
            name="Executor Agent",
            description="Executes approved proposals and manages portfolio updates",
            capabilities=capabilities
        )
        
        # Execution parameters
        self.max_order_size = 10000  # Maximum single order size
        self.execution_timeout = 300  # 5 minutes timeout for execution
        self.slippage_tolerance = 0.005  # 0.5% slippage tolerance
        self.min_execution_amount = 100  # Minimum execution amount
    
    def get_system_prompt(self) -> str:
        return """You are the Executor Agent in a wealth management system. Your role is to execute approved investment proposals and manage portfolio updates.

Your responsibilities:
1. Execute approved trades efficiently and accurately
2. Update portfolio holdings and valuations in real-time
3. Manage order routing and execution strategies
4. Handle partial fills and order lifecycle management
5. Record all transactions with proper audit trails
6. Monitor execution quality and slippage

You have access to:
- Approved proposals and trade instructions
- Portfolio holdings and current positions
- Market data and execution venues
- Order management systems
- Transaction recording systems

Always prioritize execution quality, minimize market impact, and maintain accurate records of all transactions."""
    
    async def process_message(self, message: AgentMessage) -> AgentResponse:
        """Process incoming messages and route to appropriate handlers"""
        try:
            action = message.content.get("action")
            
            if action == "ping":
                return AgentResponse(success=True, data={"status": "pong"})
            
            elif action == "execute_proposal":
                return await self._execute_proposal(message.content)
            
            elif action == "execute_trades":
                return await self._execute_trades(message.content)
            
            elif action == "rebalance_portfolio":
                return await self._rebalance_portfolio(message.content)
            
            elif action == "update_portfolio":
                return await self._update_portfolio(message.content)
            
            elif action == "cancel_order":
                return await self._cancel_order(message.content)
            
            elif action == "get_execution_status":
                return await self._get_execution_status(message.content)
            
            else:
                return AgentResponse(
                    success=False,
                    error=f"Unknown action: {action}"
                )
                
        except Exception as e:
            logger.error(f"Executor agent error: {str(e)}")
            return AgentResponse(success=False, error=str(e))
    
    async def _execute_proposal(self, content: Dict[str, Any]) -> AgentResponse:
        """Execute an approved proposal"""
        try:
            proposal_id = content.get("proposal_id")
            execution_strategy = content.get("execution_strategy", "market")
            
            proposal = Proposal.query.get(proposal_id)
            if not proposal:
                return AgentResponse(success=False, error="Proposal not found")
            
            if proposal.status != ProposalStatus.APPROVED:
                return AgentResponse(success=False, error="Proposal not approved for execution")
            
            portfolio = Portfolio.query.get(proposal.portfolio_id)
            if not portfolio:
                return AgentResponse(success=False, error="Portfolio not found")
            
            # Execute all trades in the proposal
            execution_results = []
            total_executed_value = 0
            successful_trades = 0
            failed_trades = 0
            
            for trade in proposal.proposed_trades:
                try:
                    execution_result = await self._execute_single_trade(trade, portfolio, execution_strategy)
                    execution_results.append(execution_result)
                    
                    if execution_result["status"] == "EXECUTED":
                        successful_trades += 1
                        total_executed_value += execution_result.get("executed_value", 0)
                    else:
                        failed_trades += 1
                        
                except Exception as e:
                    logger.error(f"Trade execution error: {str(e)}")
                    execution_results.append({
                        "symbol": trade.get("symbol", "Unknown"),
                        "status": "FAILED",
                        "error": str(e)
                    })
                    failed_trades += 1
            
            # Update proposal status
            if failed_trades == 0:
                proposal.status = ProposalStatus.EXECUTED
            elif successful_trades > 0:
                proposal.status = ProposalStatus.PARTIALLY_EXECUTED
            else:
                proposal.status = ProposalStatus.EXECUTION_FAILED
            
            proposal.execution_results = {
                "execution_timestamp": datetime.utcnow().isoformat(),
                "total_trades": len(proposal.proposed_trades),
                "successful_trades": successful_trades,
                "failed_trades": failed_trades,
                "total_executed_value": total_executed_value,
                "execution_results": execution_results
            }
            proposal.updated_at = datetime.utcnow()
            
            # Update portfolio valuation
            await self._update_portfolio_valuation(portfolio)
            
            db.session.commit()
            
            return AgentResponse(
                success=True,
                data={
                    "proposal_id": proposal_id,
                    "execution_summary": {
                        "total_trades": len(proposal.proposed_trades),
                        "successful_trades": successful_trades,
                        "failed_trades": failed_trades,
                        "total_executed_value": total_executed_value,
                        "execution_status": proposal.status.value
                    },
                    "execution_results": execution_results
                }
            )
            
        except Exception as e:
            logger.error(f"Proposal execution error: {str(e)}")
            db.session.rollback()
            return AgentResponse(success=False, error=str(e))
    
    async def _execute_single_trade(
        self, 
        trade: Dict[str, Any], 
        portfolio: Portfolio, 
        execution_strategy: str
    ) -> Dict[str, Any]:
        """Execute a single trade"""
        try:
            symbol = trade.get("symbol", "")
            action = trade.get("action", "").lower()
            quantity = float(trade.get("quantity", 0))
            
            if not symbol or not action or quantity <= 0:
                return {
                    "symbol": symbol,
                    "status": "FAILED",
                    "error": "Invalid trade parameters"
                }
            
            # Get current market price (simulated)
            market_price = await self._get_market_price(symbol)
            if not market_price:
                return {
                    "symbol": symbol,
                    "status": "FAILED",
                    "error": "Unable to get market price"
                }
            
            # Calculate execution price with slippage
            if execution_strategy == "market":
                execution_price = market_price * (1 + self.slippage_tolerance if action == "buy" else 1 - self.slippage_tolerance)
            else:
                execution_price = market_price  # Simplified for other strategies
            
            executed_value = quantity * execution_price
            
            # Check minimum execution amount
            if executed_value < self.min_execution_amount:
                return {
                    "symbol": symbol,
                    "status": "FAILED",
                    "error": f"Execution value ${executed_value:.2f} below minimum ${self.min_execution_amount}"
                }
            
            # Update or create holding
            holding = Holding.query.filter(
                Holding.portfolio_id == portfolio.id,
                Holding.symbol == symbol
            ).first()
            
            if action == "buy":
                if holding:
                    # Update existing holding
                    new_quantity = float(holding.quantity or 0) + quantity
                    new_cost_basis = ((float(holding.cost_basis or 0) * float(holding.quantity or 0)) + 
                                    (execution_price * quantity)) / new_quantity
                    
                    holding.quantity = new_quantity
                    holding.cost_basis = new_cost_basis
                    holding.market_value = new_quantity * market_price
                    holding.updated_at = datetime.utcnow()
                else:
                    # Create new holding
                    holding = Holding(
                        portfolio_id=portfolio.id,
                        symbol=symbol,
                        quantity=quantity,
                        cost_basis=execution_price,
                        market_value=quantity * market_price,
                        asset_class=self._determine_asset_class(symbol),
                        sector=trade.get("sector", "Unknown"),
                        created_at=datetime.utcnow(),
                        updated_at=datetime.utcnow()
                    )
                    db.session.add(holding)
                    
            elif action == "sell":
                if not holding or float(holding.quantity or 0) < quantity:
                    return {
                        "symbol": symbol,
                        "status": "FAILED",
                        "error": "Insufficient shares to sell"
                    }
                
                # Update holding
                new_quantity = float(holding.quantity or 0) - quantity
                if new_quantity <= 0:
                    # Remove holding completely
                    db.session.delete(holding)
                else:
                    holding.quantity = new_quantity
                    holding.market_value = new_quantity * market_price
                    holding.updated_at = datetime.utcnow()
            
            # Record transaction (simplified - would use a proper transaction table)
            transaction_record = {
                "symbol": symbol,
                "action": action,
                "quantity": quantity,
                "execution_price": execution_price,
                "market_price": market_price,
                "executed_value": executed_value,
                "timestamp": datetime.utcnow().isoformat(),
                "execution_strategy": execution_strategy
            }
            
            return {
                "symbol": symbol,
                "status": "EXECUTED",
                "action": action,
                "quantity": quantity,
                "execution_price": execution_price,
                "executed_value": executed_value,
                "transaction": transaction_record
            }
            
        except Exception as e:
            logger.error(f"Single trade execution error: {str(e)}")
            return {
                "symbol": trade.get("symbol", "Unknown"),
                "status": "FAILED",
                "error": str(e)
            }
    
    async def _get_market_price(self, symbol: str) -> Optional[float]:
        """Get current market price for a symbol (simulated)"""
        try:
            # In a real system, this would connect to market data feeds
            # For simulation, we'll use some basic price generation
            
            # Get the latest market data from our database
            from src.models.external_data import MarketData
            latest_data = MarketData.query.filter(
                MarketData.symbol == symbol
            ).order_by(MarketData.data_date.desc()).first()
            
            if latest_data and latest_data.close_price:
                # Add some random variation to simulate real-time price movement
                import random
                base_price = float(latest_data.close_price)
                variation = random.uniform(-0.02, 0.02)  # ±2% variation
                return base_price * (1 + variation)
            
            # Fallback to default prices for common symbols
            default_prices = {
                "AAPL": 150.0,
                "MSFT": 300.0,
                "GOOGL": 2500.0,
                "AMZN": 3000.0,
                "TSLA": 800.0,
                "SPY": 400.0,
                "QQQ": 350.0,
                "VTI": 200.0,
                "BND": 80.0,
                "GLD": 180.0
            }
            
            return default_prices.get(symbol, 100.0)  # Default to $100
            
        except Exception as e:
            logger.error(f"Market price lookup error for {symbol}: {str(e)}")
            return None
    
    def _determine_asset_class(self, symbol: str) -> AssetClass:
        """Determine asset class for a symbol (simplified)"""
        # In a real system, this would use a proper symbol lookup service
        bond_symbols = ["BND", "TLT", "IEF", "SHY", "LQD", "HYG"]
        commodity_symbols = ["GLD", "SLV", "USO", "UNG", "DBA"]
        reit_symbols = ["VNQ", "IYR", "REIT"]
        
        if symbol in bond_symbols:
            return AssetClass.BOND
        elif symbol in commodity_symbols:
            return AssetClass.COMMODITY
        elif symbol in reit_symbols:
            return AssetClass.REAL_ESTATE
        else:
            return AssetClass.EQUITY
    
    async def _update_portfolio_valuation(self, portfolio: Portfolio):
        """Update portfolio total valuation"""
        try:
            total_value = 0
            total_cost_basis = 0
            
            for holding in portfolio.holdings:
                if holding.market_value:
                    total_value += float(holding.market_value)
                if holding.cost_basis and holding.quantity:
                    total_cost_basis += float(holding.cost_basis) * float(holding.quantity)
            
            portfolio.total_value = total_value
            portfolio.total_cost_basis = total_cost_basis
            portfolio.unrealized_gain_loss = total_value - total_cost_basis
            portfolio.updated_at = datetime.utcnow()
            
            # Update performance metrics (simplified)
            if total_cost_basis > 0:
                portfolio.total_return_percentage = (total_value - total_cost_basis) / total_cost_basis
            
            logger.info(f"Updated portfolio {portfolio.id} valuation: ${total_value:.2f}")
            
        except Exception as e:
            logger.error(f"Portfolio valuation update error: {str(e)}")
    
    async def _execute_trades(self, content: Dict[str, Any]) -> AgentResponse:
        """Execute a list of trades"""
        try:
            trades = content.get("trades", [])
            portfolio_id = content.get("portfolio_id")
            execution_strategy = content.get("execution_strategy", "market")
            
            portfolio = Portfolio.query.get(portfolio_id)
            if not portfolio:
                return AgentResponse(success=False, error="Portfolio not found")
            
            execution_results = []
            for trade in trades:
                result = await self._execute_single_trade(trade, portfolio, execution_strategy)
                execution_results.append(result)
            
            # Update portfolio valuation
            await self._update_portfolio_valuation(portfolio)
            db.session.commit()
            
            return AgentResponse(
                success=True,
                data={
                    "execution_results": execution_results,
                    "portfolio_id": portfolio_id
                }
            )
            
        except Exception as e:
            db.session.rollback()
            return AgentResponse(success=False, error=str(e))
    
    async def _rebalance_portfolio(self, content: Dict[str, Any]) -> AgentResponse:
        """Execute portfolio rebalancing"""
        try:
            portfolio_id = content.get("portfolio_id")
            target_allocation = content.get("target_allocation", {})
            
            portfolio = Portfolio.query.get(portfolio_id)
            if not portfolio:
                return AgentResponse(success=False, error="Portfolio not found")
            
            # Calculate current allocation
            current_allocation = {}
            total_value = float(portfolio.total_value or 0)
            
            for holding in portfolio.holdings:
                asset_class = holding.asset_class.value if holding.asset_class else "Other"
                current_weight = float(holding.market_value or 0) / total_value if total_value > 0 else 0
                current_allocation[asset_class] = current_allocation.get(asset_class, 0) + current_weight
            
            # Generate rebalancing trades
            rebalancing_trades = []
            for asset_class, target_weight in target_allocation.items():
                current_weight = current_allocation.get(asset_class, 0)
                deviation = target_weight - current_weight
                
                if abs(deviation) > 0.01:  # 1% threshold
                    trade_value = deviation * total_value
                    
                    # Find representative symbol for asset class (simplified)
                    representative_symbols = {
                        "Equity": "VTI",
                        "Bond": "BND",
                        "Commodity": "GLD",
                        "Real Estate": "VNQ"
                    }
                    
                    symbol = representative_symbols.get(asset_class, "VTI")
                    market_price = await self._get_market_price(symbol)
                    
                    if market_price and abs(trade_value) >= self.min_execution_amount:
                        quantity = abs(trade_value) / market_price
                        action = "buy" if trade_value > 0 else "sell"
                        
                        trade = {
                            "symbol": symbol,
                            "action": action,
                            "quantity": quantity,
                            "asset_class": asset_class,
                            "rationale": f"Rebalancing {asset_class} from {current_weight:.2%} to {target_weight:.2%}"
                        }
                        
                        rebalancing_trades.append(trade)
            
            # Execute rebalancing trades
            execution_results = []
            for trade in rebalancing_trades:
                result = await self._execute_single_trade(trade, portfolio, "market")
                execution_results.append(result)
            
            # Update portfolio valuation
            await self._update_portfolio_valuation(portfolio)
            db.session.commit()
            
            return AgentResponse(
                success=True,
                data={
                    "rebalancing_trades": rebalancing_trades,
                    "execution_results": execution_results,
                    "current_allocation": current_allocation,
                    "target_allocation": target_allocation
                }
            )
            
        except Exception as e:
            db.session.rollback()
            return AgentResponse(success=False, error=str(e))
    
    async def _update_portfolio(self, content: Dict[str, Any]) -> AgentResponse:
        """Update portfolio holdings and valuations"""
        try:
            portfolio_id = content.get("portfolio_id")
            
            portfolio = Portfolio.query.get(portfolio_id)
            if not portfolio:
                return AgentResponse(success=False, error="Portfolio not found")
            
            # Update all holding market values with current prices
            updated_holdings = []
            for holding in portfolio.holdings:
                market_price = await self._get_market_price(holding.symbol)
                if market_price:
                    old_value = float(holding.market_value or 0)
                    new_value = float(holding.quantity or 0) * market_price
                    
                    holding.market_value = new_value
                    holding.updated_at = datetime.utcnow()
                    
                    updated_holdings.append({
                        "symbol": holding.symbol,
                        "old_value": old_value,
                        "new_value": new_value,
                        "change": new_value - old_value
                    })
            
            # Update portfolio valuation
            await self._update_portfolio_valuation(portfolio)
            db.session.commit()
            
            return AgentResponse(
                success=True,
                data={
                    "portfolio_id": portfolio_id,
                    "updated_holdings": updated_holdings,
                    "new_total_value": float(portfolio.total_value or 0)
                }
            )
            
        except Exception as e:
            db.session.rollback()
            return AgentResponse(success=False, error=str(e))
    
    async def _cancel_order(self, content: Dict[str, Any]) -> AgentResponse:
        """Cancel an order (placeholder for order management)"""
        try:
            order_id = content.get("order_id")
            
            # In a real system, this would cancel orders in the order management system
            # For now, return a placeholder response
            
            return AgentResponse(
                success=True,
                data={
                    "order_id": order_id,
                    "status": "CANCELLED",
                    "message": "Order cancellation requested"
                }
            )
            
        except Exception as e:
            return AgentResponse(success=False, error=str(e))
    
    async def _get_execution_status(self, content: Dict[str, Any]) -> AgentResponse:
        """Get execution status for a proposal or order"""
        try:
            proposal_id = content.get("proposal_id")
            
            if proposal_id:
                proposal = Proposal.query.get(proposal_id)
                if not proposal:
                    return AgentResponse(success=False, error="Proposal not found")
                
                return AgentResponse(
                    success=True,
                    data={
                        "proposal_id": proposal_id,
                        "status": proposal.status.value,
                        "execution_results": proposal.execution_results,
                        "updated_at": proposal.updated_at.isoformat() if proposal.updated_at else None
                    }
                )
            
            return AgentResponse(success=False, error="No proposal ID provided")
            
        except Exception as e:
            return AgentResponse(success=False, error=str(e))

