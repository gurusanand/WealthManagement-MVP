from flask import Blueprint, request, jsonify
from pydantic import BaseModel, ValidationError
from typing import Dict, List, Optional, Any
from datetime import datetime
import asyncio
import json

from src.agents.orchestrator import orchestrator
from src.agents.base_agent import agent_registry
from src.models.event import Event
from src.models.portfolio import Portfolio
from src.models.client import Client

agents_bp = Blueprint('agents', __name__)

# Pydantic models for API validation
class WorkflowRequest(BaseModel):
    event_id: Optional[str] = None
    portfolio_id: Optional[str] = None
    client_id: Optional[str] = None
    workflow_type: str = "event_processing"
    parameters: Optional[Dict[str, Any]] = {}

class AgentMessageRequest(BaseModel):
    agent_name: str
    action: str
    content: Dict[str, Any]

class CustomWorkflowRequest(BaseModel):
    workflow_type: str
    steps: List[Dict[str, Any]]
    context: Optional[Dict[str, Any]] = {}

@agents_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for the agent system"""
    try:
        return jsonify({
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "orchestrator_running": orchestrator.running,
            "registered_agents": len(agent_registry.get_all_agents())
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@agents_bp.route('/status', methods=['GET'])
def get_system_status():
    """Get comprehensive system status"""
    try:
        # Run async function in sync context
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            agent_statuses = loop.run_until_complete(orchestrator.get_agent_status())
            health_results = loop.run_until_complete(agent_registry.health_check_all())
        finally:
            loop.close()
        
        return jsonify({
            "orchestrator": {
                "running": orchestrator.running,
                "active_workflows": len(orchestrator.workflows),
                "message_queue_size": orchestrator.message_queue.qsize()
            },
            "agents": agent_statuses,
            "health_check": health_results,
            "timestamp": datetime.utcnow().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@agents_bp.route('/workflows', methods=['POST'])
def start_workflow():
    """Start a new workflow"""
    try:
        # Validate request
        try:
            workflow_request = WorkflowRequest(**request.json)
        except ValidationError as e:
            return jsonify({"error": "Invalid request", "details": e.errors()}), 400
        
        # Determine event_id based on request
        event_id = workflow_request.event_id
        
        if not event_id and workflow_request.client_id:
            # Create a general event for the client
            client = Client.query.get(workflow_request.client_id)
            if not client:
                return jsonify({"error": "Client not found"}), 404
            
            # For demo purposes, create a placeholder event
            # In practice, this would be triggered by actual events
            event_id = "demo_event_" + workflow_request.client_id
        
        if not event_id:
            return jsonify({"error": "event_id is required"}), 400
        
        # Start workflow asynchronously
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            workflow_id = loop.run_until_complete(
                orchestrator.process_event(event_id, workflow_request.workflow_type)
            )
        finally:
            loop.close()
        
        return jsonify({
            "workflow_id": workflow_id,
            "status": "started",
            "event_id": event_id,
            "workflow_type": workflow_request.workflow_type
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@agents_bp.route('/workflows/<workflow_id>', methods=['GET'])
def get_workflow_status(workflow_id):
    """Get workflow status"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            status = loop.run_until_complete(orchestrator.get_workflow_status(workflow_id))
        finally:
            loop.close()
        
        if status:
            return jsonify(status)
        else:
            return jsonify({"error": "Workflow not found"}), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@agents_bp.route('/workflows/<workflow_id>', methods=['DELETE'])
def cancel_workflow(workflow_id):
    """Cancel a running workflow"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            cancelled = loop.run_until_complete(orchestrator.cancel_workflow(workflow_id))
        finally:
            loop.close()
        
        if cancelled:
            return jsonify({"message": "Workflow cancelled", "workflow_id": workflow_id})
        else:
            return jsonify({"error": "Workflow not found or cannot be cancelled"}), 404
            
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@agents_bp.route('/workflows/custom', methods=['POST'])
def create_custom_workflow():
    """Create a custom workflow"""
    try:
        # Validate request
        try:
            workflow_request = CustomWorkflowRequest(**request.json)
        except ValidationError as e:
            return jsonify({"error": "Invalid request", "details": e.errors()}), 400
        
        # Create custom workflow
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            workflow_id = loop.run_until_complete(
                orchestrator.create_custom_workflow(
                    workflow_request.workflow_type,
                    workflow_request.steps,
                    workflow_request.context
                )
            )
        finally:
            loop.close()
        
        return jsonify({
            "workflow_id": workflow_id,
            "status": "started",
            "workflow_type": workflow_request.workflow_type
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@agents_bp.route('/agents/<agent_name>/message', methods=['POST'])
def send_agent_message(agent_name):
    """Send a direct message to an agent"""
    try:
        # Validate request
        try:
            message_request = AgentMessageRequest(**request.json)
        except ValidationError as e:
            return jsonify({"error": "Invalid request", "details": e.errors()}), 400
        
        # Send message to agent
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            response = loop.run_until_complete(
                orchestrator.send_message_to_agent(
                    agent_name,
                    message_request.action,
                    message_request.content
                )
            )
        finally:
            loop.close()
        
        return jsonify({
            "success": response.success,
            "data": response.data,
            "error": response.error,
            "processing_time": response.processing_time
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@agents_bp.route('/agents', methods=['GET'])
def list_agents():
    """List all registered agents"""
    try:
        agents = agent_registry.get_all_agents()
        
        agent_list = []
        for agent in agents:
            agent_info = {
                "agent_id": agent.agent_id,
                "agent_type": agent.agent_type.value,
                "name": agent.name,
                "description": agent.description,
                "status": agent.status.value,
                "capabilities": [cap.name for cap in agent.capabilities],
                "processed_messages": agent.processed_messages,
                "error_count": agent.error_count,
                "created_at": agent.created_at.isoformat(),
                "last_activity": agent.last_activity.isoformat()
            }
            agent_list.append(agent_info)
        
        return jsonify({
            "agents": agent_list,
            "total_count": len(agent_list)
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@agents_bp.route('/agents/<agent_name>', methods=['GET'])
def get_agent_details(agent_name):
    """Get detailed information about a specific agent"""
    try:
        agent = orchestrator.agents.get(agent_name)
        if not agent:
            return jsonify({"error": "Agent not found"}), 404
        
        agent_details = {
            "agent_id": agent.agent_id,
            "agent_type": agent.agent_type.value,
            "name": agent.name,
            "description": agent.description,
            "status": agent.status.value,
            "capabilities": [
                {
                    "name": cap.name,
                    "description": cap.description,
                    "input_schema": cap.input_schema,
                    "output_schema": cap.output_schema
                }
                for cap in agent.capabilities
            ],
            "metrics": {
                "processed_messages": agent.processed_messages,
                "error_count": agent.error_count,
                "queue_size": agent.message_queue.qsize()
            },
            "timestamps": {
                "created_at": agent.created_at.isoformat(),
                "last_activity": agent.last_activity.isoformat()
            }
        }
        
        return jsonify(agent_details)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@agents_bp.route('/agents/health', methods=['GET'])
def check_agents_health():
    """Perform health check on all agents"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            health_results = loop.run_until_complete(agent_registry.health_check_all())
        finally:
            loop.close()
        
        healthy_count = sum(1 for healthy in health_results.values() if healthy)
        total_count = len(health_results)
        
        return jsonify({
            "overall_health": "healthy" if healthy_count == total_count else "degraded",
            "healthy_agents": healthy_count,
            "total_agents": total_count,
            "agent_health": health_results,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Event processing endpoints
@agents_bp.route('/events/<event_id>/process', methods=['POST'])
def process_event(event_id):
    """Process a specific event through the agent system"""
    try:
        # Check if event exists
        event = Event.query.get(event_id)
        if not event:
            return jsonify({"error": "Event not found"}), 404
        
        # Get workflow type from request or use default
        workflow_type = request.json.get("workflow_type", "event_processing") if request.json else "event_processing"
        
        # Start event processing workflow
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            workflow_id = loop.run_until_complete(
                orchestrator.process_event(event_id, workflow_type)
            )
        finally:
            loop.close()
        
        return jsonify({
            "workflow_id": workflow_id,
            "event_id": event_id,
            "status": "processing_started",
            "workflow_type": workflow_type
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Portfolio management endpoints
@agents_bp.route('/portfolios/<portfolio_id>/rebalance', methods=['POST'])
def rebalance_portfolio(portfolio_id):
    """Trigger portfolio rebalancing through the agent system"""
    try:
        # Check if portfolio exists
        portfolio = Portfolio.query.get(portfolio_id)
        if not portfolio:
            return jsonify({"error": "Portfolio not found"}), 404
        
        # Get target allocation from request
        target_allocation = request.json.get("target_allocation", {}) if request.json else {}
        
        # Create custom rebalancing workflow
        steps = [
            {"agent": "proposer", "action": "rebalance_recommendation"},
            {"agent": "checker", "action": "validate_proposal"},
            {"agent": "executor", "action": "rebalance_portfolio", "condition": "approved"},
            {"agent": "narrator", "action": "create_portfolio_report"}
        ]
        
        context = {
            "portfolio_id": portfolio_id,
            "target_allocation": target_allocation
        }
        
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            workflow_id = loop.run_until_complete(
                orchestrator.create_custom_workflow("portfolio_rebalancing", steps, context)
            )
        finally:
            loop.close()
        
        return jsonify({
            "workflow_id": workflow_id,
            "portfolio_id": portfolio_id,
            "status": "rebalancing_started",
            "target_allocation": target_allocation
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@agents_bp.route('/portfolios/<portfolio_id>/optimize', methods=['POST'])
def optimize_portfolio(portfolio_id):
    """Trigger portfolio optimization through the agent system"""
    try:
        # Check if portfolio exists
        portfolio = Portfolio.query.get(portfolio_id)
        if not portfolio:
            return jsonify({"error": "Portfolio not found"}), 404
        
        # Get optimization parameters
        objective = request.json.get("objective", "SharpeMax") if request.json else "SharpeMax"
        constraints = request.json.get("constraints", {}) if request.json else {}
        
        # Send direct message to proposer agent
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            response = loop.run_until_complete(
                orchestrator.send_message_to_agent(
                    "proposer",
                    "optimize_portfolio",
                    {
                        "portfolio_id": portfolio_id,
                        "objective": objective,
                        "constraints": constraints
                    }
                )
            )
        finally:
            loop.close()
        
        return jsonify({
            "success": response.success,
            "data": response.data,
            "error": response.error,
            "portfolio_id": portfolio_id,
            "objective": objective
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Client communication endpoints
@agents_bp.route('/clients/<client_id>/communicate', methods=['POST'])
def generate_client_communication(client_id):
    """Generate client communication through the narrator agent"""
    try:
        # Check if client exists
        client = Client.query.get(client_id)
        if not client:
            return jsonify({"error": "Client not found"}), 404
        
        # Get communication parameters
        communication_type = request.json.get("communication_type", "general") if request.json else "general"
        content = request.json.get("content", {}) if request.json else {}
        tone = request.json.get("tone", "professional") if request.json else "professional"
        
        # Send message to narrator agent
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            response = loop.run_until_complete(
                orchestrator.send_message_to_agent(
                    "narrator",
                    "generate_client_communication",
                    {
                        "client_id": client_id,
                        "communication_type": communication_type,
                        "content": content,
                        "tone": tone
                    }
                )
            )
        finally:
            loop.close()
        
        return jsonify({
            "success": response.success,
            "data": response.data,
            "error": response.error,
            "client_id": client_id,
            "communication_type": communication_type
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@agents_bp.route('/initialize', methods=['POST'])
def initialize_agent_system():
    """Initialize the agent system"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            # Initialize orchestrator if not already done
            if not orchestrator.agents:
                loop.run_until_complete(orchestrator.initialize())
            
            # Start orchestrator if not running
            if not orchestrator.running:
                loop.run_until_complete(orchestrator.start())
        finally:
            loop.close()
        
        return jsonify({
            "message": "Agent system initialized successfully",
            "agents_count": len(orchestrator.agents),
            "orchestrator_running": orchestrator.running,
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@agents_bp.route('/shutdown', methods=['POST'])
def shutdown_agent_system():
    """Shutdown the agent system"""
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        try:
            loop.run_until_complete(orchestrator.stop())
        finally:
            loop.close()
        
        return jsonify({
            "message": "Agent system shutdown successfully",
            "timestamp": datetime.utcnow().isoformat()
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

