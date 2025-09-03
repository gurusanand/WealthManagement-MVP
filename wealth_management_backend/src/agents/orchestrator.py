import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from .base_agent import BaseAgent, AgentType, AgentMessage, AgentResponse, AgentCapability, agent_registry
from .oracle_agent import OracleAgent
from .enricher_agent import EnricherAgent
from .proposer_agent import ProposerAgent
from .checker_agent import CheckerAgent
from .executor_agent import ExecutorAgent
from .narrator_agent import NarratorAgent
from src.models.event import Event, Proposal, EventStatus, ProposalStatus
from src.models.client import Client
from src.models.portfolio import Portfolio
from src.models.user import db
import logging

logger = logging.getLogger(__name__)

class WorkflowStatus:
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AgentOrchestrator:
    """
    Agent Orchestrator - Coordinates all agents and manages workflows
    
    Responsibilities:
    1. Initialize and manage all agents
    2. Coordinate multi-agent workflows
    3. Handle event-driven processing
    4. Manage agent communication and message routing
    5. Monitor agent health and performance
    6. Implement workflow patterns and business logic
    """
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.workflows: Dict[str, Dict[str, Any]] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.running = False
        
        # Workflow templates
        self.workflow_templates = {
            "event_processing": [
                {"agent": "oracle", "action": "detect_events"},
                {"agent": "enricher", "action": "enrich_event"},
                {"agent": "proposer", "action": "generate_proposal"},
                {"agent": "checker", "action": "validate_proposal"},
                {"agent": "executor", "action": "execute_proposal", "condition": "approved"},
                {"agent": "narrator", "action": "generate_client_communication"}
            ],
            "portfolio_rebalancing": [
                {"agent": "proposer", "action": "rebalance_recommendation"},
                {"agent": "checker", "action": "validate_proposal"},
                {"agent": "executor", "action": "rebalance_portfolio", "condition": "approved"},
                {"agent": "narrator", "action": "create_portfolio_report"}
            ],
            "client_onboarding": [
                {"agent": "proposer", "action": "generate_initial_allocation"},
                {"agent": "checker", "action": "validate_proposal"},
                {"agent": "executor", "action": "execute_proposal", "condition": "approved"},
                {"agent": "narrator", "action": "generate_welcome_communication"}
            ]
        }
    
    async def initialize(self):
        """Initialize all agents and register them"""
        try:
            # Create and register all agents
            oracle = OracleAgent("oracle_001")
            enricher = EnricherAgent("enricher_001")
            proposer = ProposerAgent("proposer_001")
            checker = CheckerAgent("checker_001")
            executor = ExecutorAgent("executor_001")
            narrator = NarratorAgent("narrator_001")
            
            # Register agents
            agent_registry.register_agent(oracle)
            agent_registry.register_agent(enricher)
            agent_registry.register_agent(proposer)
            agent_registry.register_agent(checker)
            agent_registry.register_agent(executor)
            agent_registry.register_agent(narrator)
            
            # Store local references
            self.agents = {
                "oracle": oracle,
                "enricher": enricher,
                "proposer": proposer,
                "checker": checker,
                "executor": executor,
                "narrator": narrator
            }
            
            logger.info("Agent orchestrator initialized with all agents")
            
        except Exception as e:
            logger.error(f"Agent orchestrator initialization error: {str(e)}")
            raise
    
    async def start(self):
        """Start the orchestrator and begin processing"""
        if self.running:
            return
        
        self.running = True
        logger.info("Agent orchestrator started")
        
        # Start background tasks
        asyncio.create_task(self._process_message_queue())
        asyncio.create_task(self._monitor_workflows())
        asyncio.create_task(self._health_check_agents())
    
    async def stop(self):
        """Stop the orchestrator and shutdown agents"""
        self.running = False
        
        # Shutdown all agents
        for agent in self.agents.values():
            await agent.shutdown()
        
        logger.info("Agent orchestrator stopped")
    
    async def process_event(self, event_id: str, workflow_type: str = "event_processing") -> str:
        """Process an event through the multi-agent workflow"""
        try:
            workflow_id = f"{workflow_type}_{event_id}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            # Create workflow instance
            workflow = {
                "id": workflow_id,
                "type": workflow_type,
                "event_id": event_id,
                "status": WorkflowStatus.PENDING,
                "steps": self.workflow_templates.get(workflow_type, []).copy(),
                "current_step": 0,
                "results": {},
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            self.workflows[workflow_id] = workflow
            
            # Start workflow execution
            await self._execute_workflow(workflow_id)
            
            return workflow_id
            
        except Exception as e:
            logger.error(f"Event processing error: {str(e)}")
            raise
    
    async def _execute_workflow(self, workflow_id: str):
        """Execute a workflow step by step"""
        try:
            workflow = self.workflows.get(workflow_id)
            if not workflow:
                logger.error(f"Workflow {workflow_id} not found")
                return
            
            workflow["status"] = WorkflowStatus.RUNNING
            workflow["updated_at"] = datetime.utcnow()
            
            logger.info(f"Starting workflow execution: {workflow_id}")
            
            while workflow["current_step"] < len(workflow["steps"]):
                step = workflow["steps"][workflow["current_step"]]
                
                # Check step condition if any
                if "condition" in step:
                    if not await self._check_step_condition(step["condition"], workflow):
                        logger.info(f"Workflow {workflow_id} step {workflow['current_step']} condition not met, skipping")
                        workflow["current_step"] += 1
                        continue
                
                # Execute step
                step_result = await self._execute_workflow_step(workflow_id, step)
                
                # Store step result
                step_key = f"step_{workflow['current_step']}_{step['agent']}_{step['action']}"
                workflow["results"][step_key] = step_result
                
                if not step_result.get("success", False):
                    workflow["status"] = WorkflowStatus.FAILED
                    workflow["error"] = step_result.get("error", "Step execution failed")
                    logger.error(f"Workflow {workflow_id} failed at step {workflow['current_step']}: {workflow['error']}")
                    return
                
                workflow["current_step"] += 1
                workflow["updated_at"] = datetime.utcnow()
                
                # Add delay between steps to avoid overwhelming the system
                await asyncio.sleep(0.1)
            
            workflow["status"] = WorkflowStatus.COMPLETED
            workflow["completed_at"] = datetime.utcnow()
            
            logger.info(f"Workflow {workflow_id} completed successfully")
            
        except Exception as e:
            logger.error(f"Workflow execution error: {str(e)}")
            if workflow_id in self.workflows:
                self.workflows[workflow_id]["status"] = WorkflowStatus.FAILED
                self.workflows[workflow_id]["error"] = str(e)
    
    async def _execute_workflow_step(self, workflow_id: str, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single workflow step"""
        try:
            agent_name = step["agent"]
            action = step["action"]
            
            agent = self.agents.get(agent_name)
            if not agent:
                return {"success": False, "error": f"Agent {agent_name} not found"}
            
            workflow = self.workflows[workflow_id]
            
            # Prepare step content based on workflow context
            step_content = await self._prepare_step_content(workflow, step)
            
            # Create message for agent
            message = AgentMessage(
                sender_id="orchestrator",
                receiver_id=agent.agent_id,
                message_type="request",
                content={
                    "action": action,
                    **step_content
                },
                correlation_id=workflow_id
            )
            
            # Send message to agent and get response
            response = await agent.receive_message(message)
            
            logger.info(f"Workflow {workflow_id} step {workflow['current_step']} ({agent_name}.{action}) completed: {response.success}")
            
            return {
                "success": response.success,
                "data": response.data,
                "error": response.error,
                "processing_time": response.processing_time
            }
            
        except Exception as e:
            logger.error(f"Workflow step execution error: {str(e)}")
            return {"success": False, "error": str(e)}
    
    async def _prepare_step_content(self, workflow: Dict[str, Any], step: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare content for a workflow step based on context"""
        try:
            content = {}
            
            # Add common context
            if "event_id" in workflow:
                content["event_id"] = workflow["event_id"]
            
            # Add step-specific content based on action
            action = step["action"]
            
            if action == "detect_events":
                # Oracle agent event detection
                content.update({
                    "data_source": "mixed",
                    "data": {}
                })
            
            elif action == "enrich_event":
                # Enricher agent event enrichment
                content.update({
                    "enrichment_types": ["all"]
                })
            
            elif action == "generate_proposal":
                # Proposer agent proposal generation
                event = Event.query.get(workflow["event_id"]) if workflow.get("event_id") else None
                if event and event.client_id:
                    portfolio = Portfolio.query.filter(Portfolio.client_id == event.client_id).first()
                    if portfolio:
                        content["portfolio_id"] = portfolio.id
                
                content.update({
                    "objective": "SharpeMax"
                })
            
            elif action == "validate_proposal":
                # Checker agent proposal validation
                # Get proposal ID from previous step results
                proposal_id = self._extract_proposal_id_from_results(workflow["results"])
                if proposal_id:
                    content["proposal_id"] = proposal_id
            
            elif action == "execute_proposal":
                # Executor agent proposal execution
                proposal_id = self._extract_proposal_id_from_results(workflow["results"])
                if proposal_id:
                    content.update({
                        "proposal_id": proposal_id,
                        "execution_strategy": "market"
                    })
            
            elif action == "generate_client_communication":
                # Narrator agent client communication
                event = Event.query.get(workflow["event_id"]) if workflow.get("event_id") else None
                if event and event.client_id:
                    content.update({
                        "client_id": event.client_id,
                        "communication_type": "event_notification",
                        "content": {"event_id": workflow["event_id"]},
                        "tone": "professional"
                    })
            
            return content
            
        except Exception as e:
            logger.error(f"Step content preparation error: {str(e)}")
            return {}
    
    def _extract_proposal_id_from_results(self, results: Dict[str, Any]) -> Optional[str]:
        """Extract proposal ID from workflow results"""
        try:
            for step_key, step_result in results.items():
                if "proposer" in step_key and step_result.get("success"):
                    data = step_result.get("data", {})
                    if "proposal_id" in data:
                        return data["proposal_id"]
            return None
        except Exception:
            return None
    
    async def _check_step_condition(self, condition: str, workflow: Dict[str, Any]) -> bool:
        """Check if a step condition is met"""
        try:
            if condition == "approved":
                # Check if the proposal was approved by the checker
                for step_key, step_result in workflow["results"].items():
                    if "checker" in step_key and step_result.get("success"):
                        data = step_result.get("data", {})
                        return data.get("overall_status") == "APPROVED"
                return False
            
            # Add more conditions as needed
            return True
            
        except Exception as e:
            logger.error(f"Step condition check error: {str(e)}")
            return False
    
    async def _process_message_queue(self):
        """Process messages in the queue"""
        while self.running:
            try:
                # Process any queued messages
                if not self.message_queue.empty():
                    message = await self.message_queue.get()
                    await self._route_message(message)
                
                await asyncio.sleep(0.1)  # Small delay to prevent busy waiting
                
            except Exception as e:
                logger.error(f"Message queue processing error: {str(e)}")
                await asyncio.sleep(1)
    
    async def _route_message(self, message: AgentMessage):
        """Route a message to the appropriate agent"""
        try:
            target_agent = agent_registry.get_agent(message.receiver_id)
            if target_agent:
                response = await target_agent.receive_message(message)
                logger.info(f"Message {message.id} routed to {message.receiver_id}: {response.success}")
            else:
                logger.error(f"Target agent {message.receiver_id} not found for message {message.id}")
                
        except Exception as e:
            logger.error(f"Message routing error: {str(e)}")
    
    async def _monitor_workflows(self):
        """Monitor running workflows and handle timeouts"""
        while self.running:
            try:
                current_time = datetime.utcnow()
                
                for workflow_id, workflow in list(self.workflows.items()):
                    # Check for workflow timeouts (30 minutes)
                    if workflow["status"] == WorkflowStatus.RUNNING:
                        elapsed = current_time - workflow["created_at"]
                        if elapsed > timedelta(minutes=30):
                            workflow["status"] = WorkflowStatus.FAILED
                            workflow["error"] = "Workflow timeout"
                            logger.warning(f"Workflow {workflow_id} timed out")
                    
                    # Clean up old completed workflows (keep for 24 hours)
                    elif workflow["status"] in [WorkflowStatus.COMPLETED, WorkflowStatus.FAILED]:
                        elapsed = current_time - workflow.get("completed_at", workflow["updated_at"])
                        if elapsed > timedelta(hours=24):
                            del self.workflows[workflow_id]
                            logger.info(f"Cleaned up old workflow {workflow_id}")
                
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Workflow monitoring error: {str(e)}")
                await asyncio.sleep(60)
    
    async def _health_check_agents(self):
        """Perform periodic health checks on all agents"""
        while self.running:
            try:
                health_results = await agent_registry.health_check_all()
                
                unhealthy_agents = [agent_id for agent_id, healthy in health_results.items() if not healthy]
                
                if unhealthy_agents:
                    logger.warning(f"Unhealthy agents detected: {unhealthy_agents}")
                    # Could implement agent restart logic here
                
                await asyncio.sleep(300)  # Check every 5 minutes
                
            except Exception as e:
                logger.error(f"Agent health check error: {str(e)}")
                await asyncio.sleep(300)
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get the status of a workflow"""
        workflow = self.workflows.get(workflow_id)
        if workflow:
            return {
                "id": workflow["id"],
                "type": workflow["type"],
                "status": workflow["status"],
                "current_step": workflow["current_step"],
                "total_steps": len(workflow["steps"]),
                "created_at": workflow["created_at"].isoformat(),
                "updated_at": workflow["updated_at"].isoformat(),
                "error": workflow.get("error")
            }
        return None
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel a running workflow"""
        workflow = self.workflows.get(workflow_id)
        if workflow and workflow["status"] == WorkflowStatus.RUNNING:
            workflow["status"] = WorkflowStatus.CANCELLED
            workflow["updated_at"] = datetime.utcnow()
            logger.info(f"Workflow {workflow_id} cancelled")
            return True
        return False
    
    async def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        agent_statuses = {}
        for name, agent in self.agents.items():
            agent_statuses[name] = agent.get_status()
        return agent_statuses
    
    async def send_message_to_agent(
        self, 
        agent_name: str, 
        action: str, 
        content: Dict[str, Any]
    ) -> AgentResponse:
        """Send a direct message to an agent"""
        try:
            agent = self.agents.get(agent_name)
            if not agent:
                return AgentResponse(success=False, error=f"Agent {agent_name} not found")
            
            message = AgentMessage(
                sender_id="orchestrator",
                receiver_id=agent.agent_id,
                message_type="request",
                content={"action": action, **content}
            )
            
            response = await agent.receive_message(message)
            return response
            
        except Exception as e:
            logger.error(f"Direct message error: {str(e)}")
            return AgentResponse(success=False, error=str(e))
    
    async def create_custom_workflow(
        self, 
        workflow_type: str, 
        steps: List[Dict[str, Any]], 
        context: Dict[str, Any] = None
    ) -> str:
        """Create a custom workflow"""
        try:
            workflow_id = f"{workflow_type}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
            
            workflow = {
                "id": workflow_id,
                "type": workflow_type,
                "status": WorkflowStatus.PENDING,
                "steps": steps,
                "current_step": 0,
                "results": {},
                "context": context or {},
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            self.workflows[workflow_id] = workflow
            
            # Start workflow execution
            await self._execute_workflow(workflow_id)
            
            return workflow_id
            
        except Exception as e:
            logger.error(f"Custom workflow creation error: {str(e)}")
            raise

# Global orchestrator instance
orchestrator = AgentOrchestrator()

