import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field
from enum import Enum
import openai
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentType(str, Enum):
    ORACLE = "oracle"
    ENRICHER = "enricher"
    PROPOSER = "proposer"
    CHECKER = "checker"
    EXECUTOR = "executor"
    NARRATOR = "narrator"

class MessageType(str, Enum):
    REQUEST = "request"
    RESPONSE = "response"
    NOTIFICATION = "notification"
    ERROR = "error"

class AgentStatus(str, Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    WAITING = "waiting"
    ERROR = "error"
    OFFLINE = "offline"

class AgentMessage(BaseModel):
    """Standard message format for agent communication"""
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    sender_id: str
    receiver_id: str
    message_type: MessageType
    content: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    correlation_id: Optional[str] = None  # For tracking related messages
    priority: int = Field(default=5, ge=1, le=10)  # 1=highest, 10=lowest
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }

class AgentResponse(BaseModel):
    """Standard response format from agents"""
    success: bool
    data: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    processing_time: Optional[float] = None
    
class AgentCapability(BaseModel):
    """Describes what an agent can do"""
    name: str
    description: str
    input_schema: Dict[str, Any]
    output_schema: Dict[str, Any]
    
class BaseAgent(ABC):
    """Base class for all agents in the wealth management system"""
    
    def __init__(
        self,
        agent_id: str,
        agent_type: AgentType,
        name: str,
        description: str,
        capabilities: List[AgentCapability] = None
    ):
        self.agent_id = agent_id
        self.agent_type = agent_type
        self.name = name
        self.description = description
        self.capabilities = capabilities or []
        self.status = AgentStatus.IDLE
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.response_handlers: Dict[str, asyncio.Future] = {}
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.processed_messages = 0
        self.error_count = 0
        
        # OpenAI client for LLM interactions
        self.openai_client = openai.OpenAI()
        
        logger.info(f"Initialized agent: {self.name} ({self.agent_id})")
    
    @abstractmethod
    async def process_message(self, message: AgentMessage) -> AgentResponse:
        """Process an incoming message and return a response"""
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """Get the system prompt for this agent's LLM interactions"""
        pass
    
    async def send_message(
        self,
        receiver_id: str,
        content: Dict[str, Any],
        message_type: MessageType = MessageType.REQUEST,
        correlation_id: Optional[str] = None,
        priority: int = 5
    ) -> str:
        """Send a message to another agent"""
        message = AgentMessage(
            sender_id=self.agent_id,
            receiver_id=receiver_id,
            message_type=message_type,
            content=content,
            correlation_id=correlation_id,
            priority=priority
        )
        
        # In a real system, this would go through a message broker
        # For now, we'll use a simple in-memory approach
        await self._deliver_message(message)
        
        logger.info(f"Agent {self.agent_id} sent message {message.id} to {receiver_id}")
        return message.id
    
    async def _deliver_message(self, message: AgentMessage):
        """Deliver a message to the target agent (placeholder for message broker)"""
        # This would be implemented with a proper message broker like RabbitMQ or Redis
        # For now, we'll store it in a global message store
        pass
    
    async def receive_message(self, message: AgentMessage) -> AgentResponse:
        """Receive and process a message"""
        try:
            self.status = AgentStatus.PROCESSING
            self.last_activity = datetime.utcnow()
            
            start_time = datetime.utcnow()
            response = await self.process_message(message)
            end_time = datetime.utcnow()
            
            response.processing_time = (end_time - start_time).total_seconds()
            self.processed_messages += 1
            self.status = AgentStatus.IDLE
            
            logger.info(f"Agent {self.agent_id} processed message {message.id} in {response.processing_time:.2f}s")
            return response
            
        except Exception as e:
            self.error_count += 1
            self.status = AgentStatus.ERROR
            logger.error(f"Agent {self.agent_id} error processing message {message.id}: {str(e)}")
            
            return AgentResponse(
                success=False,
                error=str(e),
                metadata={"message_id": message.id, "agent_id": self.agent_id}
            )
    
    async def call_llm(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1000
    ) -> str:
        """Call OpenAI LLM with the given prompt"""
        try:
            messages = []
            
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            else:
                messages.append({"role": "system", "content": self.get_system_prompt()})
            
            messages.append({"role": "user", "content": prompt})
            
            response = await asyncio.to_thread(
                self.openai_client.chat.completions.create,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            logger.error(f"LLM call failed for agent {self.agent_id}: {str(e)}")
            raise
    
    async def call_llm_with_json(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.3
    ) -> Dict[str, Any]:
        """Call LLM and parse response as JSON"""
        try:
            json_prompt = f"{prompt}\n\nPlease respond with valid JSON only."
            response_text = await self.call_llm(
                json_prompt,
                system_prompt,
                model,
                temperature
            )
            
            # Try to extract JSON from the response
            response_text = response_text.strip()
            if response_text.startswith("```json"):
                response_text = response_text[7:]
            if response_text.endswith("```"):
                response_text = response_text[:-3]
            
            return json.loads(response_text.strip())
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {response_text}")
            raise ValueError(f"Invalid JSON response from LLM: {str(e)}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get current agent status and metrics"""
        return {
            "agent_id": self.agent_id,
            "agent_type": self.agent_type.value,
            "name": self.name,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "last_activity": self.last_activity.isoformat(),
            "processed_messages": self.processed_messages,
            "error_count": self.error_count,
            "queue_size": self.message_queue.qsize(),
            "capabilities": [cap.name for cap in self.capabilities]
        }
    
    def add_capability(self, capability: AgentCapability):
        """Add a new capability to this agent"""
        self.capabilities.append(capability)
        logger.info(f"Added capability '{capability.name}' to agent {self.agent_id}")
    
    async def health_check(self) -> bool:
        """Perform a health check on this agent"""
        try:
            # Basic health check - can be extended
            if self.status == AgentStatus.ERROR:
                return False
            
            # Check if agent is responsive
            test_message = AgentMessage(
                sender_id="health_check",
                receiver_id=self.agent_id,
                message_type=MessageType.REQUEST,
                content={"action": "ping"}
            )
            
            response = await asyncio.wait_for(
                self.receive_message(test_message),
                timeout=5.0
            )
            
            return response.success
            
        except Exception as e:
            logger.error(f"Health check failed for agent {self.agent_id}: {str(e)}")
            return False
    
    async def shutdown(self):
        """Gracefully shutdown the agent"""
        logger.info(f"Shutting down agent {self.agent_id}")
        self.status = AgentStatus.OFFLINE
        
        # Process any remaining messages
        while not self.message_queue.empty():
            try:
                message = await asyncio.wait_for(self.message_queue.get(), timeout=1.0)
                await self.receive_message(message)
            except asyncio.TimeoutError:
                break
        
        logger.info(f"Agent {self.agent_id} shutdown complete")

class AgentRegistry:
    """Registry to keep track of all agents in the system"""
    
    def __init__(self):
        self.agents: Dict[str, BaseAgent] = {}
        self.agents_by_type: Dict[AgentType, List[BaseAgent]] = {}
    
    def register_agent(self, agent: BaseAgent):
        """Register a new agent"""
        self.agents[agent.agent_id] = agent
        
        if agent.agent_type not in self.agents_by_type:
            self.agents_by_type[agent.agent_type] = []
        
        self.agents_by_type[agent.agent_type].append(agent)
        logger.info(f"Registered agent {agent.agent_id} of type {agent.agent_type.value}")
    
    def get_agent(self, agent_id: str) -> Optional[BaseAgent]:
        """Get an agent by ID"""
        return self.agents.get(agent_id)
    
    def get_agents_by_type(self, agent_type: AgentType) -> List[BaseAgent]:
        """Get all agents of a specific type"""
        return self.agents_by_type.get(agent_type, [])
    
    def get_all_agents(self) -> List[BaseAgent]:
        """Get all registered agents"""
        return list(self.agents.values())
    
    async def health_check_all(self) -> Dict[str, bool]:
        """Perform health check on all agents"""
        results = {}
        for agent_id, agent in self.agents.items():
            results[agent_id] = await agent.health_check()
        return results
    
    async def shutdown_all(self):
        """Shutdown all agents"""
        for agent in self.agents.values():
            await agent.shutdown()
        
        self.agents.clear()
        self.agents_by_type.clear()

# Global agent registry
agent_registry = AgentRegistry()

