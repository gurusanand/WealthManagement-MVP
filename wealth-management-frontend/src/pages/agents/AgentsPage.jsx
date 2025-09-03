import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../components/ui/card';
import { Button } from '../../components/ui/button';
import { Badge } from '../../components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../../components/ui/tabs';
import { Input } from '../../components/ui/input';
import { Label } from '../../components/ui/label';
import { Textarea } from '../../components/ui/textarea';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '../../components/ui/select';
import { useToast } from '../../hooks/use-toast';
import {
  Bot,
  Activity,
  MessageSquare,
  Play,
  Pause,
  RefreshCw,
  AlertCircle,
  CheckCircle,
  Clock,
  Zap,
  Brain,
  Shield,
  TrendingUp,
  FileText,
  Settings
} from 'lucide-react';

const AgentsPage = () => {
  const { toast } = useToast();
  const [agents, setAgents] = useState([]);
  const [workflows, setWorkflows] = useState([]);
  const [selectedAgent, setSelectedAgent] = useState(null);
  const [messageText, setMessageText] = useState('');
  const [workflowType, setWorkflowType] = useState('');
  const [loading, setLoading] = useState(false);

  // Mock agent data - in production this would come from the backend
  useEffect(() => {
    const mockAgents = [
      {
        id: 'oracle_001',
        name: 'Oracle Agent',
        type: 'oracle',
        status: 'healthy',
        description: 'Event detection and market intelligence',
        icon: Activity,
        capabilities: ['event_detection', 'market_analysis', 'news_monitoring'],
        lastActivity: '2 minutes ago',
        tasksCompleted: 1247,
        uptime: '99.8%',
        performance: 'excellent'
      },
      {
        id: 'enricher_001',
        name: 'Enricher Agent',
        type: 'enricher',
        status: 'healthy',
        description: 'Context analysis and data enrichment',
        icon: Brain,
        capabilities: ['context_analysis', 'sentiment_analysis', 'correlation_detection'],
        lastActivity: '5 minutes ago',
        tasksCompleted: 892,
        uptime: '99.5%',
        performance: 'excellent'
      },
      {
        id: 'proposer_001',
        name: 'Proposer Agent',
        type: 'proposer',
        status: 'healthy',
        description: 'Portfolio optimization and recommendations',
        icon: TrendingUp,
        capabilities: ['portfolio_optimization', 'recommendation_generation', 'strategy_development'],
        lastActivity: '1 minute ago',
        tasksCompleted: 634,
        uptime: '99.9%',
        performance: 'excellent'
      },
      {
        id: 'checker_001',
        name: 'Checker Agent',
        type: 'checker',
        status: 'warning',
        description: 'Compliance validation and risk assessment',
        icon: Shield,
        capabilities: ['compliance_validation', 'risk_assessment', 'policy_enforcement'],
        lastActivity: '3 minutes ago',
        tasksCompleted: 445,
        uptime: '98.2%',
        performance: 'good'
      },
      {
        id: 'executor_001',
        name: 'Executor Agent',
        type: 'executor',
        status: 'healthy',
        description: 'Trade execution and portfolio updates',
        icon: Zap,
        capabilities: ['trade_execution', 'portfolio_updates', 'order_management'],
        lastActivity: '4 minutes ago',
        tasksCompleted: 328,
        uptime: '99.7%',
        performance: 'excellent'
      },
      {
        id: 'narrator_001',
        name: 'Narrator Agent',
        type: 'narrator',
        status: 'healthy',
        description: 'Client communication and reporting',
        icon: FileText,
        capabilities: ['report_generation', 'client_communication', 'explanation_generation'],
        lastActivity: '6 minutes ago',
        tasksCompleted: 567,
        uptime: '99.4%',
        performance: 'excellent'
      }
    ];

    const mockWorkflows = [
      {
        id: 'wf_001',
        name: 'Portfolio Rebalancing',
        type: 'portfolio_rebalancing',
        status: 'running',
        progress: 75,
        startTime: '10:30 AM',
        estimatedCompletion: '11:15 AM',
        involvedAgents: ['oracle_001', 'enricher_001', 'proposer_001', 'checker_001', 'executor_001']
      },
      {
        id: 'wf_002',
        name: 'Market Event Analysis',
        type: 'event_processing',
        status: 'completed',
        progress: 100,
        startTime: '9:45 AM',
        completedTime: '10:20 AM',
        involvedAgents: ['oracle_001', 'enricher_001', 'narrator_001']
      },
      {
        id: 'wf_003',
        name: 'Compliance Review',
        type: 'compliance_check',
        status: 'pending',
        progress: 0,
        scheduledTime: '2:00 PM',
        involvedAgents: ['checker_001', 'narrator_001']
      }
    ];

    setAgents(mockAgents);
    setWorkflows(mockWorkflows);
  }, []);

  const getStatusColor = (status) => {
    switch (status) {
      case 'healthy': return 'bg-green-500';
      case 'warning': return 'bg-yellow-500';
      case 'error': return 'bg-red-500';
      case 'offline': return 'bg-gray-500';
      default: return 'bg-gray-500';
    }
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'healthy': return CheckCircle;
      case 'warning': return AlertCircle;
      case 'error': return AlertCircle;
      case 'offline': return Clock;
      default: return Clock;
    }
  };

  const getWorkflowStatusColor = (status) => {
    switch (status) {
      case 'running': return 'bg-blue-500';
      case 'completed': return 'bg-green-500';
      case 'pending': return 'bg-yellow-500';
      case 'failed': return 'bg-red-500';
      default: return 'bg-gray-500';
    }
  };

  const handleSendMessage = async (agentId) => {
    if (!messageText.trim()) return;

    setLoading(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      toast({
        title: "Message Sent",
        description: `Message sent to ${agents.find(a => a.id === agentId)?.name}`,
      });
      
      setMessageText('');
      setSelectedAgent(null);
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to send message to agent",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleStartWorkflow = async () => {
    if (!workflowType) return;

    setLoading(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1500));
      
      const newWorkflow = {
        id: `wf_${Date.now()}`,
        name: workflowType.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()),
        type: workflowType,
        status: 'running',
        progress: 0,
        startTime: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        involvedAgents: agents.slice(0, 3).map(a => a.id)
      };
      
      setWorkflows(prev => [newWorkflow, ...prev]);
      
      toast({
        title: "Workflow Started",
        description: `${newWorkflow.name} workflow has been initiated`,
      });
      
      setWorkflowType('');
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to start workflow",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const handleRestartAgent = async (agentId) => {
    setLoading(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 2000));
      
      setAgents(prev => prev.map(agent => 
        agent.id === agentId 
          ? { ...agent, status: 'healthy', lastActivity: 'Just now' }
          : agent
      ));
      
      toast({
        title: "Agent Restarted",
        description: `${agents.find(a => a.id === agentId)?.name} has been restarted successfully`,
      });
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to restart agent",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">AI Agents</h1>
          <p className="text-muted-foreground">
            Monitor and manage the multi-agent AI system
          </p>
        </div>
        <div className="flex space-x-2">
          <Button variant="outline" onClick={() => window.location.reload()}>
            <RefreshCw className="mr-2 h-4 w-4" />
            Refresh
          </Button>
        </div>
      </div>

      {/* System Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <Bot className="h-8 w-8 text-blue-500" />
              <div>
                <p className="text-2xl font-bold">{agents.length}</p>
                <p className="text-sm text-muted-foreground">Active Agents</p>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <Activity className="h-8 w-8 text-green-500" />
              <div>
                <p className="text-2xl font-bold">{agents.filter(a => a.status === 'healthy').length}</p>
                <p className="text-sm text-muted-foreground">Healthy</p>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <Play className="h-8 w-8 text-blue-500" />
              <div>
                <p className="text-2xl font-bold">{workflows.filter(w => w.status === 'running').length}</p>
                <p className="text-sm text-muted-foreground">Running Workflows</p>
              </div>
            </div>
          </CardContent>
        </Card>
        
        <Card>
          <CardContent className="p-6">
            <div className="flex items-center space-x-2">
              <CheckCircle className="h-8 w-8 text-green-500" />
              <div>
                <p className="text-2xl font-bold">{agents.reduce((sum, a) => sum + a.tasksCompleted, 0)}</p>
                <p className="text-sm text-muted-foreground">Tasks Completed</p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>

      {/* Main Content */}
      <Tabs defaultValue="agents" className="space-y-4">
        <TabsList>
          <TabsTrigger value="agents">Agents</TabsTrigger>
          <TabsTrigger value="workflows">Workflows</TabsTrigger>
          <TabsTrigger value="communication">Communication</TabsTrigger>
          <TabsTrigger value="monitoring">Monitoring</TabsTrigger>
        </TabsList>

        {/* Agents Tab */}
        <TabsContent value="agents" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {agents.map((agent) => {
              const Icon = agent.icon;
              const StatusIcon = getStatusIcon(agent.status);
              
              return (
                <Card key={agent.id} className="hover:shadow-lg transition-shadow">
                  <CardHeader className="pb-3">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <div className="p-2 bg-primary/10 rounded-lg">
                          <Icon className="h-6 w-6 text-primary" />
                        </div>
                        <div>
                          <CardTitle className="text-lg">{agent.name}</CardTitle>
                          <CardDescription>{agent.description}</CardDescription>
                        </div>
                      </div>
                      <div className="flex items-center space-x-2">
                        <div className={`w-3 h-3 rounded-full ${getStatusColor(agent.status)}`} />
                        <StatusIcon className="h-4 w-4 text-muted-foreground" />
                      </div>
                    </div>
                  </CardHeader>
                  
                  <CardContent className="space-y-4">
                    <div className="grid grid-cols-2 gap-4 text-sm">
                      <div>
                        <p className="text-muted-foreground">Status</p>
                        <Badge variant={agent.status === 'healthy' ? 'default' : 'destructive'}>
                          {agent.status}
                        </Badge>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Uptime</p>
                        <p className="font-medium">{agent.uptime}</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Tasks</p>
                        <p className="font-medium">{agent.tasksCompleted.toLocaleString()}</p>
                      </div>
                      <div>
                        <p className="text-muted-foreground">Performance</p>
                        <Badge variant="outline">{agent.performance}</Badge>
                      </div>
                    </div>
                    
                    <div>
                      <p className="text-sm text-muted-foreground mb-2">Capabilities</p>
                      <div className="flex flex-wrap gap-1">
                        {agent.capabilities.map((capability) => (
                          <Badge key={capability} variant="secondary" className="text-xs">
                            {capability.replace('_', ' ')}
                          </Badge>
                        ))}
                      </div>
                    </div>
                    
                    <div className="flex space-x-2">
                      <Button 
                        size="sm" 
                        variant="outline"
                        onClick={() => setSelectedAgent(agent.id)}
                      >
                        <MessageSquare className="mr-1 h-3 w-3" />
                        Message
                      </Button>
                      <Button 
                        size="sm" 
                        variant="outline"
                        onClick={() => handleRestartAgent(agent.id)}
                        disabled={loading}
                      >
                        <RefreshCw className="mr-1 h-3 w-3" />
                        Restart
                      </Button>
                    </div>
                    
                    <div className="text-xs text-muted-foreground">
                      Last activity: {agent.lastActivity}
                    </div>
                  </CardContent>
                </Card>
              );
            })}
          </div>
        </TabsContent>

        {/* Workflows Tab */}
        <TabsContent value="workflows" className="space-y-4">
          <div className="flex justify-between items-center">
            <h3 className="text-lg font-semibold">Active Workflows</h3>
            <div className="flex space-x-2">
              <Select value={workflowType} onValueChange={setWorkflowType}>
                <SelectTrigger className="w-48">
                  <SelectValue placeholder="Select workflow type" />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="portfolio_rebalancing">Portfolio Rebalancing</SelectItem>
                  <SelectItem value="event_processing">Event Processing</SelectItem>
                  <SelectItem value="compliance_check">Compliance Check</SelectItem>
                  <SelectItem value="risk_assessment">Risk Assessment</SelectItem>
                  <SelectItem value="client_communication">Client Communication</SelectItem>
                </SelectContent>
              </Select>
              <Button onClick={handleStartWorkflow} disabled={!workflowType || loading}>
                <Play className="mr-2 h-4 w-4" />
                Start Workflow
              </Button>
            </div>
          </div>
          
          <div className="space-y-4">
            {workflows.map((workflow) => (
              <Card key={workflow.id}>
                <CardContent className="p-6">
                  <div className="flex items-center justify-between">
                    <div className="flex items-center space-x-4">
                      <div className={`w-4 h-4 rounded-full ${getWorkflowStatusColor(workflow.status)}`} />
                      <div>
                        <h4 className="font-semibold">{workflow.name}</h4>
                        <p className="text-sm text-muted-foreground">
                          {workflow.status === 'running' && `Started at ${workflow.startTime} • ${workflow.progress}% complete`}
                          {workflow.status === 'completed' && `Completed at ${workflow.completedTime}`}
                          {workflow.status === 'pending' && `Scheduled for ${workflow.scheduledTime}`}
                        </p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-2">
                      <Badge variant="outline">{workflow.status}</Badge>
                      {workflow.status === 'running' && (
                        <Button size="sm" variant="outline">
                          <Pause className="h-3 w-3" />
                        </Button>
                      )}
                    </div>
                  </div>
                  
                  {workflow.status === 'running' && (
                    <div className="mt-4">
                      <div className="flex justify-between text-sm mb-1">
                        <span>Progress</span>
                        <span>{workflow.progress}%</span>
                      </div>
                      <div className="w-full bg-gray-200 rounded-full h-2">
                        <div 
                          className="bg-blue-600 h-2 rounded-full transition-all duration-300" 
                          style={{ width: `${workflow.progress}%` }}
                        />
                      </div>
                    </div>
                  )}
                  
                  <div className="mt-4">
                    <p className="text-sm text-muted-foreground mb-2">Involved Agents</p>
                    <div className="flex space-x-2">
                      {workflow.involvedAgents.map((agentId) => {
                        const agent = agents.find(a => a.id === agentId);
                        return agent ? (
                          <Badge key={agentId} variant="secondary">
                            {agent.name}
                          </Badge>
                        ) : null;
                      })}
                    </div>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        </TabsContent>

        {/* Communication Tab */}
        <TabsContent value="communication" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Send Message to Agent</CardTitle>
              <CardDescription>
                Communicate directly with AI agents for specific tasks or queries
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <div>
                <Label htmlFor="agent-select">Select Agent</Label>
                <Select value={selectedAgent || ''} onValueChange={setSelectedAgent}>
                  <SelectTrigger>
                    <SelectValue placeholder="Choose an agent" />
                  </SelectTrigger>
                  <SelectContent>
                    {agents.map((agent) => (
                      <SelectItem key={agent.id} value={agent.id}>
                        {agent.name} - {agent.description}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              
              <div>
                <Label htmlFor="message">Message</Label>
                <Textarea
                  id="message"
                  placeholder="Enter your message or task for the agent..."
                  value={messageText}
                  onChange={(e) => setMessageText(e.target.value)}
                  rows={4}
                />
              </div>
              
              <Button 
                onClick={() => handleSendMessage(selectedAgent)}
                disabled={!selectedAgent || !messageText.trim() || loading}
              >
                <MessageSquare className="mr-2 h-4 w-4" />
                Send Message
              </Button>
            </CardContent>
          </Card>
        </TabsContent>

        {/* Monitoring Tab */}
        <TabsContent value="monitoring" className="space-y-4">
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>System Health</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  {agents.map((agent) => (
                    <div key={agent.id} className="flex items-center justify-between">
                      <div className="flex items-center space-x-3">
                        <div className={`w-3 h-3 rounded-full ${getStatusColor(agent.status)}`} />
                        <span className="font-medium">{agent.name}</span>
                      </div>
                      <div className="text-right">
                        <p className="text-sm font-medium">{agent.uptime}</p>
                        <p className="text-xs text-muted-foreground">{agent.performance}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
            
            <Card>
              <CardHeader>
                <CardTitle>Performance Metrics</CardTitle>
              </CardHeader>
              <CardContent>
                <div className="space-y-4">
                  <div className="flex justify-between">
                    <span>Total Tasks Completed</span>
                    <span className="font-bold">{agents.reduce((sum, a) => sum + a.tasksCompleted, 0).toLocaleString()}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Average Uptime</span>
                    <span className="font-bold">99.4%</span>
                  </div>
                  <div className="flex justify-between">
                    <span>Active Workflows</span>
                    <span className="font-bold">{workflows.filter(w => w.status === 'running').length}</span>
                  </div>
                  <div className="flex justify-between">
                    <span>System Status</span>
                    <Badge variant="default">Operational</Badge>
                  </div>
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default AgentsPage;

