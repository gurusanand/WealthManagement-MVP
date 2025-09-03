import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../components/ui/card';
import { Button } from '../../components/ui/button';
import { Badge } from '../../components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../../components/ui/tabs';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line } from 'recharts';
import { AlertTriangle, CheckCircle, Clock, Users, Shield, FileText, Search, Settings, RefreshCw, Eye, Download } from 'lucide-react';

const CompliancePage = () => {
  const [refreshing, setRefreshing] = useState(false);

  // Mock compliance data
  const complianceData = {
    overview: {
      totalClients: 156,
      compliantClients: 142,
      pendingReviews: 8,
      criticalIssues: 6,
      complianceScore: 91.2
    },
    kycStatus: [
      { id: 1, client: 'Johnson Holdings LLC', status: 'expired', lastReview: '2023-10-15', nextDue: '2024-01-15', riskLevel: 'high' },
      { id: 2, client: 'Smith Family Trust', status: 'pending', lastReview: '2024-01-10', nextDue: '2024-01-25', riskLevel: 'medium' },
      { id: 3, client: 'Davis Corporation', status: 'compliant', lastReview: '2024-01-20', nextDue: '2024-04-20', riskLevel: 'low' },
      { id: 4, client: 'Wilson Enterprises', status: 'review_required', lastReview: '2023-12-01', nextDue: '2024-01-30', riskLevel: 'medium' }
    ],
    amlAlerts: [
      { id: 1, client: 'Anderson LLC', type: 'unusual_transaction', amount: 250000, date: '2024-01-22', status: 'investigating' },
      { id: 2, client: 'Brown Industries', type: 'velocity_check', amount: 180000, date: '2024-01-21', status: 'cleared' },
      { id: 3, client: 'Taylor Holdings', type: 'sanctions_screening', amount: 320000, date: '2024-01-20', status: 'pending' }
    ],
    policyViolations: [
      { id: 1, client: 'Johnson Holdings', policy: 'Position Limits', severity: 'high', description: 'Exceeded sector concentration limit', date: '2024-01-22' },
      { id: 2, client: 'Smith Trust', policy: 'Suitability', severity: 'medium', description: 'High-risk investment for conservative profile', date: '2024-01-21' },
      { id: 3, client: 'Davis Corp', policy: 'Liquidity Requirements', severity: 'low', description: 'Below minimum cash allocation', date: '2024-01-20' }
    ],
    complianceMetrics: [
      { month: 'Jul', score: 89.2, violations: 12, reviews: 45 },
      { month: 'Aug', score: 90.1, violations: 8, reviews: 52 },
      { month: 'Sep', score: 88.7, violations: 15, reviews: 38 },
      { month: 'Oct', score: 91.5, violations: 6, reviews: 41 },
      { month: 'Nov', score: 90.8, violations: 9, reviews: 47 },
      { month: 'Dec', score: 91.2, violations: 7, reviews: 43 }
    ],
    riskDistribution: [
      { name: 'Low Risk', value: 65, color: '#10b981' },
      { name: 'Medium Risk', value: 28, color: '#f59e0b' },
      { name: 'High Risk', value: 7, color: '#ef4444' }
    ],
    auditTrail: [
      { id: 1, action: 'KYC Review Completed', user: 'Sarah Johnson', client: 'Anderson LLC', timestamp: '2024-01-22 14:30:00' },
      { id: 2, action: 'Policy Violation Resolved', user: 'Mike Roberts', client: 'Smith Trust', timestamp: '2024-01-22 13:45:00' },
      { id: 3, action: 'AML Alert Cleared', user: 'Lisa Chen', client: 'Brown Industries', timestamp: '2024-01-22 12:15:00' },
      { id: 4, action: 'Suitability Assessment Updated', user: 'David Wilson', client: 'Taylor Holdings', timestamp: '2024-01-22 11:30:00' }
    ]
  };

  const handleRefresh = () => {
    setRefreshing(true);
    setTimeout(() => setRefreshing(false), 2000);
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'compliant': return 'bg-green-100 text-green-800';
      case 'pending': return 'bg-yellow-100 text-yellow-800';
      case 'expired': return 'bg-red-100 text-red-800';
      case 'review_required': return 'bg-orange-100 text-orange-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getRiskColor = (risk) => {
    switch (risk) {
      case 'low': return 'bg-green-100 text-green-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      case 'high': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getSeverityColor = (severity) => {
    switch (severity) {
      case 'low': return 'bg-green-100 text-green-800';
      case 'medium': return 'bg-yellow-100 text-yellow-800';
      case 'high': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getAMLStatusColor = (status) => {
    switch (status) {
      case 'cleared': return 'bg-green-100 text-green-800';
      case 'investigating': return 'bg-yellow-100 text-yellow-800';
      case 'pending': return 'bg-blue-100 text-blue-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Compliance Management</h1>
          <p className="text-gray-600">Monitor KYC/AML compliance, policy violations, and regulatory requirements</p>
        </div>
        <div className="flex space-x-2">
          <Button variant="outline" size="sm" onClick={handleRefresh} disabled={refreshing}>
            <RefreshCw className={`h-4 w-4 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Button variant="outline" size="sm">
            <Download className="h-4 w-4 mr-2" />
            Export Report
          </Button>
          <Button variant="outline" size="sm">
            <Settings className="h-4 w-4 mr-2" />
            Settings
          </Button>
        </div>
      </div>

      {/* Compliance Overview */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Clients</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{complianceData.overview.totalClients}</div>
            <p className="text-xs text-muted-foreground">
              Under compliance monitoring
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Compliant</CardTitle>
            <CheckCircle className="h-4 w-4 text-green-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">{complianceData.overview.compliantClients}</div>
            <p className="text-xs text-muted-foreground">
              {((complianceData.overview.compliantClients / complianceData.overview.totalClients) * 100).toFixed(1)}% of total
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Pending Reviews</CardTitle>
            <Clock className="h-4 w-4 text-yellow-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-yellow-600">{complianceData.overview.pendingReviews}</div>
            <p className="text-xs text-muted-foreground">
              Require attention
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Critical Issues</CardTitle>
            <AlertTriangle className="h-4 w-4 text-red-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-red-600">{complianceData.overview.criticalIssues}</div>
            <p className="text-xs text-muted-foreground">
              Immediate action required
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Compliance Score</CardTitle>
            <Shield className="h-4 w-4 text-blue-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-blue-600">{complianceData.overview.complianceScore}%</div>
            <p className="text-xs text-muted-foreground">
              +0.4% from last month
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Tabs */}
      <Tabs defaultValue="kyc" className="space-y-4">
        <TabsList>
          <TabsTrigger value="kyc">KYC Status</TabsTrigger>
          <TabsTrigger value="aml">AML Alerts</TabsTrigger>
          <TabsTrigger value="violations">Policy Violations</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
          <TabsTrigger value="audit">Audit Trail</TabsTrigger>
        </TabsList>

        <TabsContent value="kyc" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>KYC Status Overview</CardTitle>
              <CardDescription>Know Your Customer compliance status for all clients</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {complianceData.kycStatus.map((client) => (
                  <div key={client.id} className="p-4 border rounded-lg">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-4">
                        <div>
                          <p className="font-medium">{client.client}</p>
                          <p className="text-sm text-gray-600">Last Review: {client.lastReview}</p>
                          <p className="text-sm text-gray-600">Next Due: {client.nextDue}</p>
                        </div>
                      </div>
                      <div className="text-right space-y-2">
                        <Badge className={getStatusColor(client.status)}>
                          {client.status.replace('_', ' ').toUpperCase()}
                        </Badge>
                        <br />
                        <Badge className={getRiskColor(client.riskLevel)}>
                          {client.riskLevel.toUpperCase()} RISK
                        </Badge>
                      </div>
                    </div>
                    <div className="mt-3 flex space-x-2">
                      <Button size="sm" variant="outline">
                        <Eye className="h-4 w-4 mr-2" />
                        Review
                      </Button>
                      <Button size="sm" variant="outline">
                        <FileText className="h-4 w-4 mr-2" />
                        Documents
                      </Button>
                      <Button size="sm" variant="outline">Update Status</Button>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="aml" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>AML Alerts</CardTitle>
              <CardDescription>Anti-Money Laundering monitoring and alerts</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {complianceData.amlAlerts.map((alert) => (
                  <div key={alert.id} className="p-4 border rounded-lg">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="font-medium">{alert.client}</p>
                        <p className="text-sm text-gray-600">Type: {alert.type.replace('_', ' ')}</p>
                        <p className="text-sm text-gray-600">Amount: ${alert.amount.toLocaleString()}</p>
                        <p className="text-sm text-gray-600">Date: {alert.date}</p>
                      </div>
                      <div className="text-right">
                        <Badge className={getAMLStatusColor(alert.status)}>
                          {alert.status.toUpperCase()}
                        </Badge>
                      </div>
                    </div>
                    <div className="mt-3 flex space-x-2">
                      <Button size="sm" variant="outline">Investigate</Button>
                      <Button size="sm" variant="outline">Clear Alert</Button>
                      <Button size="sm" variant="outline">Escalate</Button>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="violations" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Policy Violations</CardTitle>
              <CardDescription>Investment policy and regulatory violations</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {complianceData.policyViolations.map((violation) => (
                  <div key={violation.id} className="p-4 border rounded-lg">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="font-medium">{violation.client}</p>
                        <p className="text-sm text-gray-600">Policy: {violation.policy}</p>
                        <p className="text-sm text-gray-600">{violation.description}</p>
                        <p className="text-sm text-gray-600">Date: {violation.date}</p>
                      </div>
                      <div className="text-right">
                        <Badge className={getSeverityColor(violation.severity)}>
                          {violation.severity.toUpperCase()}
                        </Badge>
                      </div>
                    </div>
                    <div className="mt-3 flex space-x-2">
                      <Button size="sm" variant="outline">Review</Button>
                      <Button size="sm" variant="outline">Resolve</Button>
                      <Button size="sm" variant="outline">Waive</Button>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="analytics" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Compliance Score Trend</CardTitle>
                <CardDescription>Monthly compliance score and violations</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={complianceData.complianceMetrics}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" />
                    <YAxis />
                    <Tooltip />
                    <Line type="monotone" dataKey="score" stroke="#3b82f6" strokeWidth={2} name="Compliance Score" />
                    <Line type="monotone" dataKey="violations" stroke="#ef4444" strokeWidth={2} name="Violations" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Risk Distribution</CardTitle>
                <CardDescription>Client risk level distribution</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={complianceData.riskDistribution}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={120}
                      paddingAngle={5}
                      dataKey="value"
                    >
                      {complianceData.riskDistribution.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value) => [`${value}%`, '']} />
                  </PieChart>
                </ResponsiveContainer>
                <div className="grid grid-cols-1 gap-2 mt-4">
                  {complianceData.riskDistribution.map((item, index) => (
                    <div key={index} className="flex items-center space-x-2">
                      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }}></div>
                      <span className="text-sm">{item.name}: {item.value}%</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="audit" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Audit Trail</CardTitle>
              <CardDescription>Recent compliance-related activities and changes</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {complianceData.auditTrail.map((entry) => (
                  <div key={entry.id} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex items-center space-x-4">
                      <div className="w-2 h-2 rounded-full bg-blue-500"></div>
                      <div>
                        <p className="font-medium">{entry.action}</p>
                        <p className="text-sm text-gray-600">Client: {entry.client}</p>
                        <p className="text-sm text-gray-600">User: {entry.user}</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-gray-600">{entry.timestamp}</p>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default CompliancePage;

