import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../components/ui/card';
import { Button } from '../../components/ui/button';
import { Badge } from '../../components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../../components/ui/tabs';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line, AreaChart, Area } from 'recharts';
import { FileText, Download, Calendar, Filter, TrendingUp, DollarSign, Users, BarChart3, RefreshCw, Eye, Share, Archive } from 'lucide-react';

const ReportsPage = () => {
  const [selectedPeriod, setSelectedPeriod] = useState('monthly');
  const [refreshing, setRefreshing] = useState(false);

  // Mock reports data
  const reportsData = {
    overview: {
      totalReports: 45,
      scheduledReports: 12,
      customReports: 8,
      lastGenerated: '2024-01-22'
    },
    availableReports: [
      {
        id: 1,
        name: 'Portfolio Performance Report',
        description: 'Comprehensive portfolio performance analysis with benchmarks',
        category: 'Performance',
        frequency: 'Monthly',
        lastGenerated: '2024-01-20',
        status: 'Ready',
        recipients: 15,
        format: 'PDF'
      },
      {
        id: 2,
        name: 'Client Risk Assessment',
        description: 'Risk profile analysis and compliance review for all clients',
        category: 'Risk',
        frequency: 'Quarterly',
        lastGenerated: '2024-01-15',
        status: 'Generating',
        recipients: 8,
        format: 'Excel'
      },
      {
        id: 3,
        name: 'Compliance Summary',
        description: 'KYC/AML compliance status and regulatory requirements',
        category: 'Compliance',
        frequency: 'Weekly',
        lastGenerated: '2024-01-22',
        status: 'Ready',
        recipients: 5,
        format: 'PDF'
      },
      {
        id: 4,
        name: 'Market Intelligence Report',
        description: 'Market trends, news analysis, and investment opportunities',
        category: 'Market',
        frequency: 'Daily',
        lastGenerated: '2024-01-22',
        status: 'Ready',
        recipients: 25,
        format: 'PDF'
      },
      {
        id: 5,
        name: 'Fee Analysis Report',
        description: 'Fee structure analysis and revenue breakdown by client',
        category: 'Financial',
        frequency: 'Monthly',
        lastGenerated: '2024-01-18',
        status: 'Scheduled',
        recipients: 3,
        format: 'Excel'
      }
    ],
    performanceMetrics: [
      { month: 'Jul', portfolioReturn: 8.2, benchmarkReturn: 7.1, alpha: 1.1, clientSatisfaction: 4.2 },
      { month: 'Aug', portfolioReturn: 9.5, benchmarkReturn: 8.3, alpha: 1.2, clientSatisfaction: 4.3 },
      { month: 'Sep', portfolioReturn: 7.8, benchmarkReturn: 6.9, alpha: 0.9, clientSatisfaction: 4.1 },
      { month: 'Oct', portfolioReturn: 11.2, benchmarkReturn: 9.8, alpha: 1.4, clientSatisfaction: 4.5 },
      { month: 'Nov', portfolioReturn: 13.1, benchmarkReturn: 11.5, alpha: 1.6, clientSatisfaction: 4.4 },
      { month: 'Dec', portfolioReturn: 12.8, benchmarkReturn: 10.2, alpha: 2.6, clientSatisfaction: 4.6 }
    ],
    aumBreakdown: [
      { category: 'Equities', value: 65.2, color: '#3b82f6' },
      { category: 'Fixed Income', value: 20.8, color: '#10b981' },
      { category: 'Alternatives', value: 8.5, color: '#f59e0b' },
      { category: 'Cash', value: 5.5, color: '#6b7280' }
    ],
    clientMetrics: [
      { metric: 'Total Clients', value: 156, change: '+8', trend: 'up' },
      { metric: 'Active Portfolios', value: 342, change: '+15', trend: 'up' },
      { metric: 'Average AUM', value: '$2.4M', change: '+12%', trend: 'up' },
      { metric: 'Client Retention', value: '94.2%', change: '+1.2%', trend: 'up' }
    ],
    complianceStatus: [
      { area: 'KYC Compliance', status: 'Compliant', percentage: 98.5, issues: 2 },
      { area: 'AML Monitoring', status: 'Compliant', percentage: 99.2, issues: 1 },
      { area: 'Risk Assessment', status: 'Review Required', percentage: 95.8, issues: 6 },
      { area: 'Regulatory Reporting', status: 'Compliant', percentage: 100, issues: 0 }
    ],
    recentReports: [
      { id: 1, name: 'Q4 Performance Summary', date: '2024-01-22', type: 'Performance', size: '2.4 MB' },
      { id: 2, name: 'Weekly Compliance Report', date: '2024-01-22', type: 'Compliance', size: '1.8 MB' },
      { id: 3, name: 'Market Intelligence Brief', date: '2024-01-21', type: 'Market', size: '3.2 MB' },
      { id: 4, name: 'Client Risk Assessment', date: '2024-01-20', type: 'Risk', size: '4.1 MB' }
    ]
  };

  const handleRefresh = () => {
    setRefreshing(true);
    setTimeout(() => setRefreshing(false), 2000);
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'Ready': return 'bg-green-100 text-green-800';
      case 'Generating': return 'bg-blue-100 text-blue-800';
      case 'Scheduled': return 'bg-yellow-100 text-yellow-800';
      case 'Error': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getCategoryColor = (category) => {
    switch (category) {
      case 'Performance': return 'bg-blue-100 text-blue-800';
      case 'Risk': return 'bg-red-100 text-red-800';
      case 'Compliance': return 'bg-green-100 text-green-800';
      case 'Market': return 'bg-purple-100 text-purple-800';
      case 'Financial': return 'bg-orange-100 text-orange-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getComplianceColor = (status) => {
    switch (status) {
      case 'Compliant': return 'text-green-600';
      case 'Review Required': return 'text-yellow-600';
      case 'Non-Compliant': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Reports & Analytics</h1>
          <p className="text-gray-600">Generate and manage comprehensive business reports</p>
        </div>
        <div className="flex space-x-2">
          <Button variant="outline" size="sm" onClick={handleRefresh} disabled={refreshing}>
            <RefreshCw className={`h-4 w-4 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Button variant="outline" size="sm">
            <Calendar className="h-4 w-4 mr-2" />
            Schedule Report
          </Button>
          <Button size="sm">
            <FileText className="h-4 w-4 mr-2" />
            Create Report
          </Button>
        </div>
      </div>

      {/* Reports Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Reports</CardTitle>
            <FileText className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{reportsData.overview.totalReports}</div>
            <p className="text-xs text-muted-foreground">
              Available report templates
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Scheduled Reports</CardTitle>
            <Calendar className="h-4 w-4 text-blue-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-blue-600">{reportsData.overview.scheduledReports}</div>
            <p className="text-xs text-muted-foreground">
              Automated report generation
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Custom Reports</CardTitle>
            <BarChart3 className="h-4 w-4 text-green-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">{reportsData.overview.customReports}</div>
            <p className="text-xs text-muted-foreground">
              User-defined reports
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Last Generated</CardTitle>
            <Download className="h-4 w-4 text-purple-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-purple-600">Today</div>
            <p className="text-xs text-muted-foreground">
              {reportsData.overview.lastGenerated}
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Tabs */}
      <Tabs defaultValue="reports" className="space-y-4">
        <TabsList>
          <TabsTrigger value="reports">Available Reports</TabsTrigger>
          <TabsTrigger value="analytics">Business Analytics</TabsTrigger>
          <TabsTrigger value="compliance">Compliance Dashboard</TabsTrigger>
          <TabsTrigger value="recent">Recent Reports</TabsTrigger>
        </TabsList>

        <TabsContent value="reports" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Report Library</CardTitle>
              <CardDescription>Generate and manage business reports</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {reportsData.availableReports.map((report) => (
                  <div key={report.id} className="p-4 border rounded-lg hover:shadow-md transition-shadow">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-3 mb-2">
                          <h3 className="text-lg font-semibold">{report.name}</h3>
                          <Badge className={getCategoryColor(report.category)} size="sm">
                            {report.category}
                          </Badge>
                          <Badge className={getStatusColor(report.status)} size="sm">
                            {report.status}
                          </Badge>
                        </div>
                        <p className="text-gray-600 mb-3">{report.description}</p>
                        <div className="grid grid-cols-1 md:grid-cols-4 gap-4 text-sm">
                          <div>
                            <span className="text-gray-500">Frequency:</span>
                            <p className="font-medium">{report.frequency}</p>
                          </div>
                          <div>
                            <span className="text-gray-500">Last Generated:</span>
                            <p className="font-medium">{report.lastGenerated}</p>
                          </div>
                          <div>
                            <span className="text-gray-500">Recipients:</span>
                            <p className="font-medium">{report.recipients} users</p>
                          </div>
                          <div>
                            <span className="text-gray-500">Format:</span>
                            <p className="font-medium">{report.format}</p>
                          </div>
                        </div>
                      </div>
                      <div className="flex space-x-2 ml-4">
                        <Button size="sm" variant="outline">
                          <Eye className="h-4 w-4 mr-2" />
                          Preview
                        </Button>
                        <Button size="sm" variant="outline">
                          <Download className="h-4 w-4 mr-2" />
                          Generate
                        </Button>
                        <Button size="sm" variant="outline">
                          <Share className="h-4 w-4 mr-2" />
                          Share
                        </Button>
                      </div>
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
                <CardTitle>Portfolio Performance Trends</CardTitle>
                <CardDescription>Portfolio vs benchmark performance over time</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <AreaChart data={reportsData.performanceMetrics}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" />
                    <YAxis />
                    <Tooltip />
                    <Area type="monotone" dataKey="portfolioReturn" stackId="1" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.6} name="Portfolio Return %" />
                    <Area type="monotone" dataKey="benchmarkReturn" stackId="2" stroke="#10b981" fill="#10b981" fillOpacity={0.6} name="Benchmark Return %" />
                  </AreaChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>AUM Allocation</CardTitle>
                <CardDescription>Assets under management by category</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={reportsData.aumBreakdown}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={120}
                      paddingAngle={5}
                      dataKey="value"
                    >
                      {reportsData.aumBreakdown.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value) => [`${value}%`, '']} />
                  </PieChart>
                </ResponsiveContainer>
                <div className="grid grid-cols-2 gap-2 mt-4">
                  {reportsData.aumBreakdown.map((item, index) => (
                    <div key={index} className="flex items-center space-x-2">
                      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }}></div>
                      <span className="text-sm">{item.category}: {item.value}%</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card className="lg:col-span-2">
              <CardHeader>
                <CardTitle>Key Business Metrics</CardTitle>
                <CardDescription>Important business performance indicators</CardDescription>
              </CardHeader>
              <CardContent>
                <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                  {reportsData.clientMetrics.map((metric, index) => (
                    <div key={index} className="p-4 border rounded-lg text-center">
                      <p className="text-sm text-gray-600 mb-1">{metric.metric}</p>
                      <p className="text-2xl font-bold mb-1">{metric.value}</p>
                      <div className={`flex items-center justify-center text-sm ${
                        metric.trend === 'up' ? 'text-green-600' : 'text-red-600'
                      }`}>
                        <TrendingUp className="h-3 w-3 mr-1" />
                        {metric.change}
                      </div>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="compliance" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Compliance Status Dashboard</CardTitle>
              <CardDescription>Real-time compliance monitoring and status</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {reportsData.complianceStatus.map((item, index) => (
                  <div key={index} className="p-4 border rounded-lg">
                    <div className="flex items-center justify-between mb-2">
                      <h3 className="font-medium">{item.area}</h3>
                      <div className="flex items-center space-x-2">
                        <span className={`font-medium ${getComplianceColor(item.status)}`}>
                          {item.status}
                        </span>
                        <span className="text-sm text-gray-600">
                          {item.issues} issues
                        </span>
                      </div>
                    </div>
                    <div className="w-full bg-gray-200 rounded-full h-2">
                      <div
                        className={`h-2 rounded-full ${
                          item.percentage >= 98 ? 'bg-green-500' :
                          item.percentage >= 95 ? 'bg-yellow-500' : 'bg-red-500'
                        }`}
                        style={{ width: `${item.percentage}%` }}
                      ></div>
                    </div>
                    <div className="flex justify-between text-sm text-gray-600 mt-1">
                      <span>Compliance Rate</span>
                      <span>{item.percentage}%</span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="recent" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Recently Generated Reports</CardTitle>
              <CardDescription>Download and manage recent reports</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {reportsData.recentReports.map((report) => (
                  <div key={report.id} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex items-center space-x-4">
                      <FileText className="h-8 w-8 text-blue-600" />
                      <div>
                        <p className="font-medium">{report.name}</p>
                        <p className="text-sm text-gray-600">{report.type} • {report.size}</p>
                      </div>
                    </div>
                    <div className="flex items-center space-x-4">
                      <span className="text-sm text-gray-600">{report.date}</span>
                      <div className="flex space-x-2">
                        <Button size="sm" variant="outline">
                          <Eye className="h-4 w-4 mr-2" />
                          View
                        </Button>
                        <Button size="sm" variant="outline">
                          <Download className="h-4 w-4 mr-2" />
                          Download
                        </Button>
                        <Button size="sm" variant="outline">
                          <Archive className="h-4 w-4 mr-2" />
                          Archive
                        </Button>
                      </div>
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

export default ReportsPage;

