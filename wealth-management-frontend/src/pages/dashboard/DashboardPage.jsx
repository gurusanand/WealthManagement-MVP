import { useState, useEffect } from 'react'
import { useAuth } from '@/contexts/AuthContext'
import { useData } from '@/contexts/DataContext'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Progress } from '@/components/ui/progress'
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts'
import {
  TrendingUp,
  TrendingDown,
  Users,
  DollarSign,
  PieChart as PieChartIcon,
  AlertTriangle,
  Activity,
  Target,
  Briefcase,
  Shield,
  ArrowUpRight,
  ArrowDownRight
} from 'lucide-react'

const DashboardPage = () => {
  const { user, getUserRole } = useAuth()
  const { 
    clients, 
    portfolios, 
    marketData, 
    complianceAlerts, 
    getTotalAUM, 
    getClientCount,
    getActiveComplianceAlerts,
    loading 
  } = useData()

  const userRole = getUserRole()
  const totalAUM = getTotalAUM()
  const clientCount = getClientCount()
  const activeAlerts = getActiveComplianceAlerts()

  // Mock performance data
  const performanceData = [
    { month: 'Jan', portfolio: 8.2, benchmark: 7.8 },
    { month: 'Feb', portfolio: 9.1, benchmark: 8.5 },
    { month: 'Mar', portfolio: 7.8, benchmark: 8.2 },
    { month: 'Apr', portfolio: 10.2, benchmark: 9.1 },
    { month: 'May', portfolio: 11.5, benchmark: 10.2 },
    { month: 'Jun', portfolio: 9.8, benchmark: 9.5 },
  ]

  const assetAllocationData = [
    { name: 'Equities', value: 65, color: '#3b82f6' },
    { name: 'Fixed Income', value: 25, color: '#10b981' },
    { name: 'Alternatives', value: 7, color: '#f59e0b' },
    { name: 'Cash', value: 3, color: '#6b7280' },
  ]

  const clientGrowthData = [
    { month: 'Jan', clients: 145, aum: 2.1 },
    { month: 'Feb', clients: 152, aum: 2.3 },
    { month: 'Mar', clients: 148, aum: 2.2 },
    { month: 'Apr', clients: 165, aum: 2.8 },
    { month: 'May', clients: 172, aum: 3.1 },
    { month: 'Jun', clients: 180, aum: 3.4 },
  ]

  const formatCurrency = (value) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 0,
      maximumFractionDigits: 0,
    }).format(value)
  }

  const formatPercentage = (value) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(1)}%`
  }

  const MetricCard = ({ title, value, change, icon: Icon, trend, description }) => (
    <Card>
      <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
        <CardTitle className="text-sm font-medium">{title}</CardTitle>
        <Icon className="h-4 w-4 text-muted-foreground" />
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-bold">{value}</div>
        {change !== undefined && (
          <div className="flex items-center space-x-1 text-xs text-muted-foreground">
            {trend === 'up' ? (
              <ArrowUpRight className="h-3 w-3 text-green-500" />
            ) : (
              <ArrowDownRight className="h-3 w-3 text-red-500" />
            )}
            <span className={trend === 'up' ? 'text-green-500' : 'text-red-500'}>
              {formatPercentage(change)}
            </span>
            <span>from last month</span>
          </div>
        )}
        {description && (
          <p className="text-xs text-muted-foreground mt-1">{description}</p>
        )}
      </CardContent>
    </Card>
  )

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary mx-auto"></div>
          <p className="mt-2 text-muted-foreground">Loading dashboard...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Welcome Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold tracking-tight">
            Welcome back, {user?.name?.split(' ')[0]}
          </h1>
          <p className="text-muted-foreground">
            Here's what's happening with your {userRole === 'client' ? 'portfolio' : 'business'} today.
          </p>
        </div>
        <div className="flex items-center space-x-2">
          {activeAlerts.length > 0 && (
            <Badge variant="destructive" className="flex items-center space-x-1">
              <AlertTriangle className="h-3 w-3" />
              <span>{activeAlerts.length} Alert{activeAlerts.length > 1 ? 's' : ''}</span>
            </Badge>
          )}
        </div>
      </div>

      {/* Key Metrics */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        {userRole !== 'client' && (
          <>
            <MetricCard
              title="Total AUM"
              value={formatCurrency(totalAUM)}
              change={8.2}
              trend="up"
              icon={DollarSign}
              description="Assets under management"
            />
            <MetricCard
              title="Active Clients"
              value={clientCount.toString()}
              change={5.1}
              trend="up"
              icon={Users}
              description="Total client accounts"
            />
          </>
        )}
        
        {userRole === 'client' && (
          <>
            <MetricCard
              title="Portfolio Value"
              value={formatCurrency(2500000)}
              change={3.2}
              trend="up"
              icon={PieChartIcon}
              description="Total portfolio value"
            />
            <MetricCard
              title="YTD Performance"
              value="+8.5%"
              change={1.2}
              trend="up"
              icon={TrendingUp}
              description="Year-to-date returns"
            />
          </>
        )}

        <MetricCard
          title="Performance"
          value="+9.8%"
          change={2.1}
          trend="up"
          icon={TrendingUp}
          description="Monthly performance"
        />
        
        <MetricCard
          title="Risk Score"
          value="Moderate"
          icon={Shield}
          description="Current risk level"
        />
      </div>

      {/* Charts Section */}
      <div className="grid gap-6 md:grid-cols-2">
        {/* Performance Chart */}
        <Card>
          <CardHeader>
            <CardTitle>Portfolio Performance</CardTitle>
            <CardDescription>
              Portfolio vs benchmark performance over time
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <LineChart data={performanceData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="month" />
                <YAxis />
                <Tooltip formatter={(value) => `${value}%`} />
                <Legend />
                <Line 
                  type="monotone" 
                  dataKey="portfolio" 
                  stroke="#3b82f6" 
                  strokeWidth={2}
                  name="Portfolio"
                />
                <Line 
                  type="monotone" 
                  dataKey="benchmark" 
                  stroke="#6b7280" 
                  strokeWidth={2}
                  strokeDasharray="5 5"
                  name="Benchmark"
                />
              </LineChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>

        {/* Asset Allocation */}
        <Card>
          <CardHeader>
            <CardTitle>Asset Allocation</CardTitle>
            <CardDescription>
              Current portfolio allocation by asset class
            </CardDescription>
          </CardHeader>
          <CardContent>
            <ResponsiveContainer width="100%" height={300}>
              <PieChart>
                <Pie
                  data={assetAllocationData}
                  cx="50%"
                  cy="50%"
                  innerRadius={60}
                  outerRadius={100}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {assetAllocationData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip formatter={(value) => `${value}%`} />
                <Legend />
              </PieChart>
            </ResponsiveContainer>
          </CardContent>
        </Card>
      </div>

      {/* Additional Charts for Non-Client Users */}
      {userRole !== 'client' && (
        <div className="grid gap-6 md:grid-cols-2">
          {/* Client Growth */}
          <Card>
            <CardHeader>
              <CardTitle>Client Growth</CardTitle>
              <CardDescription>
                Client count and AUM growth over time
              </CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={300}>
                <AreaChart data={clientGrowthData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="month" />
                  <YAxis yAxisId="left" />
                  <YAxis yAxisId="right" orientation="right" />
                  <Tooltip />
                  <Legend />
                  <Area
                    yAxisId="left"
                    type="monotone"
                    dataKey="clients"
                    stackId="1"
                    stroke="#3b82f6"
                    fill="#3b82f6"
                    fillOpacity={0.6}
                    name="Clients"
                  />
                  <Line
                    yAxisId="right"
                    type="monotone"
                    dataKey="aum"
                    stroke="#10b981"
                    strokeWidth={2}
                    name="AUM ($B)"
                  />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>

          {/* Market Overview */}
          <Card>
            <CardHeader>
              <CardTitle>Market Overview</CardTitle>
              <CardDescription>
                Key market indices performance
              </CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {Object.entries(marketData.indices || {}).map(([symbol, data]) => (
                  <div key={symbol} className="flex items-center justify-between">
                    <div className="flex items-center space-x-3">
                      <div className="font-medium">{symbol}</div>
                      <div className="text-sm text-muted-foreground">
                        ${data.price?.toFixed(2)}
                      </div>
                    </div>
                    <div className={`flex items-center space-x-1 ${
                      data.change >= 0 ? 'text-green-500' : 'text-red-500'
                    }`}>
                      {data.change >= 0 ? (
                        <TrendingUp className="h-4 w-4" />
                      ) : (
                        <TrendingDown className="h-4 w-4" />
                      )}
                      <span className="text-sm font-medium">
                        {formatPercentage(data.changePercent)}
                      </span>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </div>
      )}

      {/* Compliance Alerts */}
      {activeAlerts.length > 0 && (
        <Card>
          <CardHeader>
            <CardTitle className="flex items-center space-x-2">
              <AlertTriangle className="h-5 w-5 text-destructive" />
              <span>Active Compliance Alerts</span>
            </CardTitle>
            <CardDescription>
              Issues requiring immediate attention
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-3">
              {activeAlerts.slice(0, 3).map((alert) => (
                <div key={alert.id} className="flex items-center justify-between p-3 border rounded-lg">
                  <div className="flex items-center space-x-3">
                    <Badge variant={
                      alert.severity === 'critical' ? 'destructive' :
                      alert.severity === 'high' ? 'destructive' :
                      'secondary'
                    }>
                      {alert.severity}
                    </Badge>
                    <div>
                      <p className="font-medium">{alert.message}</p>
                      <p className="text-sm text-muted-foreground">
                        {new Date(alert.date).toLocaleDateString()}
                      </p>
                    </div>
                  </div>
                  <Button variant="outline" size="sm">
                    Review
                  </Button>
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}

      {/* Quick Actions */}
      <Card>
        <CardHeader>
          <CardTitle>Quick Actions</CardTitle>
          <CardDescription>
            Frequently used actions and shortcuts
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-3 md:grid-cols-2 lg:grid-cols-4">
            {userRole === 'relationship_manager' && (
              <>
                <Button variant="outline" className="h-auto p-4 flex flex-col items-center space-y-2">
                  <Users className="h-6 w-6" />
                  <span>Add Client</span>
                </Button>
                <Button variant="outline" className="h-auto p-4 flex flex-col items-center space-y-2">
                  <PieChartIcon className="h-6 w-6" />
                  <span>Create Portfolio</span>
                </Button>
              </>
            )}
            
            {userRole === 'client' && (
              <>
                <Button variant="outline" className="h-auto p-4 flex flex-col items-center space-y-2">
                  <Activity className="h-6 w-6" />
                  <span>View Transactions</span>
                </Button>
                <Button variant="outline" className="h-auto p-4 flex flex-col items-center space-y-2">
                  <Target className="h-6 w-6" />
                  <span>Update Goals</span>
                </Button>
              </>
            )}

            <Button variant="outline" className="h-auto p-4 flex flex-col items-center space-y-2">
              <Briefcase className="h-6 w-6" />
              <span>Generate Report</span>
            </Button>
            
            <Button variant="outline" className="h-auto p-4 flex flex-col items-center space-y-2">
              <Shield className="h-6 w-6" />
              <span>Compliance Check</span>
            </Button>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}

export default DashboardPage

