import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../components/ui/card';
import { Button } from '../../components/ui/button';
import { Badge } from '../../components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../../components/ui/tabs';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line, AreaChart, Area } from 'recharts';
import { TrendingUp, TrendingDown, DollarSign, Target, BarChart3, PieChart as PieChartIcon, Settings, RefreshCw, Download, Plus, Edit, Trash2 } from 'lucide-react';

const PortfolioPage = () => {
  const [selectedPortfolio, setSelectedPortfolio] = useState('portfolio1');
  const [refreshing, setRefreshing] = useState(false);

  // Mock portfolio data
  const portfolioData = {
    portfolios: [
      { id: 'portfolio1', name: 'Growth Portfolio', value: 2450000, return: 12.8, risk: 'Medium', client: 'Johnson Holdings' },
      { id: 'portfolio2', name: 'Conservative Income', value: 1850000, return: 6.2, risk: 'Low', client: 'Smith Family Trust' },
      { id: 'portfolio3', name: 'Aggressive Growth', value: 3200000, return: 18.5, risk: 'High', client: 'Davis Corporation' }
    ],
    currentPortfolio: {
      name: 'Growth Portfolio',
      value: 2450000,
      return: 12.8,
      risk: 'Medium',
      client: 'Johnson Holdings',
      benchmark: 'S&P 500',
      benchmarkReturn: 10.2,
      sharpeRatio: 1.24,
      volatility: 14.2,
      maxDrawdown: -8.5,
      beta: 1.12
    },
    holdings: [
      { symbol: 'AAPL', name: 'Apple Inc.', shares: 2500, price: 185.20, value: 463000, weight: 18.9, return: 15.2, sector: 'Technology' },
      { symbol: 'MSFT', name: 'Microsoft Corp.', shares: 1800, price: 378.85, value: 681930, weight: 27.8, return: 22.1, sector: 'Technology' },
      { symbol: 'GOOGL', name: 'Alphabet Inc.', shares: 800, price: 142.56, value: 114048, weight: 4.7, return: 8.9, sector: 'Technology' },
      { symbol: 'AMZN', name: 'Amazon.com Inc.', shares: 1200, price: 155.89, value: 187068, weight: 7.6, return: -2.1, sector: 'Consumer Discretionary' },
      { symbol: 'TSLA', name: 'Tesla Inc.', shares: 900, price: 248.50, value: 223650, weight: 9.1, return: 35.8, sector: 'Consumer Discretionary' },
      { symbol: 'NVDA', name: 'NVIDIA Corp.', shares: 600, price: 875.28, value: 525168, weight: 21.4, return: 89.2, sector: 'Technology' },
      { symbol: 'JPM', name: 'JPMorgan Chase', shares: 1500, price: 168.50, value: 252750, weight: 10.3, return: 18.7, sector: 'Financial Services' }
    ],
    allocation: [
      { name: 'Technology', value: 65.0, color: '#3b82f6' },
      { name: 'Financial Services', value: 10.3, color: '#10b981' },
      { name: 'Consumer Discretionary', value: 16.7, color: '#f59e0b' },
      { name: 'Healthcare', value: 5.2, color: '#ef4444' },
      { name: 'Cash', value: 2.8, color: '#6b7280' }
    ],
    performance: [
      { date: '2023-07', portfolio: 8.2, benchmark: 7.1, alpha: 1.1 },
      { date: '2023-08', portfolio: 9.5, benchmark: 8.3, alpha: 1.2 },
      { date: '2023-09', portfolio: 7.8, benchmark: 6.9, alpha: 0.9 },
      { date: '2023-10', portfolio: 11.2, benchmark: 9.8, alpha: 1.4 },
      { date: '2023-11', portfolio: 13.1, benchmark: 11.5, alpha: 1.6 },
      { date: '2023-12', portfolio: 12.8, benchmark: 10.2, alpha: 2.6 }
    ],
    riskMetrics: [
      { metric: 'Value at Risk (95%)', value: '-$156,800', status: 'normal' },
      { metric: 'Expected Shortfall', value: '-$234,500', status: 'normal' },
      { metric: 'Maximum Drawdown', value: '-8.5%', status: 'good' },
      { metric: 'Sharpe Ratio', value: '1.24', status: 'good' },
      { metric: 'Beta', value: '1.12', status: 'normal' },
      { metric: 'Tracking Error', value: '3.2%', status: 'normal' }
    ],
    rebalancing: [
      { asset: 'Technology Stocks', current: 65.0, target: 60.0, action: 'Sell', amount: '$122,500' },
      { asset: 'Financial Services', current: 10.3, target: 15.0, action: 'Buy', amount: '$115,150' },
      { asset: 'Healthcare', current: 5.2, target: 10.0, action: 'Buy', amount: '$117,600' },
      { asset: 'Bonds', current: 0.0, target: 5.0, action: 'Buy', amount: '$122,500' },
      { asset: 'Cash', current: 2.8, target: 2.0, action: 'Reduce', amount: '$19,600' }
    ]
  };

  const handleRefresh = () => {
    setRefreshing(true);
    setTimeout(() => setRefreshing(false), 2000);
  };

  const getReturnColor = (returnValue) => {
    return returnValue >= 0 ? 'text-green-600' : 'text-red-600';
  };

  const getReturnIcon = (returnValue) => {
    return returnValue >= 0 ? <TrendingUp className="h-4 w-4" /> : <TrendingDown className="h-4 w-4" />;
  };

  const getRiskColor = (risk) => {
    switch (risk) {
      case 'Low': return 'bg-green-100 text-green-800';
      case 'Medium': return 'bg-yellow-100 text-yellow-800';
      case 'High': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'good': return 'text-green-600';
      case 'normal': return 'text-blue-600';
      case 'warning': return 'text-yellow-600';
      case 'danger': return 'text-red-600';
      default: return 'text-gray-600';
    }
  };

  const getActionColor = (action) => {
    switch (action) {
      case 'Buy': return 'bg-green-100 text-green-800';
      case 'Sell': return 'bg-red-100 text-red-800';
      case 'Reduce': return 'bg-yellow-100 text-yellow-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Portfolio Management</h1>
          <p className="text-gray-600">Monitor and manage client portfolios, holdings, and performance</p>
        </div>
        <div className="flex space-x-2">
          <Button variant="outline" size="sm" onClick={handleRefresh} disabled={refreshing}>
            <RefreshCw className={`h-4 w-4 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Button variant="outline" size="sm">
            <Download className="h-4 w-4 mr-2" />
            Export
          </Button>
          <Button size="sm">
            <Plus className="h-4 w-4 mr-2" />
            New Portfolio
          </Button>
        </div>
      </div>

      {/* Portfolio Selector */}
      <Card>
        <CardHeader>
          <CardTitle>Portfolio Selection</CardTitle>
          <CardDescription>Select a portfolio to view detailed analysis</CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {portfolioData.portfolios.map((portfolio) => (
              <div
                key={portfolio.id}
                className={`p-4 border rounded-lg cursor-pointer transition-colors ${
                  selectedPortfolio === portfolio.id ? 'border-blue-500 bg-blue-50' : 'border-gray-200 hover:border-gray-300'
                }`}
                onClick={() => setSelectedPortfolio(portfolio.id)}
              >
                <div className="flex justify-between items-start">
                  <div>
                    <h3 className="font-medium">{portfolio.name}</h3>
                    <p className="text-sm text-gray-600">{portfolio.client}</p>
                    <p className="text-lg font-bold">${portfolio.value.toLocaleString()}</p>
                  </div>
                  <div className="text-right">
                    <div className={`flex items-center ${getReturnColor(portfolio.return)}`}>
                      {getReturnIcon(portfolio.return)}
                      <span className="ml-1 font-medium">{portfolio.return}%</span>
                    </div>
                    <Badge className={getRiskColor(portfolio.risk)} size="sm">
                      {portfolio.risk} Risk
                    </Badge>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Portfolio Overview */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Portfolio Value</CardTitle>
            <DollarSign className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">${portfolioData.currentPortfolio.value.toLocaleString()}</div>
            <p className="text-xs text-muted-foreground">
              Total portfolio value
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">YTD Return</CardTitle>
            <TrendingUp className="h-4 w-4 text-green-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">+{portfolioData.currentPortfolio.return}%</div>
            <p className="text-xs text-muted-foreground">
              vs {portfolioData.currentPortfolio.benchmark}: +{portfolioData.currentPortfolio.benchmarkReturn}%
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Sharpe Ratio</CardTitle>
            <Target className="h-4 w-4 text-blue-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-blue-600">{portfolioData.currentPortfolio.sharpeRatio}</div>
            <p className="text-xs text-muted-foreground">
              Risk-adjusted return
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Volatility</CardTitle>
            <BarChart3 className="h-4 w-4 text-yellow-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-yellow-600">{portfolioData.currentPortfolio.volatility}%</div>
            <p className="text-xs text-muted-foreground">
              Annualized volatility
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Main Content Tabs */}
      <Tabs defaultValue="holdings" className="space-y-4">
        <TabsList>
          <TabsTrigger value="holdings">Holdings</TabsTrigger>
          <TabsTrigger value="allocation">Allocation</TabsTrigger>
          <TabsTrigger value="performance">Performance</TabsTrigger>
          <TabsTrigger value="risk">Risk Analysis</TabsTrigger>
          <TabsTrigger value="rebalancing">Rebalancing</TabsTrigger>
        </TabsList>

        <TabsContent value="holdings" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Portfolio Holdings</CardTitle>
              <CardDescription>Individual securities and their performance</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {portfolioData.holdings.map((holding, index) => (
                  <div key={index} className="p-4 border rounded-lg">
                    <div className="flex items-center justify-between">
                      <div className="flex items-center space-x-4">
                        <div>
                          <p className="font-medium">{holding.symbol}</p>
                          <p className="text-sm text-gray-600">{holding.name}</p>
                          <p className="text-sm text-gray-600">{holding.sector}</p>
                        </div>
                      </div>
                      <div className="text-right">
                        <p className="font-medium">${holding.value.toLocaleString()}</p>
                        <p className="text-sm text-gray-600">{holding.shares} shares @ ${holding.price}</p>
                        <div className={`flex items-center justify-end ${getReturnColor(holding.return)}`}>
                          {getReturnIcon(holding.return)}
                          <span className="ml-1 text-sm">{holding.return}%</span>
                        </div>
                      </div>
                      <div className="text-right ml-4">
                        <p className="text-lg font-bold">{holding.weight}%</p>
                        <p className="text-sm text-gray-600">Weight</p>
                      </div>
                    </div>
                    <div className="mt-3 flex space-x-2">
                      <Button size="sm" variant="outline">
                        <Edit className="h-4 w-4 mr-2" />
                        Edit
                      </Button>
                      <Button size="sm" variant="outline">View Details</Button>
                      <Button size="sm" variant="outline">
                        <Trash2 className="h-4 w-4 mr-2" />
                        Remove
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="allocation" className="space-y-4">
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <Card>
              <CardHeader>
                <CardTitle>Asset Allocation</CardTitle>
                <CardDescription>Current portfolio allocation by sector</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={portfolioData.allocation}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={120}
                      paddingAngle={5}
                      dataKey="value"
                    >
                      {portfolioData.allocation.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value) => [`${value}%`, '']} />
                  </PieChart>
                </ResponsiveContainer>
                <div className="grid grid-cols-1 gap-2 mt-4">
                  {portfolioData.allocation.map((item, index) => (
                    <div key={index} className="flex items-center justify-between">
                      <div className="flex items-center space-x-2">
                        <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }}></div>
                        <span className="text-sm">{item.name}</span>
                      </div>
                      <span className="text-sm font-medium">{item.value}%</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>Allocation Breakdown</CardTitle>
                <CardDescription>Detailed allocation analysis</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={portfolioData.allocation}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" angle={-45} textAnchor="end" height={80} />
                    <YAxis />
                    <Tooltip formatter={(value) => [`${value}%`, 'Allocation']} />
                    <Bar dataKey="value" fill="#3b82f6" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="performance" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Performance vs Benchmark</CardTitle>
              <CardDescription>Portfolio performance compared to {portfolioData.currentPortfolio.benchmark}</CardDescription>
            </CardHeader>
            <CardContent>
              <ResponsiveContainer width="100%" height={400}>
                <AreaChart data={portfolioData.performance}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="date" />
                  <YAxis />
                  <Tooltip />
                  <Area type="monotone" dataKey="portfolio" stackId="1" stroke="#3b82f6" fill="#3b82f6" fillOpacity={0.6} name="Portfolio" />
                  <Area type="monotone" dataKey="benchmark" stackId="2" stroke="#10b981" fill="#10b981" fillOpacity={0.6} name="Benchmark" />
                  <Line type="monotone" dataKey="alpha" stroke="#ef4444" strokeWidth={2} name="Alpha" />
                </AreaChart>
              </ResponsiveContainer>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="risk" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Risk Metrics</CardTitle>
              <CardDescription>Comprehensive risk analysis and metrics</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {portfolioData.riskMetrics.map((metric, index) => (
                  <div key={index} className="p-4 border rounded-lg">
                    <div className="flex justify-between items-center">
                      <div>
                        <p className="font-medium">{metric.metric}</p>
                        <p className={`text-lg font-bold ${getStatusColor(metric.status)}`}>
                          {metric.value}
                        </p>
                      </div>
                      <div className={`w-3 h-3 rounded-full ${
                        metric.status === 'good' ? 'bg-green-500' :
                        metric.status === 'normal' ? 'bg-blue-500' :
                        metric.status === 'warning' ? 'bg-yellow-500' : 'bg-red-500'
                      }`}></div>
                    </div>
                  </div>
                ))}
              </div>
            </CardContent>
          </Card>
        </TabsContent>

        <TabsContent value="rebalancing" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Portfolio Rebalancing</CardTitle>
              <CardDescription>Recommended rebalancing actions to maintain target allocation</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {portfolioData.rebalancing.map((item, index) => (
                  <div key={index} className="p-4 border rounded-lg">
                    <div className="flex items-center justify-between">
                      <div>
                        <p className="font-medium">{item.asset}</p>
                        <p className="text-sm text-gray-600">Current: {item.current}% → Target: {item.target}%</p>
                      </div>
                      <div className="text-right">
                        <Badge className={getActionColor(item.action)}>
                          {item.action}
                        </Badge>
                        <p className="text-sm font-medium mt-1">{item.amount}</p>
                      </div>
                    </div>
                  </div>
                ))}
                <div className="mt-6 flex space-x-2">
                  <Button>Execute Rebalancing</Button>
                  <Button variant="outline">Preview Orders</Button>
                  <Button variant="outline">Save as Draft</Button>
                </div>
              </div>
            </CardContent>
          </Card>
        </TabsContent>
      </Tabs>
    </div>
  );
};

export default PortfolioPage;

