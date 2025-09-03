import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '../../components/ui/card';
import { Button } from '../../components/ui/button';
import { Badge } from '../../components/ui/badge';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '../../components/ui/tabs';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line } from 'recharts';
import { Users, DollarSign, TrendingUp, AlertTriangle, Search, Filter, Plus, Edit, Eye, Phone, Mail, MapPin, Calendar, Star, RefreshCw } from 'lucide-react';

const ClientsPage = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedFilter, setSelectedFilter] = useState('all');
  const [refreshing, setRefreshing] = useState(false);

  // Mock client data
  const clientData = {
    overview: {
      totalClients: 156,
      activeClients: 142,
      newThisMonth: 8,
      totalAUM: 24500000,
      averageAUM: 157051
    },
    clients: [
      {
        id: 1,
        name: 'Johnson Holdings LLC',
        type: 'Corporate',
        aum: 4300000,
        return: 12.8,
        riskProfile: 'Moderate',
        status: 'Active',
        lastContact: '2024-01-20',
        email: 'contact@johnsonholdings.com',
        phone: '+1 (555) 123-4567',
        location: 'New York, NY',
        joinDate: '2019-03-15',
        rating: 5,
        compliance: 'Compliant',
        portfolios: 3
      },
      {
        id: 2,
        name: 'Smith Family Trust',
        type: 'Trust',
        aum: 2850000,
        return: 8.2,
        riskProfile: 'Conservative',
        status: 'Active',
        lastContact: '2024-01-18',
        email: 'trustees@smithfamily.com',
        phone: '+1 (555) 234-5678',
        location: 'San Francisco, CA',
        joinDate: '2020-07-22',
        rating: 4,
        compliance: 'Compliant',
        portfolios: 2
      },
      {
        id: 3,
        name: 'Davis Corporation',
        type: 'Corporate',
        aum: 6200000,
        return: 15.6,
        riskProfile: 'Aggressive',
        status: 'Active',
        lastContact: '2024-01-22',
        email: 'finance@daviscorp.com',
        phone: '+1 (555) 345-6789',
        location: 'Chicago, IL',
        joinDate: '2018-11-08',
        rating: 5,
        compliance: 'Compliant',
        portfolios: 4
      },
      {
        id: 4,
        name: 'Wilson Enterprises',
        type: 'Corporate',
        aum: 1950000,
        return: 6.8,
        riskProfile: 'Moderate',
        status: 'Review Required',
        lastContact: '2024-01-10',
        email: 'admin@wilsonent.com',
        phone: '+1 (555) 456-7890',
        location: 'Austin, TX',
        joinDate: '2021-02-14',
        rating: 3,
        compliance: 'Pending Review',
        portfolios: 2
      },
      {
        id: 5,
        name: 'Anderson LLC',
        type: 'Corporate',
        aum: 3400000,
        return: 11.2,
        riskProfile: 'Moderate',
        status: 'Active',
        lastContact: '2024-01-19',
        email: 'info@andersonllc.com',
        phone: '+1 (555) 567-8901',
        location: 'Seattle, WA',
        joinDate: '2019-09-30',
        rating: 4,
        compliance: 'Compliant',
        portfolios: 3
      }
    ],
    clientTypes: [
      { name: 'Corporate', value: 45, color: '#3b82f6' },
      { name: 'Trust', value: 25, color: '#10b981' },
      { name: 'Individual', value: 20, color: '#f59e0b' },
      { name: 'Foundation', value: 10, color: '#ef4444' }
    ],
    aumDistribution: [
      { range: '$0-1M', count: 45, percentage: 28.8 },
      { range: '$1-5M', count: 68, percentage: 43.6 },
      { range: '$5-10M', count: 32, percentage: 20.5 },
      { range: '$10M+', count: 11, percentage: 7.1 }
    ],
    clientGrowth: [
      { month: 'Jul', clients: 142, aum: 22.1 },
      { month: 'Aug', clients: 145, aum: 22.8 },
      { month: 'Sep', clients: 148, aum: 23.2 },
      { month: 'Oct', clients: 151, aum: 23.9 },
      { month: 'Nov', clients: 154, aum: 24.1 },
      { month: 'Dec', clients: 156, aum: 24.5 }
    ],
    recentActivities: [
      { id: 1, client: 'Johnson Holdings', activity: 'Portfolio Rebalancing', date: '2024-01-22', type: 'portfolio' },
      { id: 2, client: 'Davis Corporation', activity: 'New Investment Proposal', date: '2024-01-21', type: 'proposal' },
      { id: 3, client: 'Smith Family Trust', activity: 'Quarterly Review Meeting', date: '2024-01-20', type: 'meeting' },
      { id: 4, client: 'Anderson LLC', activity: 'Risk Assessment Update', date: '2024-01-19', type: 'assessment' }
    ]
  };

  const handleRefresh = () => {
    setRefreshing(true);
    setTimeout(() => setRefreshing(false), 2000);
  };

  const getStatusColor = (status) => {
    switch (status) {
      case 'Active': return 'bg-green-100 text-green-800';
      case 'Review Required': return 'bg-yellow-100 text-yellow-800';
      case 'Inactive': return 'bg-gray-100 text-gray-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getRiskColor = (risk) => {
    switch (risk) {
      case 'Conservative': return 'bg-green-100 text-green-800';
      case 'Moderate': return 'bg-yellow-100 text-yellow-800';
      case 'Aggressive': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getComplianceColor = (compliance) => {
    switch (compliance) {
      case 'Compliant': return 'bg-green-100 text-green-800';
      case 'Pending Review': return 'bg-yellow-100 text-yellow-800';
      case 'Non-Compliant': return 'bg-red-100 text-red-800';
      default: return 'bg-gray-100 text-gray-800';
    }
  };

  const getReturnColor = (returnValue) => {
    return returnValue >= 0 ? 'text-green-600' : 'text-red-600';
  };

  const renderStars = (rating) => {
    return Array.from({ length: 5 }, (_, i) => (
      <Star
        key={i}
        className={`h-4 w-4 ${i < rating ? 'text-yellow-400 fill-current' : 'text-gray-300'}`}
      />
    ));
  };

  const filteredClients = clientData.clients.filter(client => {
    const matchesSearch = client.name.toLowerCase().includes(searchTerm.toLowerCase()) ||
                         client.email.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesFilter = selectedFilter === 'all' || 
                         (selectedFilter === 'active' && client.status === 'Active') ||
                         (selectedFilter === 'review' && client.status === 'Review Required') ||
                         (selectedFilter === 'corporate' && client.type === 'Corporate') ||
                         (selectedFilter === 'trust' && client.type === 'Trust');
    return matchesSearch && matchesFilter;
  });

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Client Management</h1>
          <p className="text-gray-600">Manage client relationships, portfolios, and communications</p>
        </div>
        <div className="flex space-x-2">
          <Button variant="outline" size="sm" onClick={handleRefresh} disabled={refreshing}>
            <RefreshCw className={`h-4 w-4 mr-2 ${refreshing ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Button size="sm">
            <Plus className="h-4 w-4 mr-2" />
            Add Client
          </Button>
        </div>
      </div>

      {/* Client Overview */}
      <div className="grid grid-cols-1 md:grid-cols-5 gap-6">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total Clients</CardTitle>
            <Users className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{clientData.overview.totalClients}</div>
            <p className="text-xs text-muted-foreground">
              +{clientData.overview.newThisMonth} this month
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Active Clients</CardTitle>
            <Users className="h-4 w-4 text-green-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-green-600">{clientData.overview.activeClients}</div>
            <p className="text-xs text-muted-foreground">
              {((clientData.overview.activeClients / clientData.overview.totalClients) * 100).toFixed(1)}% of total
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Total AUM</CardTitle>
            <DollarSign className="h-4 w-4 text-blue-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-blue-600">${(clientData.overview.totalAUM / 1000000).toFixed(1)}M</div>
            <p className="text-xs text-muted-foreground">
              Assets under management
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Average AUM</CardTitle>
            <TrendingUp className="h-4 w-4 text-purple-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-purple-600">${(clientData.overview.averageAUM / 1000).toFixed(0)}K</div>
            <p className="text-xs text-muted-foreground">
              Per client average
            </p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">New This Month</CardTitle>
            <Plus className="h-4 w-4 text-orange-600" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold text-orange-600">{clientData.overview.newThisMonth}</div>
            <p className="text-xs text-muted-foreground">
              New client acquisitions
            </p>
          </CardContent>
        </Card>
      </div>

      {/* Search and Filter */}
      <Card>
        <CardContent className="pt-6">
          <div className="flex flex-col md:flex-row gap-4">
            <div className="flex-1">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 text-gray-400 h-4 w-4" />
                <input
                  type="text"
                  placeholder="Search clients by name or email..."
                  className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                  value={searchTerm}
                  onChange={(e) => setSearchTerm(e.target.value)}
                />
              </div>
            </div>
            <div className="flex gap-2">
              <select
                className="px-3 py-2 border border-gray-300 rounded-md focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                value={selectedFilter}
                onChange={(e) => setSelectedFilter(e.target.value)}
              >
                <option value="all">All Clients</option>
                <option value="active">Active</option>
                <option value="review">Review Required</option>
                <option value="corporate">Corporate</option>
                <option value="trust">Trust</option>
              </select>
              <Button variant="outline" size="sm">
                <Filter className="h-4 w-4 mr-2" />
                More Filters
              </Button>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Main Content Tabs */}
      <Tabs defaultValue="clients" className="space-y-4">
        <TabsList>
          <TabsTrigger value="clients">Client List</TabsTrigger>
          <TabsTrigger value="analytics">Analytics</TabsTrigger>
          <TabsTrigger value="activities">Recent Activities</TabsTrigger>
        </TabsList>

        <TabsContent value="clients" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Client Directory</CardTitle>
              <CardDescription>Comprehensive client information and management</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {filteredClients.map((client) => (
                  <div key={client.id} className="p-6 border rounded-lg hover:shadow-md transition-shadow">
                    <div className="flex items-start justify-between">
                      <div className="flex-1">
                        <div className="flex items-center space-x-4 mb-3">
                          <div>
                            <h3 className="text-lg font-semibold">{client.name}</h3>
                            <div className="flex items-center space-x-2 mt-1">
                              <Badge className={getStatusColor(client.status)} size="sm">
                                {client.status}
                              </Badge>
                              <Badge variant="outline" size="sm">
                                {client.type}
                              </Badge>
                              <Badge className={getRiskColor(client.riskProfile)} size="sm">
                                {client.riskProfile}
                              </Badge>
                            </div>
                          </div>
                        </div>
                        
                        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-4">
                          <div>
                            <p className="text-sm text-gray-600">Assets Under Management</p>
                            <p className="text-xl font-bold">${(client.aum / 1000000).toFixed(2)}M</p>
                            <p className={`text-sm ${getReturnColor(client.return)}`}>
                              {client.return >= 0 ? '+' : ''}{client.return}% YTD
                            </p>
                          </div>
                          <div>
                            <p className="text-sm text-gray-600">Contact Information</p>
                            <div className="space-y-1">
                              <div className="flex items-center text-sm">
                                <Mail className="h-3 w-3 mr-2 text-gray-400" />
                                {client.email}
                              </div>
                              <div className="flex items-center text-sm">
                                <Phone className="h-3 w-3 mr-2 text-gray-400" />
                                {client.phone}
                              </div>
                              <div className="flex items-center text-sm">
                                <MapPin className="h-3 w-3 mr-2 text-gray-400" />
                                {client.location}
                              </div>
                            </div>
                          </div>
                          <div>
                            <p className="text-sm text-gray-600">Account Details</p>
                            <div className="space-y-1">
                              <div className="flex items-center text-sm">
                                <Calendar className="h-3 w-3 mr-2 text-gray-400" />
                                Joined: {client.joinDate}
                              </div>
                              <div className="flex items-center text-sm">
                                <span className="text-gray-400 mr-2">Portfolios:</span>
                                {client.portfolios}
                              </div>
                              <div className="flex items-center text-sm">
                                <span className="text-gray-400 mr-2">Rating:</span>
                                <div className="flex">{renderStars(client.rating)}</div>
                              </div>
                            </div>
                          </div>
                        </div>

                        <div className="flex items-center justify-between">
                          <div className="flex items-center space-x-4">
                            <Badge className={getComplianceColor(client.compliance)} size="sm">
                              {client.compliance}
                            </Badge>
                            <span className="text-sm text-gray-600">
                              Last Contact: {client.lastContact}
                            </span>
                          </div>
                          <div className="flex space-x-2">
                            <Button size="sm" variant="outline">
                              <Eye className="h-4 w-4 mr-2" />
                              View
                            </Button>
                            <Button size="sm" variant="outline">
                              <Edit className="h-4 w-4 mr-2" />
                              Edit
                            </Button>
                            <Button size="sm" variant="outline">
                              <Phone className="h-4 w-4 mr-2" />
                              Contact
                            </Button>
                          </div>
                        </div>
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
                <CardTitle>Client Type Distribution</CardTitle>
                <CardDescription>Breakdown of clients by type</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <PieChart>
                    <Pie
                      data={clientData.clientTypes}
                      cx="50%"
                      cy="50%"
                      innerRadius={60}
                      outerRadius={120}
                      paddingAngle={5}
                      dataKey="value"
                    >
                      {clientData.clientTypes.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.color} />
                      ))}
                    </Pie>
                    <Tooltip formatter={(value) => [`${value}%`, '']} />
                  </PieChart>
                </ResponsiveContainer>
                <div className="grid grid-cols-2 gap-2 mt-4">
                  {clientData.clientTypes.map((item, index) => (
                    <div key={index} className="flex items-center space-x-2">
                      <div className="w-3 h-3 rounded-full" style={{ backgroundColor: item.color }}></div>
                      <span className="text-sm">{item.name}: {item.value}%</span>
                    </div>
                  ))}
                </div>
              </CardContent>
            </Card>

            <Card>
              <CardHeader>
                <CardTitle>AUM Distribution</CardTitle>
                <CardDescription>Client distribution by assets under management</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <BarChart data={clientData.aumDistribution}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="range" />
                    <YAxis />
                    <Tooltip />
                    <Bar dataKey="count" fill="#3b82f6" name="Number of Clients" />
                  </BarChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>

            <Card className="lg:col-span-2">
              <CardHeader>
                <CardTitle>Client Growth Trend</CardTitle>
                <CardDescription>Client count and AUM growth over time</CardDescription>
              </CardHeader>
              <CardContent>
                <ResponsiveContainer width="100%" height={300}>
                  <LineChart data={clientData.clientGrowth}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="month" />
                    <YAxis yAxisId="left" />
                    <YAxis yAxisId="right" orientation="right" />
                    <Tooltip />
                    <Line yAxisId="left" type="monotone" dataKey="clients" stroke="#3b82f6" strokeWidth={2} name="Client Count" />
                    <Line yAxisId="right" type="monotone" dataKey="aum" stroke="#10b981" strokeWidth={2} name="AUM ($M)" />
                  </LineChart>
                </ResponsiveContainer>
              </CardContent>
            </Card>
          </div>
        </TabsContent>

        <TabsContent value="activities" className="space-y-4">
          <Card>
            <CardHeader>
              <CardTitle>Recent Client Activities</CardTitle>
              <CardDescription>Latest client interactions and updates</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="space-y-4">
                {clientData.recentActivities.map((activity) => (
                  <div key={activity.id} className="flex items-center justify-between p-4 border rounded-lg">
                    <div className="flex items-center space-x-4">
                      <div className={`w-3 h-3 rounded-full ${
                        activity.type === 'portfolio' ? 'bg-blue-500' :
                        activity.type === 'proposal' ? 'bg-green-500' :
                        activity.type === 'meeting' ? 'bg-purple-500' : 'bg-orange-500'
                      }`}></div>
                      <div>
                        <p className="font-medium">{activity.activity}</p>
                        <p className="text-sm text-gray-600">Client: {activity.client}</p>
                      </div>
                    </div>
                    <div className="text-right">
                      <p className="text-sm text-gray-600">{activity.date}</p>
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

export default ClientsPage;

