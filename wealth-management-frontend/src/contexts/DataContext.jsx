import { createContext, useContext, useState, useEffect } from 'react'
import { useAuth } from './AuthContext'

const DataContext = createContext({})

export const useData = () => {
  const context = useContext(DataContext)
  if (!context) {
    throw new Error('useData must be used within a DataProvider')
  }
  return context
}

export const DataProvider = ({ children }) => {
  const { user, isAuthenticated } = useAuth()
  const [clients, setClients] = useState([])
  const [portfolios, setPortfolios] = useState([])
  const [transactions, setTransactions] = useState([])
  const [marketData, setMarketData] = useState({})
  const [complianceAlerts, setComplianceAlerts] = useState([])
  const [loading, setLoading] = useState(false)

  // Mock data - in production this would come from your backend API
  const mockClients = [
    {
      id: 'client-001',
      name: 'Michael Chen',
      email: 'michael.chen@email.com',
      phone: '+1 (555) 123-4567',
      totalAssets: 2500000,
      riskTolerance: 'moderate',
      investmentObjective: 'balanced_growth',
      kycStatus: 'compliant',
      lastReview: '2024-01-15',
      portfolioIds: ['portfolio-001', 'portfolio-002']
    },
    {
      id: 'client-002',
      name: 'Sarah Williams',
      email: 'sarah.williams@email.com',
      phone: '+1 (555) 234-5678',
      totalAssets: 1800000,
      riskTolerance: 'conservative',
      investmentObjective: 'income_generation',
      kycStatus: 'compliant',
      lastReview: '2024-01-20',
      portfolioIds: ['portfolio-003']
    },
    {
      id: 'client-003',
      name: 'Robert Johnson',
      email: 'robert.johnson@email.com',
      phone: '+1 (555) 345-6789',
      totalAssets: 5200000,
      riskTolerance: 'aggressive',
      investmentObjective: 'capital_appreciation',
      kycStatus: 'pending_review',
      lastReview: '2024-01-10',
      portfolioIds: ['portfolio-004', 'portfolio-005']
    }
  ]

  const mockPortfolios = [
    {
      id: 'portfolio-001',
      clientId: 'client-001',
      name: 'Growth Portfolio',
      totalValue: 1500000,
      performance: {
        ytd: 8.5,
        oneYear: 12.3,
        threeYear: 9.8
      },
      allocation: {
        stocks: 70,
        bonds: 25,
        alternatives: 5
      },
      riskMetrics: {
        var95: 4.2,
        sharpeRatio: 1.35,
        volatility: 12.8
      },
      lastRebalanced: '2024-01-15'
    },
    {
      id: 'portfolio-002',
      clientId: 'client-001',
      name: 'Income Portfolio',
      totalValue: 1000000,
      performance: {
        ytd: 5.2,
        oneYear: 7.8,
        threeYear: 6.5
      },
      allocation: {
        stocks: 40,
        bonds: 55,
        alternatives: 5
      },
      riskMetrics: {
        var95: 2.8,
        sharpeRatio: 1.12,
        volatility: 8.5
      },
      lastRebalanced: '2024-01-20'
    },
    {
      id: 'portfolio-003',
      clientId: 'client-002',
      name: 'Conservative Portfolio',
      totalValue: 1800000,
      performance: {
        ytd: 4.1,
        oneYear: 6.2,
        threeYear: 5.8
      },
      allocation: {
        stocks: 30,
        bonds: 65,
        alternatives: 5
      },
      riskMetrics: {
        var95: 2.1,
        sharpeRatio: 0.95,
        volatility: 6.2
      },
      lastRebalanced: '2024-01-18'
    }
  ]

  const mockTransactions = [
    {
      id: 'txn-001',
      portfolioId: 'portfolio-001',
      clientId: 'client-001',
      type: 'buy',
      symbol: 'AAPL',
      quantity: 100,
      price: 185.50,
      amount: 18550,
      date: '2024-01-22',
      status: 'executed',
      fees: 9.95
    },
    {
      id: 'txn-002',
      portfolioId: 'portfolio-001',
      clientId: 'client-001',
      type: 'sell',
      symbol: 'MSFT',
      quantity: 50,
      price: 412.30,
      amount: 20615,
      date: '2024-01-21',
      status: 'executed',
      fees: 9.95
    },
    {
      id: 'txn-003',
      portfolioId: 'portfolio-002',
      clientId: 'client-001',
      type: 'buy',
      symbol: 'BND',
      quantity: 200,
      price: 78.25,
      amount: 15650,
      date: '2024-01-20',
      status: 'pending',
      fees: 0
    }
  ]

  const mockComplianceAlerts = [
    {
      id: 'alert-001',
      type: 'position_limit',
      severity: 'warning',
      clientId: 'client-001',
      portfolioId: 'portfolio-001',
      message: 'AAPL position exceeds 8% concentration limit',
      date: '2024-01-22',
      status: 'active'
    },
    {
      id: 'alert-002',
      type: 'kyc_review',
      severity: 'high',
      clientId: 'client-003',
      message: 'KYC review required - documentation pending',
      date: '2024-01-21',
      status: 'active'
    },
    {
      id: 'alert-003',
      type: 'risk_limit',
      severity: 'critical',
      clientId: 'client-002',
      portfolioId: 'portfolio-003',
      message: 'Portfolio VaR exceeds client risk tolerance',
      date: '2024-01-20',
      status: 'resolved'
    }
  ]

  const mockMarketData = {
    indices: {
      'SPY': { price: 485.20, change: 2.15, changePercent: 0.44 },
      'QQQ': { price: 412.80, change: -1.25, changePercent: -0.30 },
      'IWM': { price: 198.45, change: 0.85, changePercent: 0.43 }
    },
    currencies: {
      'EURUSD': { price: 1.0875, change: 0.0025, changePercent: 0.23 },
      'GBPUSD': { price: 1.2650, change: -0.0015, changePercent: -0.12 },
      'USDJPY': { price: 149.25, change: 0.45, changePercent: 0.30 }
    },
    commodities: {
      'GOLD': { price: 2025.50, change: 12.30, changePercent: 0.61 },
      'OIL': { price: 78.25, change: -0.85, changePercent: -1.07 },
      'SILVER': { price: 23.45, change: 0.15, changePercent: 0.64 }
    }
  }

  useEffect(() => {
    if (isAuthenticated) {
      loadData()
    }
  }, [isAuthenticated])

  const loadData = async () => {
    setLoading(true)
    try {
      // Simulate API calls
      await new Promise(resolve => setTimeout(resolve, 1000))
      
      setClients(mockClients)
      setPortfolios(mockPortfolios)
      setTransactions(mockTransactions)
      setMarketData(mockMarketData)
      setComplianceAlerts(mockComplianceAlerts)
    } catch (error) {
      console.error('Error loading data:', error)
    } finally {
      setLoading(false)
    }
  }

  const getClientById = (clientId) => {
    return clients.find(client => client.id === clientId)
  }

  const getPortfolioById = (portfolioId) => {
    return portfolios.find(portfolio => portfolio.id === portfolioId)
  }

  const getClientPortfolios = (clientId) => {
    return portfolios.filter(portfolio => portfolio.clientId === clientId)
  }

  const getPortfolioTransactions = (portfolioId) => {
    return transactions.filter(transaction => transaction.portfolioId === portfolioId)
  }

  const getActiveComplianceAlerts = () => {
    return complianceAlerts.filter(alert => alert.status === 'active')
  }

  const getTotalAUM = () => {
    return portfolios.reduce((total, portfolio) => total + portfolio.totalValue, 0)
  }

  const getClientCount = () => {
    return clients.length
  }

  const value = {
    // Data
    clients,
    portfolios,
    transactions,
    marketData,
    complianceAlerts,
    loading,
    
    // Actions
    loadData,
    
    // Getters
    getClientById,
    getPortfolioById,
    getClientPortfolios,
    getPortfolioTransactions,
    getActiveComplianceAlerts,
    getTotalAUM,
    getClientCount,
    
    // Setters
    setClients,
    setPortfolios,
    setTransactions,
    setMarketData,
    setComplianceAlerts
  }

  return (
    <DataContext.Provider value={value}>
      {children}
    </DataContext.Provider>
  )
}

