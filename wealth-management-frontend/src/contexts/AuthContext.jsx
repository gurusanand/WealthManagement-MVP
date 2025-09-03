import { createContext, useContext, useState, useEffect } from 'react'

const AuthContext = createContext({})

export const useAuth = () => {
  const context = useContext(AuthContext)
  if (!context) {
    throw new Error('useAuth must be used within an AuthProvider')
  }
  return context
}

export const AuthProvider = ({ children }) => {
  const [user, setUser] = useState(null)
  const [loading, setLoading] = useState(true)
  const [isAuthenticated, setIsAuthenticated] = useState(false)

  // Mock user data - in production this would come from your backend
  const mockUsers = {
    'rm@wealth.com': {
      id: 'rm-001',
      email: 'rm@wealth.com',
      name: 'Sarah Johnson',
      role: 'relationship_manager',
      permissions: ['view_clients', 'manage_portfolios', 'execute_trades', 'view_reports'],
      avatar: '/api/placeholder/40/40'
    },
    'client@wealth.com': {
      id: 'client-001',
      email: 'client@wealth.com',
      name: 'Michael Chen',
      role: 'client',
      permissions: ['view_own_portfolio', 'view_own_reports'],
      avatar: '/api/placeholder/40/40'
    },
    'ops@wealth.com': {
      id: 'ops-001',
      email: 'ops@wealth.com',
      name: 'Jennifer Martinez',
      role: 'operations',
      permissions: ['view_all_clients', 'manage_compliance', 'system_admin', 'view_all_reports'],
      avatar: '/api/placeholder/40/40'
    },
    'admin@wealth.com': {
      id: 'admin-001',
      email: 'admin@wealth.com',
      name: 'David Wilson',
      role: 'admin',
      permissions: ['full_access'],
      avatar: '/api/placeholder/40/40'
    }
  }

  useEffect(() => {
    // Check for existing session
    const storedUser = localStorage.getItem('wealth_management_user')
    if (storedUser) {
      try {
        const userData = JSON.parse(storedUser)
        setUser(userData)
        setIsAuthenticated(true)
      } catch (error) {
        console.error('Error parsing stored user data:', error)
        localStorage.removeItem('wealth_management_user')
      }
    }
    setLoading(false)
  }, [])

  const login = async (email, password) => {
    try {
      setLoading(true)
      
      // Mock authentication - in production this would be an API call
      await new Promise(resolve => setTimeout(resolve, 1000)) // Simulate API delay
      
      const userData = mockUsers[email]
      if (userData && password === 'password') { // Mock password check
        setUser(userData)
        setIsAuthenticated(true)
        localStorage.setItem('wealth_management_user', JSON.stringify(userData))
        return { success: true, user: userData }
      } else {
        throw new Error('Invalid credentials')
      }
    } catch (error) {
      console.error('Login error:', error)
      return { success: false, error: error.message }
    } finally {
      setLoading(false)
    }
  }

  const logout = () => {
    setUser(null)
    setIsAuthenticated(false)
    localStorage.removeItem('wealth_management_user')
  }




  const hasPermission = (permission) => {
    if (!user) return false
    if (user.permissions.includes('full_access')) return true
    return user.permissions.includes(permission)
  }

  const getUserRole = () => {
    return user?.role || 'guest'
  }

  const value = {
    user,
    loading,
    isAuthenticated,
    login,
    logout,
    hasPermission,
    getUserRole
  }

  return (
    <AuthContext.Provider value={value}>
      {children}
    </AuthContext.Provider>
  )
}

