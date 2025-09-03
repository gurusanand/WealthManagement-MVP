import { useState } from 'react'
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom'
import { ThemeProvider } from '@/components/theme-provider'
import { Toaster } from '@/components/ui/toaster'
import { AuthProvider } from '@/contexts/AuthContext'
import { DataProvider } from '@/contexts/DataContext'

// Layout Components
import MainLayout from '@/components/layout/MainLayout'
import AuthLayout from '@/components/layout/AuthLayout'

// Page Components
import LoginPage from '@/pages/auth/LoginPage'
import DashboardPage from '@/pages/dashboard/DashboardPage'
import RMWorkbenchPage from '@/pages/rm-workbench/RMWorkbenchPage'
import ClientPortalPage from '@/pages/client-portal/ClientPortalPage'
import OperationsConsolePage from '@/pages/operations/OperationsConsolePage'
import CompliancePage from '@/pages/compliance/CompliancePage'
import PortfolioPage from '@/pages/portfolio/PortfolioPage'
import ClientsPage from '@/pages/clients/ClientsPage'
import ReportsPage from '@/pages/reports/ReportsPage'
import SettingsPage from '@/pages/settings/SettingsPage'
import AgentsPage from '@/pages/agents/AgentsPage'

// Protected Route Component
import ProtectedRoute from '@/components/auth/ProtectedRoute'

import './App.css'

function App() {
  return (
    <ThemeProvider defaultTheme="light" storageKey="wealth-management-theme">
      <AuthProvider>
        <DataProvider>
          <Router>
            <div className="min-h-screen bg-background">
              <Routes>
                {/* Authentication Routes */}
                <Route path="/auth/*" element={
                  <AuthLayout>
                    <Routes>
                      <Route path="login" element={<LoginPage />} />
                      <Route path="*" element={<Navigate to="/auth/login" replace />} />
                    </Routes>
                  </AuthLayout>
                } />

                {/* Protected Application Routes */}
                <Route path="/*" element={
                  <ProtectedRoute>
                    <MainLayout>
                      <Routes>
                        {/* Dashboard */}
                        <Route path="/" element={<DashboardPage />} />
                        <Route path="/dashboard" element={<DashboardPage />} />
                        
                        {/* RM Workbench */}
                        <Route path="/rm-workbench" element={<RMWorkbenchPage />} />
                        
                        {/* Client Portal */}
                        <Route path="/client-portal" element={<ClientPortalPage />} />
                        
                        {/* Operations Console */}
                        <Route path="/operations" element={<OperationsConsolePage />} />
                        
                        {/* Compliance */}
                        <Route path="/compliance" element={<CompliancePage />} />
                        
                        {/* Portfolio Management */}
                        <Route path="/portfolio" element={<PortfolioPage />} />
                        <Route path="/portfolio/:portfolioId" element={<PortfolioPage />} />
                        
                        {/* Client Management */}
                        <Route path="/clients" element={<ClientsPage />} />
                        <Route path="/clients/:clientId" element={<ClientsPage />} />
                        
                        {/* Agents */}
                        <Route path="/agents" element={<AgentsPage />} />
                        
                        {/* Reports */}
                        <Route path="/reports" element={<ReportsPage />} />
                        
                        {/* Settings */}
                        <Route path="/settings" element={<SettingsPage />} />
                        
                        {/* Catch all - redirect to dashboard */}
                        <Route path="*" element={<Navigate to="/dashboard" replace />} />
                      </Routes>
                    </MainLayout>
                  </ProtectedRoute>
                } />
              </Routes>
              
              {/* Global Toast Notifications */}
              <Toaster />
            </div>
          </Router>
        </DataProvider>
      </AuthProvider>
    </ThemeProvider>
  )
}

export default App

