// API Service for Backend Integration
const API_BASE_URL = 'http://localhost:5000/api';

class ApiService {
  constructor() {
    this.baseURL = API_BASE_URL;
  }

  async request(endpoint, options = {}) {
    const url = `${this.baseURL}${endpoint}`;
    const config = {
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
      ...options,
    };

    try {
      const response = await fetch(url, config);
      
      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }
      
      return await response.json();
    } catch (error) {
      console.error('API request failed:', error);
      throw error;
    }
  }

  // Health check
  async healthCheck() {
    return this.request('/health');
  }

  // Client endpoints
  async getClients() {
    return this.request('/clients');
  }

  async getClient(id) {
    return this.request(`/clients/${id}`);
  }

  async createClient(clientData) {
    return this.request('/clients', {
      method: 'POST',
      body: JSON.stringify(clientData),
    });
  }

  async updateClient(id, clientData) {
    return this.request(`/clients/${id}`, {
      method: 'PUT',
      body: JSON.stringify(clientData),
    });
  }

  // Portfolio endpoints
  async getPortfolios() {
    return this.request('/portfolios');
  }

  async getPortfolio(id) {
    return this.request(`/portfolios/${id}`);
  }

  async createPortfolio(portfolioData) {
    return this.request('/portfolios', {
      method: 'POST',
      body: JSON.stringify(portfolioData),
    });
  }

  async optimizePortfolio(id, optimizationData) {
    return this.request(`/portfolios/${id}/optimize`, {
      method: 'POST',
      body: JSON.stringify(optimizationData),
    });
  }

  async getPortfolioPerformance(id) {
    return this.request(`/portfolios/${id}/performance`);
  }

  async getPortfolioRisk(id) {
    return this.request(`/portfolios/${id}/risk`);
  }

  // Holdings endpoints
  async getHoldings(portfolioId) {
    return this.request(`/portfolios/${portfolioId}/holdings`);
  }

  async addHolding(portfolioId, holdingData) {
    return this.request(`/portfolios/${portfolioId}/holdings`, {
      method: 'POST',
      body: JSON.stringify(holdingData),
    });
  }

  async updateHolding(portfolioId, holdingId, holdingData) {
    return this.request(`/portfolios/${portfolioId}/holdings/${holdingId}`, {
      method: 'PUT',
      body: JSON.stringify(holdingData),
    });
  }

  async deleteHolding(portfolioId, holdingId) {
    return this.request(`/portfolios/${portfolioId}/holdings/${holdingId}`, {
      method: 'DELETE',
    });
  }

  // Event endpoints
  async getEvents() {
    return this.request('/events');
  }

  async createEvent(eventData) {
    return this.request('/events', {
      method: 'POST',
      body: JSON.stringify(eventData),
    });
  }

  async processEvent(id) {
    return this.request(`/events/${id}/process`, {
      method: 'POST',
    });
  }

  // External data endpoints
  async getMarketData() {
    return this.request('/external-data/market');
  }

  async getNewsData() {
    return this.request('/external-data/news');
  }

  async getWeatherData() {
    return this.request('/external-data/weather');
  }

  // Agent endpoints
  async getAgentHealth() {
    return this.request('/agents/health');
  }

  async startWorkflow(workflowData) {
    return this.request('/agents/workflows', {
      method: 'POST',
      body: JSON.stringify(workflowData),
    });
  }

  async sendAgentMessage(agentId, message) {
    return this.request(`/agents/agents/${agentId}/message`, {
      method: 'POST',
      body: JSON.stringify({ message }),
    });
  }

  async rebalancePortfolio(portfolioId, rebalanceData) {
    return this.request(`/agents/portfolios/${portfolioId}/rebalance`, {
      method: 'POST',
      body: JSON.stringify(rebalanceData),
    });
  }

  async generateClientCommunication(clientId, communicationData) {
    return this.request(`/agents/clients/${clientId}/communicate`, {
      method: 'POST',
      body: JSON.stringify(communicationData),
    });
  }

  // Compliance endpoints
  async getComplianceStatus() {
    return this.request('/compliance/status');
  }

  async getKYCStatus() {
    return this.request('/compliance/kyc');
  }

  async getAMLAlerts() {
    return this.request('/compliance/aml');
  }

  async getPolicyViolations() {
    return this.request('/compliance/violations');
  }

  async getAuditTrail() {
    return this.request('/compliance/audit');
  }

  // Analytics endpoints
  async getDashboardData() {
    return this.request('/analytics/dashboard');
  }

  async getPerformanceAnalytics() {
    return this.request('/analytics/performance');
  }

  async getRiskAnalytics() {
    return this.request('/analytics/risk');
  }

  async getClientAnalytics() {
    return this.request('/analytics/clients');
  }

  // Reports endpoints
  async getReports() {
    return this.request('/reports');
  }

  async generateReport(reportType, parameters) {
    return this.request('/reports/generate', {
      method: 'POST',
      body: JSON.stringify({ type: reportType, parameters }),
    });
  }

  async getReportStatus(reportId) {
    return this.request(`/reports/${reportId}/status`);
  }

  async downloadReport(reportId) {
    const url = `${this.baseURL}/reports/${reportId}/download`;
    window.open(url, '_blank');
  }
}

// Create and export a singleton instance
const apiService = new ApiService();
export default apiService;

// Export individual methods for convenience
export const {
  healthCheck,
  getClients,
  getClient,
  createClient,
  updateClient,
  getPortfolios,
  getPortfolio,
  createPortfolio,
  optimizePortfolio,
  getPortfolioPerformance,
  getPortfolioRisk,
  getHoldings,
  addHolding,
  updateHolding,
  deleteHolding,
  getEvents,
  createEvent,
  processEvent,
  getMarketData,
  getNewsData,
  getWeatherData,
  getAgentHealth,
  startWorkflow,
  sendAgentMessage,
  rebalancePortfolio,
  generateClientCommunication,
  getComplianceStatus,
  getKYCStatus,
  getAMLAlerts,
  getPolicyViolations,
  getAuditTrail,
  getDashboardData,
  getPerformanceAnalytics,
  getRiskAnalytics,
  getClientAnalytics,
  getReports,
  generateReport,
  getReportStatus,
  downloadReport,
} = apiService;

