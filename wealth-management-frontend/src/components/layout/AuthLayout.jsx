import { Card } from '@/components/ui/card'

const AuthLayout = ({ children }) => {
  return (
    <div className="min-h-screen bg-gradient-to-br from-blue-50 via-white to-indigo-50 dark:from-gray-900 dark:via-gray-800 dark:to-gray-900 flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        {/* Logo and Branding */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 bg-primary rounded-xl mb-4">
            <svg
              className="w-8 h-8 text-primary-foreground"
              fill="none"
              stroke="currentColor"
              viewBox="0 0 24 24"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M13 10V3L4 14h7v7l9-11h-7z"
              />
            </svg>
          </div>
          <h1 className="text-2xl font-bold text-foreground">
            Agentic AI Wealth
          </h1>
          <p className="text-muted-foreground mt-1">
            Strategy Orchestrator
          </p>
        </div>

        {/* Auth Content */}
        <Card className="p-6 shadow-lg border-0 bg-card/80 backdrop-blur-sm">
          {children}
        </Card>

        {/* Footer */}
        <div className="text-center mt-8 text-sm text-muted-foreground">
          <p>© 2024 Agentic AI Wealth Management. All rights reserved.</p>
        </div>
      </div>
    </div>
  )
}

export default AuthLayout

