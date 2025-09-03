import { useState } from 'react'
import { useNavigate, useLocation } from 'react-router-dom'
import { useAuth } from '@/contexts/AuthContext'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Alert, AlertDescription } from '@/components/ui/alert'
import { Loader2, Eye, EyeOff, AlertCircle } from 'lucide-react'
import { useToast } from '@/hooks/use-toast'

const LoginPage = () => {
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [showPassword, setShowPassword] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const { login } = useAuth()
  const navigate = useNavigate()
  const location = useLocation()
  const { toast } = useToast()

  const from = location.state?.from?.pathname || '/dashboard'

  // Demo accounts for easy testing
  const demoAccounts = [
    {
      email: 'rm@wealth.com',
      role: 'Relationship Manager',
      description: 'Full client and portfolio management access'
    },
    {
      email: 'client@wealth.com',
      role: 'Client',
      description: 'Self-service portfolio access'
    },
    {
      email: 'ops@wealth.com',
      role: 'Operations',
      description: 'Back-office operations and compliance'
    },
    {
      email: 'admin@wealth.com',
      role: 'Administrator',
      description: 'Full system administration access'
    }
  ]

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setLoading(true)

    try {
      const result = await login(email, password)
      
      if (result.success) {
        toast({
          title: "Login successful",
          description: `Welcome back, ${result.user.name}!`,
        })
        navigate(from, { replace: true })
      } else {
        setError(result.error || 'Login failed')
      }
    } catch (err) {
      setError('An unexpected error occurred')
    } finally {
      setLoading(false)
    }
  }

  const handleDemoLogin = async (demoEmail) => {
    setEmail(demoEmail)
    setPassword('password')
    setError('')
    setLoading(true)

    try {
      const result = await login(demoEmail, 'password')
      
      if (result.success) {
        toast({
          title: "Demo login successful",
          description: `Logged in as ${result.user.name}`,
        })
        navigate(from, { replace: true })
      } else {
        setError(result.error || 'Demo login failed')
      }
    } catch (err) {
      setError('An unexpected error occurred')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Login Form */}
      <Card>
        <CardHeader className="space-y-1">
          <CardTitle className="text-2xl text-center">Sign in</CardTitle>
          <CardDescription className="text-center">
            Enter your credentials to access your account
          </CardDescription>
        </CardHeader>
        <CardContent>
          <form onSubmit={handleSubmit} className="space-y-4">
            {error && (
              <Alert variant="destructive">
                <AlertCircle className="h-4 w-4" />
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            <div className="space-y-2">
              <Label htmlFor="email">Email</Label>
              <Input
                id="email"
                type="email"
                placeholder="Enter your email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                disabled={loading}
              />
            </div>

            <div className="space-y-2">
              <Label htmlFor="password">Password</Label>
              <div className="relative">
                <Input
                  id="password"
                  type={showPassword ? 'text' : 'password'}
                  placeholder="Enter your password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  required
                  disabled={loading}
                />
                <Button
                  type="button"
                  variant="ghost"
                  size="sm"
                  className="absolute right-0 top-0 h-full px-3 py-2 hover:bg-transparent"
                  onClick={() => setShowPassword(!showPassword)}
                  disabled={loading}
                >
                  {showPassword ? (
                    <EyeOff className="h-4 w-4" />
                  ) : (
                    <Eye className="h-4 w-4" />
                  )}
                </Button>
              </div>
            </div>

            <Button type="submit" className="w-full" disabled={loading}>
              {loading && <Loader2 className="mr-2 h-4 w-4 animate-spin" />}
              Sign in
            </Button>
          </form>
        </CardContent>
      </Card>

      {/* Demo Accounts */}
      <Card>
        <CardHeader>
          <CardTitle className="text-lg">Demo Accounts</CardTitle>
          <CardDescription>
            Try different user roles with these demo accounts (password: "password")
          </CardDescription>
        </CardHeader>
        <CardContent>
          <div className="grid gap-3">
            {demoAccounts.map((account) => (
              <div
                key={account.email}
                className="flex items-center justify-between p-3 border rounded-lg hover:bg-accent/50 transition-colors"
              >
                <div className="space-y-1">
                  <div className="flex items-center space-x-2">
                    <span className="font-medium">{account.role}</span>
                    <span className="text-sm text-muted-foreground">
                      ({account.email})
                    </span>
                  </div>
                  <p className="text-sm text-muted-foreground">
                    {account.description}
                  </p>
                </div>
                <Button
                  variant="outline"
                  size="sm"
                  onClick={() => handleDemoLogin(account.email)}
                  disabled={loading}
                >
                  {loading && email === account.email && (
                    <Loader2 className="mr-2 h-4 w-4 animate-spin" />
                  )}
                  Try
                </Button>
              </div>
            ))}
          </div>
        </CardContent>
      </Card>

      {/* Help Text */}
      <div className="text-center text-sm text-muted-foreground">
        <p>
          This is a demo application showcasing an AI-powered wealth management system.
        </p>
        <p className="mt-1">
          Use any of the demo accounts above to explore different user roles and features.
        </p>
      </div>
    </div>
  )
}

export default LoginPage

