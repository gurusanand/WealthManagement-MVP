import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

const ClientPortalPage = () => {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">Client Portal</h1>
        <p className="text-muted-foreground">
          Self-service client interface with portfolio management.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Client Portal</CardTitle>
          <CardDescription>
            This page is under development. Client self-service tools will be available here.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">
            Coming soon: Portfolio views, performance tracking, document management, and goal setting.
          </p>
        </CardContent>
      </Card>
    </div>
  )
}

export default ClientPortalPage

