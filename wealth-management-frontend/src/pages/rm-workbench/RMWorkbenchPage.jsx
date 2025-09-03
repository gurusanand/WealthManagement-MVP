import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'

const RMWorkbenchPage = () => {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold tracking-tight">RM Workbench</h1>
        <p className="text-muted-foreground">
          Comprehensive relationship manager tools and client management.
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>RM Workbench</CardTitle>
          <CardDescription>
            This page is under development. Advanced RM tools will be available here.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <p className="text-muted-foreground">
            Coming soon: Client management, portfolio oversight, compliance monitoring, and advanced analytics.
          </p>
        </CardContent>
      </Card>
    </div>
  )
}

export default RMWorkbenchPage

