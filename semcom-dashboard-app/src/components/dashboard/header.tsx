import { Cpu, Wifi, WifiOff, Activity } from "lucide-react"

interface HeaderProps {
  isConnected?: boolean
}

export function Header({ isConnected = false }: HeaderProps) {
  return (
    <header className="border-b border-border bg-card/50 backdrop-blur-sm sticky top-0 z-50">
      <div className="container mx-auto px-4 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="relative">
              <div className="w-10 h-10 rounded-lg bg-primary/20 flex items-center justify-center">
                {isConnected ? (
                  <Wifi className="w-5 h-5 text-primary" />
                ) : (
                  <WifiOff className="w-5 h-5 text-muted-foreground" />
                )}
              </div>
              <div
                className={`absolute -top-1 -right-1 w-3 h-3 rounded-full ${isConnected ? "bg-green-500 animate-pulse" : "bg-amber-500"}`}
              />
            </div>
            <div>
              <h1 className="text-xl font-bold text-foreground">SemCom Lab</h1>
              <p className="text-xs text-muted-foreground">Uncertainty-Aware Semantic Communication</p>
            </div>
          </div>

          <div className="flex items-center gap-6">
            <div className="hidden md:flex items-center gap-2 text-sm text-muted-foreground">
              <Cpu className="w-4 h-4 text-primary" />
              <span>VAE + CLIP Pipeline</span>
            </div>
            <div className="hidden md:flex items-center gap-2 text-sm text-muted-foreground">
              <Activity className="w-4 h-4 text-accent" />
              <span>{isConnected ? "Live Processing" : "Demo Mode"}</span>
            </div>
            <div
              className={`px-3 py-1.5 rounded-full text-xs font-medium ${isConnected ? "bg-green-500/10 border border-green-500/30 text-green-500" : "bg-primary/10 border border-primary/30 text-primary"}`}
            >
              {isConnected ? "API Connected" : "Research Demo"}
            </div>
          </div>
        </div>
      </div>
    </header>
  )
}
