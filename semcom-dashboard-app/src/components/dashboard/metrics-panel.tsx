"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Progress } from "@/components/ui/progress"
import { BarChart3, TrendingUp, TrendingDown, Minus } from "lucide-react"
import type { SimulationResult } from "../semantic-comm-dashboard"

interface MetricsPanelProps {
  result: SimulationResult
  isSimulating: boolean
}

export function MetricsPanel({ result, isSimulating }: MetricsPanelProps) {
  const degradation = result.simLocal - result.simRx
  const qualityPreserved = (result.effectiveSim / result.simLocal) * 100

  return (
    <Card className="lg:col-span-2">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-lg">
          <BarChart3 className="w-5 h-5 text-primary" />
          Performance Metrics
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          {/* Semantic Preservation */}
          <MetricCard
            label="Semantic Preservation"
            value={`${qualityPreserved.toFixed(1)}%`}
            progress={qualityPreserved}
            color="primary"
            trend={qualityPreserved >= 98 ? "up" : qualityPreserved >= 95 ? "neutral" : "down"}
            isLoading={isSimulating}
          />

          {/* Channel Degradation */}
          <MetricCard
            label="Channel Degradation"
            value={`${(degradation * 100).toFixed(2)}%`}
            progress={Math.min(degradation * 500, 100)}
            color="destructive"
            trend={degradation < 0.02 ? "up" : degradation < 0.05 ? "neutral" : "down"}
            isLoading={isSimulating}
            invertProgress
          />

          {/* Transmission Decision */}
          <MetricCard
            label="Decision"
            value={result.transmit ? "TRANSMIT" : "SKIP"}
            color={result.transmit ? "chart-5" : "chart-3"}
            isLoading={isSimulating}
            badge={result.transmit ? "High Uncertainty" : "Low Uncertainty"}
          />

          {/* Effective Quality */}
          <MetricCard
            label="Effective Quality"
            value={result.effectiveSim.toFixed(4)}
            progress={result.effectiveSim * 100}
            color="accent"
            trend="up"
            isLoading={isSimulating}
          />
        </div>

        {/* Key Insight */}
        <div className="mt-4 p-3 rounded-lg bg-muted/30 border border-border">
          <p className="text-sm text-foreground">
            <span className="font-semibold text-primary">💡 Insight: </span>
            {result.transmit
              ? `High uncertainty detected (u = ${result.uncertainty.toFixed(3)} > τ). Transmitting through channel with ${(degradation * 100).toFixed(1)}% quality loss.`
              : `Low uncertainty (u = ${result.uncertainty.toFixed(3)} ≤ τ). Skipping channel transmission saves bandwidth while maintaining quality!`}
          </p>
        </div>
      </CardContent>
    </Card>
  )
}

function MetricCard({
  label,
  value,
  progress,
  color,
  trend,
  isLoading,
  badge,
  invertProgress,
}: {
  label: string
  value: string
  progress?: number
  color: string
  trend?: "up" | "down" | "neutral"
  isLoading: boolean
  badge?: string
  invertProgress?: boolean
}) {
  const colorClasses: Record<string, { text: string; bg: string; progress: string }> = {
    primary: { text: "text-primary", bg: "bg-primary/10", progress: "bg-primary" },
    accent: { text: "text-accent", bg: "bg-accent/10", progress: "bg-accent" },
    destructive: { text: "text-destructive", bg: "bg-destructive/10", progress: "bg-destructive" },
    "chart-3": { text: "text-chart-3", bg: "bg-chart-3/10", progress: "bg-chart-3" },
    "chart-5": { text: "text-chart-5", bg: "bg-chart-5/10", progress: "bg-chart-5" },
  }

  const colors = colorClasses[color] || colorClasses.primary

  return (
    <div className={`p-4 rounded-lg ${colors.bg} transition-all duration-300 ${isLoading && "opacity-70"}`}>
      <div className="flex items-center justify-between mb-2">
        <p className="text-xs text-muted-foreground">{label}</p>
        {trend && (
          <span className={colors.text}>
            {trend === "up" && <TrendingUp className="w-4 h-4" />}
            {trend === "down" && <TrendingDown className="w-4 h-4" />}
            {trend === "neutral" && <Minus className="w-4 h-4" />}
          </span>
        )}
      </div>
      <p className={`text-xl font-bold font-mono ${colors.text}`}>{value}</p>
      {progress !== undefined && <Progress value={invertProgress ? 100 - progress : progress} className="h-1.5 mt-2" />}
      {badge && (
        <span className={`inline-block mt-2 text-xs px-2 py-0.5 rounded ${colors.bg} ${colors.text}`}>{badge}</span>
      )}
    </div>
  )
}
