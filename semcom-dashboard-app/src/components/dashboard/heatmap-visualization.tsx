"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Grid3X3 } from "lucide-react"
import type { SimulationParams } from "../semantic-comm-dashboard"

interface HeatmapVisualizationProps {
  params: SimulationParams
}

// Pre-computed heatmap data based on the research experiments
const SIGMA_VALUES = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]
const TAU_VALUES = [0.0, 0.02, 0.05, 0.08, 0.1, 0.15, 0.2]

function computeEffectiveSim(sigma: number, tau: number): number {
  // Simplified model based on the research findings
  const baseLocal = 0.978
  const channelDegradation = sigma * 0.12
  const simRx = baseLocal - channelDegradation

  // Transmission rate decreases with tau
  const transmitRate = Math.max(0, 1 - tau * 10)

  // Effective similarity is weighted average
  return transmitRate * simRx + (1 - transmitRate) * baseLocal
}

export function HeatmapVisualization({ params }: HeatmapVisualizationProps) {
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-lg">
          <Grid3X3 className="w-5 h-5 text-primary" />
          Quality Heatmap (σ × τ)
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="overflow-x-auto">
          <div className="min-w-[400px]">
            {/* Header Row - Sigma values */}
            <div className="flex items-center gap-1 mb-2">
              <div className="w-16 h-8 flex items-center justify-center text-xs text-muted-foreground font-medium">
                σ →
              </div>
              {SIGMA_VALUES.map((sigma) => (
                <div
                  key={sigma}
                  className="flex-1 h-8 flex items-center justify-center text-xs font-mono text-muted-foreground"
                >
                  {sigma.toFixed(2)}
                </div>
              ))}
            </div>

            {/* Heatmap Grid */}
            <div className="space-y-1">
              {TAU_VALUES.map((tau) => (
                <div key={tau} className="flex items-center gap-1">
                  {/* Row label - Tau */}
                  <div className="w-16 h-10 flex items-center justify-center text-xs font-mono text-muted-foreground">
                    τ={tau.toFixed(2)}
                  </div>

                  {/* Cells */}
                  {SIGMA_VALUES.map((sigma) => {
                    const value = computeEffectiveSim(sigma, tau)
                    const isSelected = Math.abs(params.sigma - sigma) < 0.01 && Math.abs(params.tau - tau) < 0.005

                    return <HeatmapCell key={`${sigma}-${tau}`} value={value} isSelected={isSelected} />
                  })}
                </div>
              ))}
            </div>

            {/* Legend */}
            <div className="flex items-center justify-between mt-4 pt-4 border-t border-border">
              <span className="text-xs text-muted-foreground">Low Quality</span>
              <div className="flex gap-1">
                {[0.75, 0.8, 0.85, 0.9, 0.95, 0.98].map((v) => (
                  <div
                    key={v}
                    className="w-8 h-4 rounded text-[10px] flex items-center justify-center font-mono"
                    style={{
                      backgroundColor: getHeatmapColor(v),
                      color: v > 0.9 ? "#000" : "#fff",
                    }}
                  >
                    {v.toFixed(2)}
                  </div>
                ))}
              </div>
              <span className="text-xs text-muted-foreground">High Quality</span>
            </div>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function HeatmapCell({ value, isSelected }: { value: number; isSelected: boolean }) {
  const bgColor = getHeatmapColor(value)
  const textColor = value > 0.9 ? "#000" : "#fff"

  return (
    <div
      className={`
        flex-1 h-10 rounded flex items-center justify-center
        text-xs font-mono font-medium transition-all
        ${isSelected ? "ring-2 ring-primary ring-offset-2 ring-offset-background scale-105" : ""}
      `}
      style={{
        backgroundColor: bgColor,
        color: textColor,
      }}
    >
      {value.toFixed(3)}
    </div>
  )
}

function getHeatmapColor(value: number): string {
  // Color scale from red (low) to green (high)
  const normalized = Math.max(0, Math.min(1, (value - 0.75) / 0.25))

  if (normalized < 0.33) {
    return `hsl(0, 70%, ${40 + normalized * 30}%)`
  } else if (normalized < 0.66) {
    return `hsl(45, 80%, ${50 + (normalized - 0.33) * 30}%)`
  } else {
    return `hsl(120, 60%, ${40 + (normalized - 0.66) * 40}%)`
  }
}
