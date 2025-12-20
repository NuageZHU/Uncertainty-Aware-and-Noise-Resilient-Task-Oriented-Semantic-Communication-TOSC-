"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { LineChart, Loader2 } from "lucide-react"
import type { SimulationParams } from "../semantic-comm-dashboard"
import { useTauScan } from "@/hooks/use-api"

interface TradeoffChartProps {
  params: SimulationParams
}

// Fallback data for demo mode
const TAU_RANGE = [0, 0.01, 0.02, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2]

function generateTradeoffData(sigma: number) {
  return TAU_RANGE.map((tau) => {
    const baseLocal = 0.978
    const channelDegradation = sigma * 0.12
    const simRx = baseLocal - channelDegradation
    const transmitRate = Math.max(0, 1 - tau * 8)
    const effectiveSim = transmitRate * simRx + (1 - transmitRate) * baseLocal

    return {
      tau,
      transmitRate: transmitRate * 100,
      effectiveSim,
      bandwidthSaved: (1 - transmitRate) * 100,
    }
  })
}

export function TradeoffChart({ params }: TradeoffChartProps) {
  const { data: apiData, isLoading } = useTauScan()

  // Use API data if available, otherwise fall back to simulated data
  const data = apiData
    ? apiData.tau.map((tau: number, i: number) => ({
        tau,
        transmitRate: apiData.transmit_rate[i] * 100,
        effectiveSim: apiData.mean_effective_sim[i],
        bandwidthSaved: (1 - apiData.transmit_rate[i]) * 100,
      }))
    : generateTradeoffData(params.sigma)

  const tauValues = apiData ? apiData.tau : TAU_RANGE

  // Find current position
  const currentIdx = tauValues.findIndex((t: number) => Math.abs(t - params.tau) < 0.01)

  // Chart dimensions
  const width = 400
  const height = 200
  const padding = { top: 20, right: 20, bottom: 30, left: 40 }
  const chartWidth = width - padding.left - padding.right
  const chartHeight = height - padding.top - padding.bottom

  const maxTau = Math.max(...tauValues)
  const minSim = Math.min(...data.map((d: { effectiveSim: number }) => d.effectiveSim)) - 0.02
  const maxSim = Math.max(...data.map((d: { effectiveSim: number }) => d.effectiveSim)) + 0.01

  // Scale functions
  const xScale = (val: number) => padding.left + (val / maxTau) * chartWidth
  const yScale = (val: number) => padding.top + (1 - (val - minSim) / (maxSim - minSim)) * chartHeight
  const yScaleRate = (val: number) => padding.top + (1 - val / 100) * chartHeight

  // Generate path
  const effectivePath = data
    .map((d: { tau: number; effectiveSim: number }, i: number) => `${i === 0 ? "M" : "L"} ${xScale(d.tau)} ${yScale(d.effectiveSim)}`)
    .join(" ")

  const transmitPath = data
    .map((d: { tau: number; transmitRate: number }, i: number) => `${i === 0 ? "M" : "L"} ${xScale(d.tau)} ${yScaleRate(d.transmitRate)}`)
    .join(" ")

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-lg">
          <LineChart className="w-5 h-5 text-primary" />
          Trade-off: Quality vs Bandwidth
          {isLoading && <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />}
          {apiData && (
            <Badge variant="outline" className="ml-2 text-xs text-green-500 border-green-500/30">
              Live Data
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto">
          {/* Grid lines */}
          {[
            minSim,
            minSim + (maxSim - minSim) * 0.25,
            minSim + (maxSim - minSim) * 0.5,
            minSim + (maxSim - minSim) * 0.75,
            maxSim,
          ].map((v) => (
            <g key={v}>
              <line
                x1={padding.left}
                y1={yScale(v)}
                x2={width - padding.right}
                y2={yScale(v)}
                stroke="currentColor"
                strokeOpacity={0.1}
              />
              <text
                x={padding.left - 5}
                y={yScale(v)}
                textAnchor="end"
                alignmentBaseline="middle"
                className="fill-muted-foreground text-[10px] font-mono"
              >
                {v.toFixed(2)}
              </text>
            </g>
          ))}

          {/* X axis labels */}
          {[0, maxTau * 0.25, maxTau * 0.5, maxTau * 0.75, maxTau].map((v) => (
            <text
              key={v}
              x={xScale(v)}
              y={height - 10}
              textAnchor="middle"
              className="fill-muted-foreground text-[10px] font-mono"
            >
              {v.toFixed(2)}
            </text>
          ))}

          {/* Axis labels */}
          <text x={width / 2} y={height} textAnchor="middle" className="fill-muted-foreground text-[11px]">
            Uncertainty Threshold (τ)
          </text>

          {/* Transmission rate line (dashed) */}
          <path
            d={transmitPath}
            fill="none"
            stroke="hsl(var(--chart-5))"
            strokeWidth={2}
            strokeDasharray="4 2"
            opacity={0.6}
          />

          {/* Effective similarity line */}
          <path d={effectivePath} fill="none" stroke="hsl(var(--primary))" strokeWidth={2.5} />

          {/* Data points */}
          {data.map((d: { tau: number; effectiveSim: number }, i: number) => (
            <g key={i}>
              <circle
                cx={xScale(d.tau)}
                cy={yScale(d.effectiveSim)}
                r={currentIdx === i ? 6 : 3}
                fill={currentIdx === i ? "hsl(var(--primary))" : "hsl(var(--card))"}
                stroke="hsl(var(--primary))"
                strokeWidth={2}
              />
            </g>
          ))}

          {/* Current point highlight */}
          {currentIdx >= 0 && (
            <g>
              <line
                x1={xScale(data[currentIdx].tau)}
                y1={padding.top}
                x2={xScale(data[currentIdx].tau)}
                y2={height - padding.bottom}
                stroke="hsl(var(--primary))"
                strokeDasharray="2 2"
                opacity={0.5}
              />
            </g>
          )}

          {/* Legend */}
          <g transform={`translate(${width - 130}, 15)`}>
            <line x1={0} y1={0} x2={20} y2={0} stroke="hsl(var(--primary))" strokeWidth={2} />
            <text x={25} y={4} className="fill-foreground text-[10px]">
              Effective Sim
            </text>

            <line x1={0} y1={15} x2={20} y2={15} stroke="hsl(var(--chart-5))" strokeWidth={2} strokeDasharray="4 2" />
            <text x={25} y={19} className="fill-foreground text-[10px]">
              Transmit Rate
            </text>
          </g>
        </svg>

        {/* Current values */}
        <div className="grid grid-cols-3 gap-2 mt-4 pt-4 border-t border-border text-center">
          <div>
            <p className="text-xs text-muted-foreground">Current τ</p>
            <p className="text-lg font-mono font-bold text-primary">{params.tau.toFixed(2)}</p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Quality</p>
            <p className="text-lg font-mono font-bold text-chart-3">
              {currentIdx >= 0 ? data[currentIdx].effectiveSim.toFixed(3) : "-"}
            </p>
          </div>
          <div>
            <p className="text-xs text-muted-foreground">Bandwidth Saved</p>
            <p className="text-lg font-mono font-bold text-accent">
              {currentIdx >= 0 ? `${data[currentIdx].bandwidthSaved.toFixed(0)}%` : "-"}
            </p>
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
