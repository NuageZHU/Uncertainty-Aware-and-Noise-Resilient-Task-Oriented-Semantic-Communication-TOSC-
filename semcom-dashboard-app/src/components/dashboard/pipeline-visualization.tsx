"use client"

import type React from "react"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ArrowRight, Zap, Radio, ImageIcon, Brain, CheckCircle2, XCircle } from "lucide-react"
import type { SimulationParams, SimulationResult } from "../semantic-comm-dashboard"

interface PipelineVisualizationProps {
  params: SimulationParams
  result: SimulationResult
  isSimulating: boolean
}

export function PipelineVisualization({ params, result, isSimulating }: PipelineVisualizationProps) {
  const hasRealImages = result.originalUrl && result.localReconUrl

  return (
    <Card className="overflow-hidden">
      <CardHeader className="pb-2">
        <CardTitle className="flex items-center gap-2 text-lg">
          <Brain className="w-5 h-5 text-primary" />
          Semantic Communication Pipeline
        </CardTitle>
      </CardHeader>
      <CardContent>
        {hasRealImages && (
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mb-6 p-4 rounded-lg bg-muted/30 border border-border">
            <div className="space-y-2">
              <p className="text-xs font-medium text-muted-foreground text-center">Original</p>
              <div className="aspect-square rounded-lg overflow-hidden border border-border">
                <img
                  src={result.originalUrl || "/placeholder.svg"}
                  alt="Original"
                  className="w-full h-full object-cover"
                />
              </div>
            </div>
            <div className="space-y-2">
              <p className="text-xs font-medium text-muted-foreground text-center">
                Local Reconstruction (sim={result.simLocal.toFixed(3)})
              </p>
              <div className="aspect-square rounded-lg overflow-hidden border border-primary/30">
                <img
                  src={result.localReconUrl || "/placeholder.svg"}
                  alt="Local Reconstruction"
                  className="w-full h-full object-cover"
                />
              </div>
            </div>
            <div className="space-y-2">
              <p className="text-xs font-medium text-muted-foreground text-center">
                Channel Output (sim={result.simRx.toFixed(3)})
              </p>
              <div
                className={`aspect-square rounded-lg overflow-hidden border ${result.transmit ? "border-destructive/30" : "border-border opacity-50"}`}
              >
                <img
                  src={result.channelReconUrl || "/placeholder.svg"}
                  alt="Channel Reconstruction"
                  className="w-full h-full object-cover"
                />
                {!result.transmit && (
                  <div className="absolute inset-0 flex items-center justify-center bg-background/80">
                    <span className="text-xs text-muted-foreground">Skipped</span>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        <div className="relative">
          {/* Pipeline Flow */}
          <div className="flex flex-col md:flex-row items-center justify-between gap-4 py-8">
            {/* Original Image */}
            <PipelineNode
              icon={<ImageIcon className="w-6 h-6" />}
              label="Input Image"
              sublabel="Original"
              color="primary"
              active={true}
            />

            {/* Arrow with animation */}
            <PipelineArrow isSimulating={isSimulating} />

            {/* VAE Encoder */}
            <PipelineNode
              icon={<Zap className="w-6 h-6" />}
              label="VAE Encoder"
              sublabel="Latent z"
              color="accent"
              active={true}
              metrics={[`u = ${result.uncertainty.toFixed(3)}`]}
            />

            {/* Transmission Gate */}
            <TransmissionGate transmit={result.transmit} tau={params.tau} />

            {/* Channel (conditional) */}
            <PipelineNode
              icon={<Radio className="w-6 h-6" />}
              label="Channel"
              sublabel={`σ=${params.sigma}, ${params.nBits}-bit`}
              color={result.transmit ? "destructive" : "muted"}
              active={result.transmit}
              hasNoise={result.transmit}
            />

            {/* Arrow */}
            <PipelineArrow isSimulating={isSimulating && result.transmit} muted={!result.transmit} />

            {/* VAE Decoder */}
            <PipelineNode
              icon={<Zap className="w-6 h-6" />}
              label="VAE Decoder"
              sublabel="Reconstruction"
              color="accent"
              active={true}
            />

            {/* Arrow */}
            <PipelineArrow isSimulating={isSimulating} />

            {/* Output */}
            <PipelineNode
              icon={<ImageIcon className="w-6 h-6" />}
              label="Output"
              sublabel={`sim = ${result.effectiveSim.toFixed(3)}`}
              color="chart-3"
              active={true}
            />
          </div>

          {/* Bottom Stats */}
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4 pt-4 border-t border-border">
            <StatCard label="CLIP Similarity (Local)" value={result.simLocal.toFixed(4)} color="text-primary" />
            <StatCard label="CLIP Similarity (Rx)" value={result.simRx.toFixed(4)} color="text-accent" />
            <StatCard
              label="Effective Similarity"
              value={result.effectiveSim.toFixed(4)}
              color="text-chart-3"
              highlight
            />
            <StatCard
              label="Bandwidth Saved"
              value={`${result.bandwidthSaved.toFixed(0)}%`}
              color={result.bandwidthSaved > 0 ? "text-green-400" : "text-muted-foreground"}
            />
          </div>
        </div>
      </CardContent>
    </Card>
  )
}

function PipelineNode({
  icon,
  label,
  sublabel,
  color,
  active,
  metrics,
  hasNoise,
}: {
  icon: React.ReactNode
  label: string
  sublabel: string
  color: string
  active: boolean
  metrics?: string[]
  hasNoise?: boolean
}) {
  const colorClasses: Record<string, string> = {
    primary: "bg-primary/20 text-primary border-primary/40",
    accent: "bg-accent/20 text-accent border-accent/40",
    destructive: "bg-destructive/20 text-destructive border-destructive/40",
    muted: "bg-muted/50 text-muted-foreground border-border",
    "chart-3": "bg-chart-3/20 text-chart-3 border-chart-3/40",
  }

  return (
    <div className={`flex flex-col items-center gap-2 transition-all duration-300 ${!active && "opacity-40"}`}>
      <div
        className={`
        relative w-16 h-16 rounded-xl border-2 flex items-center justify-center
        ${colorClasses[color] || colorClasses.muted}
        ${active && "shadow-lg"}
        ${hasNoise && "animate-noise"}
      `}
      >
        {icon}
        {active && <div className="absolute inset-0 rounded-xl bg-current opacity-10 animate-pulse" />}
      </div>
      <div className="text-center">
        <p className="text-sm font-medium text-foreground">{label}</p>
        <p className="text-xs text-muted-foreground">{sublabel}</p>
        {metrics?.map((m, i) => (
          <p key={i} className="text-xs font-mono text-primary mt-1">
            {m}
          </p>
        ))}
      </div>
    </div>
  )
}

function PipelineArrow({ isSimulating, muted }: { isSimulating: boolean; muted?: boolean }) {
  return (
    <div className={`relative flex-shrink-0 ${muted && "opacity-30"}`}>
      <ArrowRight className="w-6 h-6 text-muted-foreground" />
      {isSimulating && !muted && (
        <div className="absolute inset-0 overflow-hidden">
          <div className="w-2 h-2 bg-primary rounded-full animate-data-flow" />
        </div>
      )}
    </div>
  )
}

function TransmissionGate({
  transmit,
  tau,
}: {
  transmit: boolean
  tau: number
}) {
  return (
    <div className="flex flex-col items-center gap-2">
      <div
        className={`
        w-16 h-16 rounded-xl border-2 flex items-center justify-center
        transition-all duration-300
        ${
          transmit
            ? "bg-chart-5/20 text-chart-5 border-chart-5/40"
            : "bg-green-500/20 text-green-400 border-green-500/40"
        }
      `}
      >
        {transmit ? <XCircle className="w-6 h-6" /> : <CheckCircle2 className="w-6 h-6" />}
      </div>
      <div className="text-center">
        <p className="text-sm font-medium text-foreground">Gate</p>
        <p className="text-xs text-muted-foreground">τ = {tau}</p>
        <p className={`text-xs font-medium ${transmit ? "text-chart-5" : "text-green-400"}`}>
          {transmit ? "TRANSMIT" : "SKIP"}
        </p>
      </div>
    </div>
  )
}

function StatCard({
  label,
  value,
  color,
  highlight,
}: {
  label: string
  value: string
  color: string
  highlight?: boolean
}) {
  return (
    <div
      className={`
      text-center p-3 rounded-lg
      ${highlight ? "bg-card border border-primary/30" : "bg-muted/30"}
    `}
    >
      <p className="text-xs text-muted-foreground mb-1">{label}</p>
      <p className={`text-lg font-mono font-bold ${color}`}>{value}</p>
    </div>
  )
}
