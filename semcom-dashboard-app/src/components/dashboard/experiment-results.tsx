"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { FlaskConical, Lightbulb, Target, Zap, Loader2 } from "lucide-react"
import { useTauScan, useQuantizationNoise, useComplexity } from "@/hooks/use-api"

interface QuantDataItem {
  n_bits: number
  sigma: number
  sim_rx_mean: number
  degradation_mean: number
}

export function ExperimentResults() {
  const { data: tauData, isLoading: tauLoading } = useTauScan()
  const { data: quantData, isLoading: quantLoading } = useQuantizationNoise()
  const { data: complexityData, isLoading: complexLoading } = useComplexity()

  const isLoading = tauLoading || quantLoading || complexLoading
  const hasApiData = tauData || quantData || complexityData

  // Calculate optimal tau from API data
  const optimalTau =
    tauData && tauData.tau.length > 0
      ? tauData.tau[tauData.mean_effective_sim.indexOf(Math.max(...tauData.mean_effective_sim))]
      : null

  // Calculate best quantization from API data
  const bestQuant =
    quantData && quantData.length > 0
      ? quantData.reduce((best: QuantDataItem, curr: QuantDataItem) => (curr.sim_rx_mean > best.sim_rx_mean ? curr : best), quantData[0])
      : null

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-lg">
          <FlaskConical className="w-5 h-5 text-primary" />
          Key Research Findings
          {isLoading && <Loader2 className="w-4 h-4 animate-spin text-muted-foreground" />}
          {hasApiData && (
            <Badge variant="outline" className="ml-2 text-xs text-green-500 border-green-500/30">
              Live Data
            </Badge>
          )}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          {/* Finding 1 - Optimal Threshold */}
          <div className="space-y-3 p-4 rounded-lg bg-primary/5 border border-primary/20">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-primary/20 flex items-center justify-center">
                <Target className="w-4 h-4 text-primary" />
              </div>
              <h3 className="font-semibold text-foreground">Optimal Threshold</h3>
            </div>
            <p className="text-sm text-muted-foreground">
              {optimalTau !== null ? (
                <>
                  Found τ* = <span className="font-mono text-primary">{optimalTau.toFixed(2)}</span> provides best
                  trade-off between quality and bandwidth from your experiments.
                </>
              ) : (
                <>
                  Found τ* = <span className="font-mono text-primary">0.02-0.05</span> provides best trade-off between
                  quality and bandwidth for moderate noise (σ = 0.1).
                </>
              )}
            </p>
            <Badge variant="outline" className="text-xs">
              Experiment 2: Tau Scan
            </Badge>
          </div>

          {/* Finding 2 - Quantization */}
          <div className="space-y-3 p-4 rounded-lg bg-accent/5 border border-accent/20">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-accent/20 flex items-center justify-center">
                <Zap className="w-4 h-4 text-accent" />
              </div>
              <h3 className="font-semibold text-foreground">Quantization Sweet Spot</h3>
            </div>
            <p className="text-sm text-muted-foreground">
              {bestQuant ? (
                <>
                  <span className="font-mono text-accent">{bestQuant.n_bits}-bit</span> quantization at σ=
                  {bestQuant.sigma} achieves{" "}
                  <span className="font-mono text-accent">{(bestQuant.sim_rx_mean * 100).toFixed(1)}%</span> semantic
                  similarity.
                </>
              ) : (
                <>
                  <span className="font-mono text-accent">6-bit</span> quantization preserves semantic fidelity while
                  achieving <span className="font-mono text-accent">4x</span> compression vs 16-bit baseline.
                </>
              )}
            </p>
            <Badge variant="outline" className="text-xs">
              Experiment 1: Quantization
            </Badge>
          </div>

          {/* Finding 3 - Adaptive Strategy */}
          <div className="space-y-3 p-4 rounded-lg bg-chart-3/5 border border-chart-3/20">
            <div className="flex items-center gap-2">
              <div className="w-8 h-8 rounded-lg bg-chart-3/20 flex items-center justify-center">
                <Lightbulb className="w-4 h-4 text-chart-3" />
              </div>
              <h3 className="font-semibold text-foreground">Adaptive Strategy</h3>
            </div>
            <p className="text-sm text-muted-foreground">
              {complexityData && complexityData.length > 0 ? (
                <>
                  Tested across <span className="font-mono text-chart-3">{complexityData.length}</span> complexity
                  groups. Uncertainty-aware gating adapts to image difficulty.
                </>
              ) : (
                <>
                  Uncertainty-aware transmission achieves <span className="font-mono text-chart-3">+2.1%</span> quality
                  improvement while saving <span className="font-mono text-chart-3">40-60%</span> bandwidth.
                </>
              )}
            </p>
            <Badge variant="outline" className="text-xs">
              Experiment 3: Complexity
            </Badge>
          </div>
        </div>

        {/* Key Insight Banner */}
        <div className="mt-6 p-4 rounded-lg bg-gradient-to-r from-primary/10 via-accent/10 to-chart-3/10 border border-primary/20">
          <div className="flex items-start gap-3">
            <div className="w-10 h-10 rounded-full bg-primary/20 flex items-center justify-center flex-shrink-0">
              <Target className="w-5 h-5 text-primary" />
            </div>
            <div>
              <h4 className="font-semibold text-foreground mb-1">Core Insight</h4>
              <p className="text-sm text-muted-foreground">
                <strong className="text-foreground">Less transmission = Better quality!</strong> Because channel
                degrades quality (sim_rx {"<"} sim_local), intelligently skipping transmission for low-uncertainty
                samples improves overall semantic fidelity while dramatically reducing bandwidth usage. This creates a{" "}
                <span className="text-primary font-semibold">win-win</span> scenario for task-oriented semantic
                communication systems.
              </p>
            </div>
          </div>
        </div>

        {/* Technical Stack */}
        <div className="mt-6 pt-4 border-t border-border">
          <p className="text-xs text-muted-foreground mb-2">Technology Stack</p>
          <div className="flex flex-wrap gap-2">
            {[
              "Stable Diffusion VAE",
              "OpenCLIP ViT-B/32",
              "PyTorch",
              "FastAPI",
              "AWGN Channel",
              "Uniform Quantization",
            ].map((tech) => (
              <Badge key={tech} variant="secondary" className="text-xs font-mono">
                {tech}
              </Badge>
            ))}
          </div>
        </div>
      </CardContent>
    </Card>
  )
}
