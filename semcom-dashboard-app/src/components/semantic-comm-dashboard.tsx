"use client"

import { useState } from "react"
import { Header } from "./dashboard/header"
import { PipelineVisualization } from "./dashboard/pipeline-visualization"
import { ParameterControls } from "./dashboard/parameter-controls"
import { MetricsPanel } from "./dashboard/metrics-panel"
import { HeatmapVisualization } from "./dashboard/heatmap-visualization"
import { TradeoffChart } from "./dashboard/tradeoff-charts"
import { ExperimentResults } from "./dashboard/experiment-results"
import { ImageUploader } from "./dashboard/image-uploader"
import { api } from "@/lib/api"

export interface SimulationParams {
  tau: number
  sigma: number
  nBits: number
}

export interface SimulationResult {
  simLocal: number
  simRx: number
  uncertainty: number
  transmit: boolean
  effectiveSim: number
  bandwidthSaved: number
  semanticDegradation?: number
  // URLs from API
  originalUrl?: string
  localReconUrl?: string
  channelReconUrl?: string | null
}

function computeSimulation(params: SimulationParams): SimulationResult {
  const baseSimLocal = 0.975 - params.tau * 0.1
  const noiseDegradation = params.sigma * 0.15
  const quantizationDegradation = Math.max(0, (8 - params.nBits) * 0.008)
  const simRx = Math.max(0.7, baseSimLocal - noiseDegradation - quantizationDegradation)
  const uncertainty = 1 - baseSimLocal
  const transmit = uncertainty > params.tau
  const effectiveSim = transmit ? simRx : baseSimLocal
  const bandwidthSaved = transmit ? 0 : 100

  return {
    simLocal: baseSimLocal,
    simRx,
    uncertainty,
    transmit,
    effectiveSim,
    bandwidthSaved,
  }
}

export function SemanticCommDashboard() {
  const [params, setParams] = useState<SimulationParams>({
    tau: 0.02,
    sigma: 0.1,
    nBits: 6,
  })

  const [isSimulating, setIsSimulating] = useState(false)
  const [apiResult, setApiResult] = useState<SimulationResult | null>(null)
  const [apiError, setApiError] = useState<string | null>(null)
  const [isConnected, setIsConnected] = useState(false)

  // Use API result if available, otherwise use local simulation
  const result = apiResult || computeSimulation(params)

  const handleParamChange = (newParams: Partial<SimulationParams>) => {
    setIsSimulating(true)
    setTimeout(() => {
      setParams((prev) => ({ ...prev, ...newParams }))
      setIsSimulating(false)
      // Clear API result when params change to trigger re-upload
      setApiResult(null)
    }, 300)
  }

  const handleRunPipeline = async (file: File) => {
    setIsSimulating(true)
    setApiError(null)

    try {
      const response = await api.uploadAndRun(file, {
        n_bits: params.nBits,
        sigma: params.sigma,
        tau: params.tau,
      })

      setApiResult({
        simLocal: response.sim_local,
        simRx: response.sim_rx,
        uncertainty: response.uncertainty,
        transmit: response.transmitted,
        effectiveSim: response.effective_sim,
        bandwidthSaved: response.transmitted ? 0 : 100,
        semanticDegradation: response.semantic_degradation,
        originalUrl: api.getStaticUrl(response.original_url),
        localReconUrl: api.getStaticUrl(response.local_recon_url),
        channelReconUrl: api.getStaticUrl(response.channel_recon_url),
      })
      setIsConnected(true)
    } catch (err) {
      setApiError(err instanceof Error ? err.message : "Failed to process image")
      setIsConnected(false)
    } finally {
      setIsSimulating(false)
    }
  }

  return (
    <div className="min-h-screen bg-background">
      <Header isConnected={isConnected} />

      <main className="container mx-auto px-4 py-6 space-y-6">
        {/* Parameter Controls - At the top for visibility */}
        <ParameterControls params={params} onParamChange={handleParamChange} />

        <ImageUploader onRunPipeline={handleRunPipeline} isProcessing={isSimulating} error={apiError} />

        {/* Pipeline Visualization - Hero Section */}
        <PipelineVisualization params={params} result={result} isSimulating={isSimulating} />

        {/* Metrics Row */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <MetricsPanel result={result} isSimulating={isSimulating} />
        </div>

        {/* Visualizations Row */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <HeatmapVisualization params={params} />
          <TradeoffChart params={params} />
        </div>

        {/* Experiment Results */}
        <ExperimentResults />
      </main>
    </div>
  )
}
