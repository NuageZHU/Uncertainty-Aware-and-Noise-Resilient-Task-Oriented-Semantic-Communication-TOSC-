"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Slider } from "@/components/ui/slider"
import { Label } from "@/components/ui/label"
import { Settings2 } from "lucide-react"
import type { SimulationParams } from "../semantic-comm-dashboard"

interface ParameterControlsProps {
  params: SimulationParams
  onParamChange: (params: Partial<SimulationParams>) => void
}

export function ParameterControls({ params, onParamChange }: ParameterControlsProps) {
  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-lg">
          <Settings2 className="w-5 h-5 text-primary" />
          Channel Parameters
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-6">
        {/* Tau - Uncertainty Threshold */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <Label className="text-sm font-medium">Uncertainty Threshold (τ)</Label>
            <span className="text-sm font-mono text-primary bg-primary/10 px-2 py-0.5 rounded">
              {params.tau.toFixed(2)}
            </span>
          </div>
          <Slider
            value={[params.tau]}
            onValueChange={([v]: number[]) => onParamChange({ tau: v })}
            min={0}
            max={0.2}
            step={0.01}
            className="w-full"
          />
          <p className="text-xs text-muted-foreground">Higher τ → Less transmission → More bandwidth saved</p>
        </div>

        {/* Sigma - Channel Noise */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <Label className="text-sm font-medium">Channel Noise (σ)</Label>
            <span className="text-sm font-mono text-accent bg-accent/10 px-2 py-0.5 rounded">
              {params.sigma.toFixed(2)}
            </span>
          </div>
          <Slider
            value={[params.sigma]}
            onValueChange={([v]: number[]) => onParamChange({ sigma: v })}
            min={0}
            max={0.3}
            step={0.01}
            className="w-full"
          />
          <p className="text-xs text-muted-foreground">AWGN standard deviation in latent space</p>
        </div>

        {/* n_bits - Quantization */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <Label className="text-sm font-medium">Quantization Bits</Label>
            <span className="text-sm font-mono text-chart-3 bg-chart-3/10 px-2 py-0.5 rounded">{params.nBits}-bit</span>
          </div>
          <Slider
            value={[params.nBits]}
            onValueChange={([v]: number[]) => onParamChange({ nBits: v })}
            min={2}
            max={16}
            step={1}
            className="w-full"
          />
          <p className="text-xs text-muted-foreground">Uniform quantization bit depth (higher = less compression)</p>
        </div>
      </CardContent>
    </Card>
  )
}
