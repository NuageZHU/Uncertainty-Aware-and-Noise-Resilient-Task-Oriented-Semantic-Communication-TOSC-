const API_BASE_URL = "http://127.0.0.1:8000"

export interface Sample {
  name: string
  path: string
}

export interface RunRequest {
  image_name: string
  n_bits: number
  sigma: number
  tau: number
  dropout_p?: number
}

export interface RunResponse {
  transmitted: boolean
  uncertainty: number
  sim_local: number
  sim_rx: number
  effective_sim: number
  original_url: string
  local_recon_url: string
  channel_recon_url: string | null
}

export interface TauScanData {
  tau: number[]
  transmit_rate: number[]
  mean_effective_sim: number[]
  mean_sim_rx: number[]
  mean_sim_local: number[]
}

export interface QuantizationNoiseData {
  n_bits: number
  sigma: number
  sim_rx_mean: number
  degradation_mean: number
}

export interface ComplexityData {
  complexity_group: string
  [key: string]: string | number
}

export interface UploadRunResponse {
  run_id: string
  transmitted: boolean
  uncertainty: number
  sim_local: number
  sim_rx: number
  effective_sim: number
  semantic_degradation: number
  original_url: string
  local_recon_url: string
  channel_recon_url: string
  n_bits: number
  sigma: number
  tau: number
}

export const api = {
  // Get list of available images
  async getSamples(): Promise<Sample[]> {
    const res = await fetch(`${API_BASE_URL}/api/samples`)
    if (!res.ok) throw new Error("Failed to fetch samples")
    return res.json()
  },

  // Run the semantic communication pipeline
  async run(params: RunRequest): Promise<RunResponse> {
    const res = await fetch(`${API_BASE_URL}/api/run`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(params),
    })
    if (!res.ok) throw new Error("Failed to run simulation")
    return res.json()
  },

  // Get tau scan experiment data
  async getTauScan(): Promise<TauScanData> {
    const res = await fetch(`${API_BASE_URL}/api/experiments/tau-scan`)
    if (!res.ok) throw new Error("Failed to fetch tau scan data")
    return res.json()
  },

  // Get quantization noise experiment data
  async getQuantizationNoise(): Promise<QuantizationNoiseData[]> {
    const res = await fetch(`${API_BASE_URL}/api/experiments/quantization-noise`)
    if (!res.ok) throw new Error("Failed to fetch quantization noise data")
    return res.json()
  },

  // Get complexity experiment data
  async getComplexity(): Promise<ComplexityData[]> {
    const res = await fetch(`${API_BASE_URL}/api/experiments/complexity`)
    if (!res.ok) throw new Error("Failed to fetch complexity data")
    return res.json()
  },

  async uploadAndRun(
    image: File,
    params: { n_bits: number; sigma: number; tau: number; dropout_p?: number },
  ): Promise<UploadRunResponse> {
    const formData = new FormData()
    formData.append("image", image)
    formData.append("n_bits", params.n_bits.toString())
    formData.append("sigma", params.sigma.toString())
    formData.append("tau", params.tau.toString())
    if (params.dropout_p !== undefined) {
      formData.append("dropout_p", params.dropout_p.toString())
    }

    const res = await fetch(`${API_BASE_URL}/api/upload-run`, {
      method: "POST",
      body: formData,
    })
    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: "Failed to process image" }))
      throw new Error(error.detail || "Failed to process image")
    }
    return res.json()
  },

  // Helper to get full URL for static files
  getStaticUrl(path: string): string {
    return `${API_BASE_URL}${path}`
  },
}
