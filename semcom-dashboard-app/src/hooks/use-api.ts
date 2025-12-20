"use client"

import useSWR from "swr"
import { api, type RunResponse, type Sample } from "@/lib/api"

export function useSamples() {
  return useSWR<Sample[]>("samples", api.getSamples, {
    revalidateOnFocus: false,
  })
}

export function useTauScan() {
  return useSWR("tau-scan", api.getTauScan, {
    revalidateOnFocus: false,
  })
}

export function useQuantizationNoise() {
  return useSWR("quantization-noise", api.getQuantizationNoise, {
    revalidateOnFocus: false,
  })
}

export function useComplexity() {
  return useSWR("complexity", api.getComplexity, {
    revalidateOnFocus: false,
  })
}

export type { RunResponse }
