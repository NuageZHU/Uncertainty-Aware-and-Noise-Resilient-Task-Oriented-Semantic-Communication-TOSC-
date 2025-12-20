"use client"

import type React from "react"

import { useState, useCallback } from "react"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { Upload, Play, Loader2, AlertCircle, X, ImageIcon } from "lucide-react"

interface ImageUploaderProps {
  onRunPipeline: (file: File) => Promise<void>
  isProcessing: boolean
  error: string | null
}

export function ImageUploader({ onRunPipeline, isProcessing, error }: ImageUploaderProps) {
  const [selectedFile, setSelectedFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [isDragging, setIsDragging] = useState(false)

  const handleFileSelect = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) {
      return
    }
    setSelectedFile(file)
    // Create preview URL
    const url = URL.createObjectURL(file)
    setPreviewUrl(url)
  }, [])

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault()
      setIsDragging(false)
      const file = e.dataTransfer.files[0]
      if (file) handleFileSelect(file)
    },
    [handleFileSelect],
  )

  const handleDragOver = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }, [])

  const handleDragLeave = useCallback((e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
  }, [])

  const handleInputChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const file = e.target.files?.[0]
      if (file) handleFileSelect(file)
    },
    [handleFileSelect],
  )

  const clearSelection = useCallback(() => {
    setSelectedFile(null)
    if (previewUrl) {
      URL.revokeObjectURL(previewUrl)
      setPreviewUrl(null)
    }
  }, [previewUrl])

  const handleRun = async () => {
    if (!selectedFile) return
    await onRunPipeline(selectedFile)
  }

  return (
    <Card>
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center gap-2 text-lg">
          <Upload className="w-5 h-5 text-primary" />
          Upload Image
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Drop Zone / Preview */}
        <div
          onDrop={handleDrop}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          className={`
            relative border-2 border-dashed rounded-lg transition-all
            ${isDragging ? "border-primary bg-primary/5" : "border-border"}
            ${selectedFile ? "p-2" : "p-8"}
          `}
        >
          {selectedFile && previewUrl ? (
            <div className="flex items-center gap-4">
              <div className="relative w-32 h-32 rounded-lg overflow-hidden border border-border flex-shrink-0">
                <img src={previewUrl || "/placeholder.svg"} alt="Preview" className="w-full h-full object-cover" />
                <button
                  onClick={clearSelection}
                  className="absolute top-1 right-1 p-1 rounded-full bg-background/80 hover:bg-background border border-border"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
              <div className="flex-1 min-w-0">
                <p className="font-medium truncate">{selectedFile.name}</p>
                <p className="text-sm text-muted-foreground">{(selectedFile.size / 1024).toFixed(1)} KB</p>
              </div>
            </div>
          ) : (
            <label className="flex flex-col items-center justify-center cursor-pointer">
              <div className="w-16 h-16 rounded-full bg-primary/10 flex items-center justify-center mb-4">
                <ImageIcon className="w-8 h-8 text-primary" />
              </div>
              <p className="text-foreground font-medium mb-1">Drop your image here or click to browse</p>
              <p className="text-sm text-muted-foreground">Supports PNG, JPG, JPEG, BMP</p>
              <input type="file" accept="image/*" onChange={handleInputChange} className="hidden" />
            </label>
          )}
        </div>

        {/* Run Button */}
        <Button onClick={handleRun} disabled={!selectedFile || isProcessing} className="w-full" size="lg">
          {isProcessing ? (
            <>
              <Loader2 className="w-4 h-4 mr-2 animate-spin" />
              Processing Pipeline...
            </>
          ) : (
            <>
              <Play className="w-4 h-4 mr-2" />
              Run Semantic Pipeline
            </>
          )}
        </Button>

        {/* Error Display */}
        {error && (
          <Alert variant="destructive">
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>{error}</AlertDescription>
          </Alert>
        )}
      </CardContent>
    </Card>
  )
}
