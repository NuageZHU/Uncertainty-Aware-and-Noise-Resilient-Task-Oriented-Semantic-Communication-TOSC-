"use client"

import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area"
import { Alert, AlertDescription } from "@/components/ui/alert"
import { ImageIcon, Play, Loader2, AlertCircle, Wifi, WifiOff } from "lucide-react"
import { useSamples } from "@/hooks/use-api"
import { api } from "@/lib/api"

interface ImageSelectorProps {
  selectedImage: string | null
  onSelectImage: (name: string) => void
  onRunSimulation: () => void
  isSimulating: boolean
  error: string | null
}

export function ImageSelector({
  selectedImage,
  onSelectImage,
  onRunSimulation,
  isSimulating,
  error,
}: ImageSelectorProps) {
  const { data: samples, error: samplesError, isLoading } = useSamples()

  const isApiAvailable = !samplesError && samples && samples.length > 0

  return (
    <Card>
      <CardHeader className="pb-3">
        <div className="flex items-center justify-between">
          <CardTitle className="flex items-center gap-2 text-lg">
            <ImageIcon className="w-5 h-5 text-primary" />
            Image Selection
          </CardTitle>
          <div className="flex items-center gap-2">
            {isApiAvailable ? (
              <span className="flex items-center gap-1.5 text-xs text-green-500">
                <Wifi className="w-3.5 h-3.5" />
                API Connected
              </span>
            ) : (
              <span className="flex items-center gap-1.5 text-xs text-muted-foreground">
                <WifiOff className="w-3.5 h-3.5" />
                Demo Mode
              </span>
            )}
          </div>
        </div>
      </CardHeader>
      <CardContent className="space-y-4">
        {/* Connection Status / Instructions */}
        {!isApiAvailable && !isLoading && (
          <Alert>
            <AlertCircle className="h-4 w-4" />
            <AlertDescription>
              FastAPI not detected. Start your backend with{" "}
              <code className="px-1 py-0.5 rounded bg-muted font-mono text-xs">uvicorn app.main:app --reload</code> to
              enable live processing. Currently showing simulated results.
            </AlertDescription>
          </Alert>
        )}

        {/* Image Grid */}
        {isLoading ? (
          <div className="flex items-center justify-center h-32">
            <Loader2 className="w-6 h-6 animate-spin text-primary" />
            <span className="ml-2 text-sm text-muted-foreground">Loading images...</span>
          </div>
        ) : isApiAvailable && samples ? (
          <ScrollArea className="w-full whitespace-nowrap">
            <div className="flex gap-3 pb-4">
              {samples.map((sample) => (
                <button
                  key={sample.name}
                  onClick={() => onSelectImage(sample.name)}
                  className={`
                    relative flex-shrink-0 w-24 h-24 rounded-lg overflow-hidden border-2 transition-all
                    ${
                      selectedImage === sample.name
                        ? "border-primary ring-2 ring-primary/30"
                        : "border-border hover:border-primary/50"
                    }
                  `}
                >
                  <img
                    src={api.getStaticUrl(sample.path) || "/placeholder.svg"}
                    alt={sample.name}
                    className="w-full h-full object-cover"
                    onError={(e) => {
                      // Fallback placeholder if image fails to load
                      e.currentTarget.src = `/placeholder.svg?height=96&width=96&query=${encodeURIComponent(sample.name)}`
                    }}
                  />
                  {selectedImage === sample.name && (
                    <div className="absolute inset-0 bg-primary/20 flex items-center justify-center">
                      <div className="w-6 h-6 rounded-full bg-primary flex items-center justify-center">
                        <svg
                          className="w-4 h-4 text-primary-foreground"
                          fill="none"
                          viewBox="0 0 24 24"
                          stroke="currentColor"
                        >
                          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                        </svg>
                      </div>
                    </div>
                  )}
                </button>
              ))}
            </div>
            <ScrollBar orientation="horizontal" />
          </ScrollArea>
        ) : (
          <div className="grid grid-cols-4 md:grid-cols-6 lg:grid-cols-8 gap-3">
            {/* Demo placeholder images */}
            {["cat", "dog", "car", "house", "tree", "person", "flower", "bird"].map((name, i) => (
              <button
                key={i}
                onClick={() => onSelectImage(name)}
                className={`
                  relative aspect-square rounded-lg overflow-hidden border-2 transition-all
                  ${
                    selectedImage === name
                      ? "border-primary ring-2 ring-primary/30"
                      : "border-border hover:border-primary/50"
                  }
                `}
              >
                <img
                  src={`/.jpg?height=96&width=96&query=${name}`}
                  alt={name}
                  className="w-full h-full object-cover"
                />
              </button>
            ))}
          </div>
        )}

        {/* Run Button */}
        <div className="flex items-center gap-4">
          <Button
            onClick={onRunSimulation}
            disabled={!selectedImage || isSimulating}
            className="flex-1 md:flex-none"
            size="lg"
          >
            {isSimulating ? (
              <>
                <Loader2 className="w-4 h-4 mr-2 animate-spin" />
                Processing...
              </>
            ) : (
              <>
                <Play className="w-4 h-4 mr-2" />
                Run Pipeline
              </>
            )}
          </Button>

          {selectedImage && (
            <span className="text-sm text-muted-foreground">
              Selected: <span className="font-mono text-foreground">{selectedImage}</span>
            </span>
          )}
        </div>

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
