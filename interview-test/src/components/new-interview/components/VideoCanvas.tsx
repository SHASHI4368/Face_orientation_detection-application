// src/components/new-interview/components/VideoCanvas.tsx
import React from "react";

interface VideoCanvasProps {
  videoRef: React.RefObject<HTMLVideoElement | null>;
  canvasRef: React.RefObject<HTMLCanvasElement | null>;
  fps: number;
  faceCount: number;
  detectionsCount: number;
  poseHistoryLength: number;
  cheatingProbability: number;
}

export function VideoCanvas({
  videoRef,
  canvasRef,
  fps,
  faceCount,
  detectionsCount,
  poseHistoryLength,
  cheatingProbability,
}: VideoCanvasProps) {
  return (
    <div className="lg:col-span-2 bg-gray-800/50 rounded-lg p-6 backdrop-blur-sm border border-gray-700">
      <h2 className="text-xl font-semibold mb-4">Live Monitoring Feed</h2>
      <div className="relative bg-black rounded-lg overflow-hidden aspect-video">
        <div className="flex absolute top-4 right-4 z-10 flex-col gap-2">
          <div className="flex justify-center items-center bg-black/70 border border-white px-4 py-2 rounded text-sm font-semibold">
            Cheating: {(cheatingProbability * 100).toFixed(2)}%
          </div>
        </div>

        <video
          ref={videoRef}
          className="absolute inset-0 w-full h-full object-cover"
          autoPlay
          playsInline
          muted
        />
        <canvas ref={canvasRef} className="relative w-full h-full" />

        {/* Status Indicators */}
        <div className="absolute top-2 left-2 flex flex-col gap-2">
          <div className="bg-black/70 px-3 py-1 rounded text-sm">{fps} FPS</div>
          <div
            className={`px-3 py-1 rounded text-sm font-semibold ${
              faceCount === 1
                ? "bg-green-600"
                : faceCount > 1
                ? "bg-red-600"
                : "bg-gray-600"
            }`}
          >
            {faceCount === 0
              ? "No Face"
              : faceCount === 1
              ? "1 Face ✓"
              : `${faceCount} Faces ⚠️`}
          </div>
          {detectionsCount > 0 && (
            <div className="bg-red-600 px-3 py-1 rounded text-sm font-semibold animate-pulse">
              ⚠️ Objects Detected
            </div>
          )}
        </div>

        <div className="absolute bottom-2 left-2 bg-black/70 px-3 py-1 rounded text-xs">
          Recorded: {poseHistoryLength} poses
        </div>
      </div>
    </div>
  );
}
