// src/components/new-interview/components/DetectionStatsPanel.tsx
import React from "react";

interface DetectionStatsPanelProps {
  violationsCount: number;
  poseHistoryLength: number;
  faceCount: number;
  detectionsCount: number;
}

export function DetectionStatsPanel({
  violationsCount,
  poseHistoryLength,
  faceCount,
  detectionsCount,
}: DetectionStatsPanelProps) {
  return (
    <div className="bg-gray-800/50 rounded-lg p-6 backdrop-blur-sm border border-gray-700">
      <h2 className="text-xl font-semibold mb-4">Detection Summary</h2>
      <div className="space-y-4">
        <div className="flex items-center justify-between p-4 bg-gray-700/50 rounded-lg">
          <div>
            <p className="text-sm text-gray-400">Total Violations</p>
            <p className="text-2xl font-bold">{violationsCount}</p>
          </div>
          <div className="text-3xl">🚨</div>
        </div>

        <div className="flex items-center justify-between p-4 bg-gray-700/50 rounded-lg">
          <div>
            <p className="text-sm text-gray-400">Poses Recorded</p>
            <p className="text-2xl font-bold">{poseHistoryLength}</p>
          </div>
          <div className="text-3xl">📊</div>
        </div>

        <div className="flex items-center justify-between p-4 bg-gray-700/50 rounded-lg">
          <div>
            <p className="text-sm text-gray-400">Current Face Count</p>
            <p className="text-2xl font-bold">{faceCount}</p>
          </div>
          <div className="text-3xl">👤</div>
        </div>

        <div className="flex items-center justify-between p-4 bg-gray-700/50 rounded-lg">
          <div>
            <p className="text-sm text-gray-400">Suspicious Objects</p>
            <p className="text-2xl font-bold">{detectionsCount}</p>
          </div>
          <div className="text-3xl">📱</div>
        </div>
      </div>
    </div>
  );
}
