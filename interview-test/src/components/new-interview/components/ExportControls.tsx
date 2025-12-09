// src/components/new-interview/components/ExportControls.tsx
import React from "react";

interface ExportControlsProps {
  onExportPose: () => void;
  onExportCheating: () => void;
  onClearHistory: () => void;
  hasData: boolean;
}

export function ExportControls({
  onExportPose,
  onExportCheating,
  onClearHistory,
  hasData,
}: ExportControlsProps) {
  return (
    <div className="bg-gray-800/50 rounded-lg p-6 backdrop-blur-sm border border-gray-700">
      <div className="flex items-center justify-between flex-wrap gap-4">
        <div>
          <h3 className="text-lg font-semibold mb-1">Data Export</h3>
          <p className="text-sm text-gray-400">
            Export monitoring data for analysis
          </p>
        </div>
        <div className="flex gap-3">
          <button
            onClick={onClearHistory}
            disabled={!hasData}
            className="px-6 py-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg font-semibold transition-colors"
          >
            Clear All Data
          </button>
          <button
            onClick={onExportPose}
            disabled={!hasData}
            className="px-6 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg font-semibold transition-colors"
          >
            📥 Export Poses
          </button>
          <button
            onClick={onExportCheating}
            disabled={!hasData}
            className="px-6 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg font-semibold transition-colors"
          >
            📥 Export Cheating
          </button>
        </div>
      </div>
    </div>
  );
}
