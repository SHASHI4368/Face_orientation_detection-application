// src/components/new-interview/components/AlertsPanel.tsx
import React from "react";
import { ViolationAlert } from "../types";

interface AlertsPanelProps {
  violations: ViolationAlert[];
}

export function AlertsPanel({ violations }: AlertsPanelProps) {
  return (
    <div className="bg-gray-800/50 rounded-lg p-6 backdrop-blur-sm border border-gray-700">
      <h2 className="text-xl font-semibold mb-4 flex items-center gap-2">
        <span
          className={violations.length > 0 ? "text-red-400" : "text-green-400"}
        >
          {violations.length > 0 ? "⚠️" : "✓"}
        </span>
        Security Alerts
      </h2>

      <div className="space-y-2 max-h-96 overflow-y-auto">
        {violations.length === 0 ? (
          <div className="text-center py-8 text-gray-400">
            <div className="text-4xl mb-2">✓</div>
            <p>No violations detected</p>
            <p className="text-sm mt-1">Monitoring active</p>
          </div>
        ) : (
          violations
            .slice()
            .reverse()
            .map((violation, idx) => (
              <div
                key={idx}
                className={`p-3 rounded-lg border ${
                  violation.type === "phone"
                    ? "bg-red-900/30 border-red-500"
                    : violation.type === "multiple_faces"
                    ? "bg-orange-900/30 border-orange-500"
                    : "bg-yellow-900/30 border-yellow-500"
                }`}
              >
                <div className="flex items-start gap-2">
                  <span className="text-xl">
                    {violation.type === "phone"
                      ? "📱"
                      : violation.type === "multiple_faces"
                      ? "👥"
                      : "⚠️"}
                  </span>
                  <div className="flex-1">
                    <p className="font-semibold text-sm">{violation.message}</p>
                    <p className="text-xs text-gray-400 mt-1">
                      {new Date(violation.timestamp).toLocaleTimeString()}
                    </p>
                  </div>
                </div>
              </div>
            ))
        )}
      </div>
    </div>
  );
}
