// src/components/new-interview/components/HeadOrientationPanel.tsx
import React from "react";
import { HeadPose } from "../types";

interface HeadOrientationPanelProps {
  headPose: HeadPose | null;
}

export function HeadOrientationPanel({ headPose }: HeadOrientationPanelProps) {
  return (
    <div className="bg-gray-800/50 rounded-lg p-6 backdrop-blur-sm border border-gray-700">
      <h2 className="text-xl font-semibold mb-4">Head Orientation</h2>

      {headPose ? (
        <div className="space-y-4">
          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-red-400 font-semibold">Roll</span>
              <span className="text-xl font-bold">
                {headPose.roll.toFixed(1)}°
              </span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div
                className="bg-linear-to-r from-red-500 to-red-400 h-2 rounded-full transition-all duration-200"
                style={{
                  width: `${Math.min(
                    (Math.abs(headPose.roll) / 90) * 100,
                    100
                  )}%`,
                }}
              />
            </div>
          </div>

          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-green-400 font-semibold">Pitch</span>
              <span className="text-xl font-bold">
                {headPose.pitch.toFixed(1)}°
              </span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div
                className="bg-linear-to-r from-green-500 to-green-400 h-2 rounded-full transition-all duration-200"
                style={{
                  width: `${Math.min(
                    (Math.abs(headPose.pitch) / 90) * 100,
                    100
                  )}%`,
                }}
              />
            </div>
          </div>

          <div>
            <div className="flex justify-between items-center mb-2">
              <span className="text-blue-400 font-semibold">Yaw</span>
              <span className="text-xl font-bold">
                {headPose.yaw.toFixed(1)}°
              </span>
            </div>
            <div className="w-full bg-gray-700 rounded-full h-2">
              <div
                className="bg-linear-to-r from-blue-500 to-blue-400 h-2 rounded-full transition-all duration-200"
                style={{
                  width: `${Math.min(
                    (Math.abs(headPose.yaw) / 90) * 100,
                    100
                  )}%`,
                }}
              />
            </div>
          </div>
        </div>
      ) : (
        <div className="text-center py-12 text-gray-400">
          <p>No face detected</p>
        </div>
      )}
    </div>
  );
}
