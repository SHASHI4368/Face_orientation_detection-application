/* eslint-disable react-hooks/immutability */
/* eslint-disable react-hooks/purity */
// src/components/new-interview/index.tsx
"use client";

import React, { useRef, useState, useEffect, useCallback } from "react";
import { predictNewCheating } from "@/components/new-interview/utils/cheatDetector-new";
import { useModels } from "./hooks/useModels";
import { useWebcam } from "./hooks/useWebcam";
import { calculatePoseFromFace, hasPoseChanged } from "./utils/poseCalculation";
import {
  exportPoseToCSV,
  exportCheatingProbabilitiesToCSV,
} from "./utils/exportUtils";
import { VideoCanvas } from "./components/VideoCanvas";
import { AlertsPanel } from "./components/AlertsPanel";
import { HeadOrientationPanel } from "./components/HeadOrientationPanel";
import { DetectionStatsPanel } from "./components/DetectionStatsPanel";
import { ExportControls } from "./components/ExportControls";
import type {
  HeadPose,
  Detection,
  ViolationAlert,
  CheatingProbability,
} from "./types";

const HeadPoseDetector: React.FC = () => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const animationRef = useRef<number | null>(null);
  const lastTimeRef = useRef<number>(Date.now());
  const frameCountRef = useRef<number>(0);
  const lastPoseRef = useRef<HeadPose | null>(null);

  const { faceModel, objectModel, isLoading, error: modelError } = useModels();
  const [error, setError] = useState<string>("");
  const videoRef = useWebcam(setError);

  const [headPose, setHeadPose] = useState<HeadPose | null>(null);
  const [poseHistory, setPoseHistory] = useState<HeadPose[]>([]);
  const [detections, setDetections] = useState<Detection[]>([]);
  const [violations, setViolations] = useState<ViolationAlert[]>([]);
  const [fps, setFps] = useState(0);
  const [faceCount, setFaceCount] = useState(0);
  const [cheatingProbability, setCheatingProbability] = useState<number>(0);
  const [cheatingProbabilityList, setCheatingProbabilityList] = useState<
    CheatingProbability[]
  >([]);

  const checkCheating = useCallback(
    async (roll: number, pitch: number, yaw: number) => {
      const probability = await predictNewCheating(roll, pitch, yaw);
      setCheatingProbability(Number(probability));
      setCheatingProbabilityList((prev) => [
        ...prev,
        { probability: Number(probability), timestamp: Date.now() },
      ]);
    },
    []
  );

  const addViolation = useCallback(
    (type: ViolationAlert["type"], message: string) => {
      setViolations((prev) => {
        const recent = prev.filter(
          (v) => Date.now() - v.timestamp < 2000 && v.type === type
        );
        if (recent.length > 0) return prev;

        return [...prev.slice(-9), { type, message, timestamp: Date.now() }];
      });
    },
    []
  );

  const predictPose = useCallback(async () => {
    if (
      !videoRef.current ||
      !canvasRef.current ||
      !faceModel ||
      !objectModel ||
      isLoading
    ) {
      animationRef.current = requestAnimationFrame(predictPose);
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    if (!ctx || video.readyState !== 4) {
      animationRef.current = requestAnimationFrame(predictPose);
      return;
    }

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    try {
      const [facePredictions, objectPredictions] = await Promise.all([
        faceModel.estimateFaces(video, false),
        objectModel.detect(video),
      ]);

      // Process faces
      setFaceCount(facePredictions.length);
      if (facePredictions.length > 1) {
        addViolation(
          "multiple_faces",
          `${facePredictions.length} faces detected`
        );
      }

      // Process objects
      const suspiciousObjects: Detection[] = [];
      const phoneClasses = [
        "cell phone",
        "phone",
        "mobile phone",
        "laptop",
        "book",
      ];

      objectPredictions.forEach((pred) => {
        const detection: Detection = {
          class: pred.class,
          score: pred.score,
          bbox: pred.bbox as [number, number, number, number],
        };

        if (
          phoneClasses.some((cls) => pred.class.toLowerCase().includes(cls))
        ) {
          suspiciousObjects.push(detection);

          if (pred.class.toLowerCase().includes("phone")) {
            addViolation(
              "phone",
              `Phone detected (${(pred.score * 100).toFixed(0)}% confidence)`
            );
          } else {
            addViolation("suspicious_object", `${pred.class} detected`);
          }
        }

        // Draw detection boxes
        const [x, y, width, height] = pred.bbox;
        const isPhone = pred.class.toLowerCase().includes("phone");

        ctx.strokeStyle = isPhone ? "#ff0000" : "#ffa500";
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, width, height);

        ctx.fillStyle = isPhone
          ? "rgba(255, 0, 0, 0.7)"
          : "rgba(255, 165, 0, 0.7)";
        ctx.fillRect(x, y - 25, width, 25);

        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 14px Arial";
        ctx.fillText(
          `${pred.class} ${(pred.score * 100).toFixed(0)}%`,
          x + 5,
          y - 8
        );
      });

      setDetections(suspiciousObjects);

      // Draw faces and calculate pose
      if (facePredictions.length > 0) {
        const face = facePredictions[0];
        const box = face.topLeft as number[];
        const bottomRight = face.bottomRight as number[];

        // Draw face box
        ctx.strokeStyle = facePredictions.length > 1 ? "#ff9900" : "#00ff00";
        ctx.lineWidth = 2;
        ctx.strokeRect(
          box[0],
          box[1],
          bottomRight[0] - box[0],
          bottomRight[1] - box[1]
        );

        // Draw landmarks
        const keypoints = face.landmarks as number[][];
        const landmarkLabels = [
          "Right Eye",
          "Left Eye",
          "Nose",
          "Mouth",
          "Right Ear",
          "Left Ear",
        ];

        keypoints.forEach((point, idx) => {
          ctx.fillStyle = "#00ff00";
          ctx.beginPath();
          ctx.arc(point[0], point[1], 4, 0, 2 * Math.PI);
          ctx.fill();

          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 1;
          ctx.stroke();

          ctx.fillStyle = "#ffffff";
          ctx.font = "10px Arial";
          ctx.fillText(landmarkLabels[idx], point[0] + 8, point[1] - 8);
        });

        // Calculate and display pose
        const pose = calculatePoseFromFace(face, canvas.width, canvas.height);
        setHeadPose(pose);

        await checkCheating(pose.roll, pose.pitch, pose.yaw);

        if (hasPoseChanged(pose, lastPoseRef.current)) {
          setPoseHistory((prev) => [...prev, pose]);
          lastPoseRef.current = pose;
        }

        // Draw pose overlay
        ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
        ctx.fillRect(10, 10, 200, 90);
        ctx.fillStyle = "#ffffff";
        ctx.font = "bold 16px Arial";
        ctx.fillText(`Roll:  ${pose.roll.toFixed(1)}°`, 20, 35);
        ctx.fillText(`Pitch: ${pose.pitch.toFixed(1)}°`, 20, 55);
        ctx.fillText(`Yaw:   ${pose.yaw.toFixed(1)}°`, 20, 75);
      } else {
        setHeadPose(null);
      }

      // Calculate FPS
      frameCountRef.current++;
      const currentTime = Date.now();
      if (currentTime - lastTimeRef.current >= 1000) {
        setFps(frameCountRef.current);
        frameCountRef.current = 0;
        lastTimeRef.current = currentTime;
      }
    } catch (err) {
      console.error("Prediction error:", err);
    }

    animationRef.current = requestAnimationFrame(predictPose);
  }, [
    videoRef,
    faceModel,
    objectModel,
    isLoading,
    addViolation,
    checkCheating,
  ]);

  useEffect(() => {
    if (faceModel && objectModel && !isLoading) {
      predictPose();
    }

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, [faceModel, objectModel, isLoading, predictPose]);

  const handleClearHistory = () => {
    setPoseHistory([]);
    setCheatingProbabilityList([]);
    setViolations([]);
    lastPoseRef.current = null;
  };

  const combinedError = error || modelError;

  return (
    <div className="min-h-screen bg-linear-to-br w-full from-gray-900 via-blue-900 to-gray-900 text-white p-8">
      <div className="w-full max-w-7xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-2 bg-clip-text text-transparent bg-linear-to-r from-blue-400 to-purple-400">
            Real-Time Exam Monitoring System
          </h1>
          <p className="text-gray-300">
            Head Pose Detection + Object Recognition
          </p>
        </div>

        {combinedError && (
          <div className="bg-red-900/50 border border-red-500 rounded-lg p-4 mb-6">
            <p className="text-red-200">{combinedError}</p>
          </div>
        )}

        {isLoading && (
          <div className="text-center mb-6">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-400 mb-4"></div>
            <p className="text-gray-300">
              Loading AI models and initializing camera...
            </p>
          </div>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 mb-6">
          <VideoCanvas
            videoRef={videoRef}
            canvasRef={canvasRef}
            fps={fps}
            faceCount={faceCount}
            detectionsCount={detections.length}
            poseHistoryLength={poseHistory.length}
            cheatingProbability={cheatingProbability}
          />
          <AlertsPanel violations={violations} />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-6">
          <HeadOrientationPanel headPose={headPose} />
          <DetectionStatsPanel
            violationsCount={violations.length}
            poseHistoryLength={poseHistory.length}
            faceCount={faceCount}
            detectionsCount={detections.length}
          />
        </div>

        <ExportControls
          onExportPose={() => exportPoseToCSV(poseHistory)}
          onExportCheating={() =>
            exportCheatingProbabilitiesToCSV(cheatingProbabilityList)
          }
          onClearHistory={handleClearHistory}
          hasData={poseHistory.length > 0 || violations.length > 0}
        />
      </div>
    </div>
  );
};

export default HeadPoseDetector;
