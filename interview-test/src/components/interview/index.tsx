/* eslint-disable react-hooks/immutability */
"use client";

import React, { useEffect, useRef, useState, useCallback } from "react";
import * as tf from "@tensorflow/tfjs";
import * as faceapi from "@vladmandic/face-api";

const HeadPoseEstimation = () => {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [model, setModel] = useState<tf.LayersModel | null>(null);
  const [scaler, setScaler] = useState<{
    mean: number[];
    scale: number[];
  } | null>(null);
  const [pose, setPose] = useState({
    roll: "0.00",
    pitch: "0.00",
    yaw: "0.00",
  });
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [faceDetected, setFaceDetected] = useState(false);
  const animationRef = useRef<number | null>(null);
  const [setupStep, setSetupStep] = useState("Initializing...");

  // Initialize webcam
  useEffect(() => {
    const setupWebcam = async () => {
      try {
        setSetupStep("Accessing webcam...");
        const stream = await navigator.mediaDevices.getUserMedia({
          video: {
            width: 640,
            height: 480,
            facingMode: "user",
          },
        });
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      } catch (err) {
        setError(
          "Failed to access webcam. Please ensure you have granted camera permissions."
        );
        setIsLoading(false);
      }
    };
    setupWebcam();

    return () => {
      if (videoRef.current && videoRef.current.srcObject) {
        const stream = videoRef.current.srcObject as MediaStream;
        stream.getTracks().forEach((track) => track.stop());
      }
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  // Load face-api.js models and custom pose model
  useEffect(() => {
    const loadModels = async () => {
      try {
        // Load face-api.js models (lightweight landmark detector)
        setSetupStep("Loading face detection model...");

        const MODEL_URL =
          "https://cdn.jsdelivr.net/npm/@vladmandic/face-api/model";
        await faceapi.nets.tinyFaceDetector.loadFromUri(MODEL_URL);
        await faceapi.nets.faceLandmark68Net.loadFromUri(MODEL_URL);
        console.log("Face detection models loaded ✅");

        setSetupStep("Loading pose estimation model...");
        try {
          const poseModel = await tf.loadLayersModel("/model/model.json", {
            strict: false, // Add this option to be more lenient with layer configs
          });
          setModel(poseModel);
          console.log("Custom pose model loaded ✅");
          console.log("Model summary:");
          poseModel.summary();
        } catch (modelErr) {
          console.error("Custom model loading failed:", modelErr);
          throw new Error(
            `Failed to load custom model: ${
              modelErr instanceof Error ? modelErr.message : "Unknown error"
            }`
          );
        }
        // Load scaler parameters
        setSetupStep("Loading scaler parameters...");
        try {
          const scalerResponse = await fetch("/model/scaler_params.json");
          if (!scalerResponse.ok) {
            throw new Error(
              `HTTP ${scalerResponse.status}: ${scalerResponse.statusText}`
            );
          }
          const scalerParams = await scalerResponse.json();
          setScaler(scalerParams);
          console.log("Scaler parameters loaded ✅");
        } catch (scalerErr) {
          console.error("Scaler loading failed:", scalerErr);
          throw new Error(
            `Failed to load scaler: ${
              scalerErr instanceof Error ? scalerErr.message : "Unknown error"
            }`
          );
        }

        setSetupStep("Ready!");
        setIsLoading(false);
        console.log("All models loaded successfully ✅");
      } catch (err) {
        const errorMessage =
          err instanceof Error ? err.message : "Unknown error";
        setError("Failed to load models: " + errorMessage);
        setIsLoading(false);
        console.error("Full error:", err);
      }
    };
    loadModels();
  }, []);

  // Compute pairwise distances between facial landmarks
  const computeFeatures = useCallback((landmarks: faceapi.Point[]) => {
    const features: number[] = [];
    const points = landmarks.map((lm) => [lm.x, lm.y]);

    for (let i = 0; i < 68; i++) {
      for (let j = i + 1; j < 68; j++) {
        const dx = points[i][0] - points[j][0];
        const dy = points[i][1] - points[j][1];
        const distance = Math.sqrt(dx * dx + dy * dy);
        features.push(distance);
      }
    }
    return features;
  }, []);

  // Standardize features using saved scaler parameters
  const standardizeFeatures = useCallback(
    (
      features: number[],
      scaler: { mean: number[]; scale: number[] } | null
    ) => {
      if (!scaler || !scaler.mean || !scaler.scale) return features;
      return features.map((f, i) => (f - scaler.mean[i]) / scaler.scale[i]);
    },
    []
  );

  // Main detection and estimation loop
  const detectAndEstimate = useCallback(async () => {
    
    if (
      !videoRef.current ||
      !canvasRef.current ||
      !model ||
      !scaler ||
      isLoading
    ) {
      animationRef.current = requestAnimationFrame(detectAndEstimate);
      return;
    }

    if (
      !videoRef.current ||
      !canvasRef.current ||
      !model ||
      !scaler ||
      isLoading
    ) {
      animationRef.current = requestAnimationFrame(detectAndEstimate);
      return;
    }

    const video = videoRef.current;
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");

    if (!ctx) {
      animationRef.current = requestAnimationFrame(detectAndEstimate);
      return;
    }

    // Check if video is actually playing
    if (video.readyState !== video.HAVE_ENOUGH_DATA) {
      animationRef.current = requestAnimationFrame(detectAndEstimate);
      return;
    }

    if (
      canvas.width !== video.videoWidth ||
      canvas.height !== video.videoHeight
    ) {
      canvas.width = video.videoWidth;
      canvas.height = video.videoHeight;
    }

    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    try {
      // Detect face and 68 landmarks using face-api.js
      const detection = await faceapi
        .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks();

      if (detection) {
        setFaceDetected(true);
        const landmarks = detection.landmarks.positions; // 68 landmarks

        // Draw landmarks
        landmarks.forEach((point, index) => {
          ctx.beginPath();
          ctx.arc(point.x, point.y, 2, 0, 2 * Math.PI);
          ctx.fillStyle =
            index < 17
              ? "#00ff00"
              : index < 27
              ? "#ffff00"
              : index < 36
              ? "#ff00ff"
              : index < 48
              ? "#00ffff"
              : "#ff0000";
          ctx.fill();
        });

        // Compute features and predict pose using YOUR custom model
        const features = computeFeatures(landmarks);
        const standardizedFeatures = standardizeFeatures(features, scaler);

        const inputTensor = tf.tensor2d([standardizedFeatures]);
        const prediction = model.predict(inputTensor) as tf.Tensor;
        const predictionArray = await prediction.data();

        const estimatedPose = {
          roll: predictionArray[0],
          pitch: predictionArray[1],
          yaw: predictionArray[2],
        };

        setPose({
          roll: estimatedPose.roll.toFixed(2),
          pitch: estimatedPose.pitch.toFixed(2),
          yaw: estimatedPose.yaw.toFixed(2),
        });

        // Draw pose info overlay
        ctx.fillStyle = "rgba(0, 0, 0, 0.7)";
        ctx.fillRect(10, 10, 220, 110);
        ctx.fillStyle = "white";
        ctx.font = "16px monospace";
        ctx.fillText(`Roll:  ${estimatedPose.roll.toFixed(2)}°`, 20, 35);
        ctx.fillText(`Pitch: ${estimatedPose.pitch.toFixed(2)}°`, 20, 60);
        ctx.fillText(`Yaw:   ${estimatedPose.yaw.toFixed(2)}°`, 20, 85);
        ctx.font = "12px monospace";
        ctx.fillStyle = "#888";
        ctx.fillText("Custom ML Model", 20, 100);

        inputTensor.dispose();
        prediction.dispose();
      } else {
        setFaceDetected(false);
        ctx.fillStyle = "rgba(255, 0, 0, 0.7)";
        ctx.fillRect(10, 10, 220, 50);
        ctx.fillStyle = "white";
        ctx.font = "16px monospace";
        ctx.fillText("No face detected", 20, 35);
      }
    } catch (err) {
      console.error("Detection error:", err);
    }

    animationRef.current = requestAnimationFrame(detectAndEstimate);
  }, [model, scaler, isLoading, computeFeatures, standardizeFeatures]);

  // Start detection when video is ready
  const handleVideoPlay = useCallback(() => {
    console.log("Video playing, starting detection...");
    console.log(
      "isLoading:",
      isLoading,
      "model:",
      !!model,
      "scaler:",
      !!scaler
    );
    if (!isLoading && model && scaler) {
      detectAndEstimate();
    }
  }, [isLoading, model, scaler, detectAndEstimate]);

  // Start detection when models finish loading (if video already playing)
  useEffect(() => {
    if (
      !isLoading &&
      model &&
      scaler &&
      videoRef.current &&
      !videoRef.current.paused
    ) {
      console.log("Models loaded, video already playing - starting detection");
      if (!animationRef.current) {
        detectAndEstimate();
      }
    }
  }, [isLoading, model, scaler, detectAndEstimate]);

  useEffect(() => {
    if (!videoRef.current) return;

    const video = videoRef.current;
    const keepAlive = () => {
      if (video.paused) video.play();
    };

    video.addEventListener("loadeddata", keepAlive);
    video.addEventListener("pause", keepAlive);

    return () => {
      video.removeEventListener("loadeddata", keepAlive);
      video.removeEventListener("pause", keepAlive);
    };
  }, []);


  if (error) {
    return (
      <div className="flex items-center justify-center min-h-screen bg-gray-900 text-white p-4">
        <div className="text-center p-8 bg-red-900 bg-opacity-50 rounded-lg max-w-md">
          <h2 className="text-xl font-bold mb-4">⚠️ Error</h2>
          <p className="mb-4">{error}</p>
          <button
            onClick={() => window.location.reload()}
            className="px-4 py-2 bg-red-700 hover:bg-red-600 rounded"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="flex flex-col items-center justify-center min-h-screen bg-gray-900 p-4">
      <div className="max-w-4xl w-full">
        <h1 className="text-3xl font-bold text-white mb-2 text-center">
          Head Pose Estimation
        </h1>
        <p className="text-gray-400 text-center mb-6">
          Using custom ML model for pose estimation
        </p>

        {isLoading && (
          <div className="text-white text-center mb-4 bg-gray-800 p-6 rounded-lg">
            <div className="inline-block animate-spin rounded-full h-8 w-8 border-b-2 border-white mb-2"></div>
            <p className="text-lg">{setupStep}</p>
          </div>
        )}

        <div className="bg-gray-800 rounded-lg shadow-2xl overflow-hidden">
          <div className="relative">
            <video
              ref={videoRef}
              autoPlay
              playsInline
              muted
              onPlay={handleVideoPlay}
              className="absolute top-0 left-0 w-full h-auto pointer-events-none opacity-[0.01]"
            />

            <canvas ref={canvasRef} className="w-full h-auto" />

            {!isLoading && (
              <div className="absolute top-4 right-4">
                <div
                  className={`px-3 py-1 rounded-full text-sm font-medium ${
                    faceDetected
                      ? "bg-green-500 text-white"
                      : "bg-red-500 text-white"
                  }`}
                >
                  {faceDetected ? "● Face Detected" : "○ No Face"}
                </div>
              </div>
            )}
          </div>

          {!isLoading && (
            <div className="p-6 bg-gray-700">
              <div className="grid grid-cols-3 gap-4 text-center">
                <div className="bg-gray-600 p-4 rounded-lg">
                  <div className="text-gray-300 text-sm mb-1">Roll (Tilt)</div>
                  <div className="text-2xl font-bold text-blue-400">
                    {pose.roll}°
                  </div>
                  <div className="text-xs text-gray-400 mt-1">← →</div>
                </div>
                <div className="bg-gray-600 p-4 rounded-lg">
                  <div className="text-gray-300 text-sm mb-1">Pitch (Nod)</div>
                  <div className="text-2xl font-bold text-green-400">
                    {pose.pitch}°
                  </div>
                  <div className="text-xs text-gray-400 mt-1">↑ ↓</div>
                </div>
                <div className="bg-gray-600 p-4 rounded-lg">
                  <div className="text-gray-300 text-sm mb-1">Yaw (Turn)</div>
                  <div className="text-2xl font-bold text-purple-400">
                    {pose.yaw}°
                  </div>
                  <div className="text-xs text-gray-400 mt-1">↶ ↷</div>
                </div>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default HeadPoseEstimation;
