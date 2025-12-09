import React, { useEffect, useRef, useState } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl"; // Add this
import "@tensorflow/tfjs-backend-cpu"; // Add this as fallback
import * as blazeface from "@tensorflow-models/blazeface";
import ManHeadViewer from "./ManHeadViewer";
import { predictNewCheating } from "@/utils/cheatDetector-new";

interface HeadPose {
  roll: number;
  pitch: number;
  yaw: number;
  timestamp: number;
}

const HeadPoseDetector: React.FC = () => {
  const videoRef = useRef<HTMLVideoElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [faceModel, setFaceModel] = useState<blazeface.BlazeFaceModel | null>(
    null
  );
  const [headPose, setHeadPose] = useState<HeadPose | null>(null);
  const [poseHistory, setPoseHistory] = useState<HeadPose[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string>("");
  const [fps, setFps] = useState(0);
  const animationRef = useRef<number>();
  const lastTimeRef = useRef<number>(Date.now());
  const frameCountRef = useRef<number>(0);
  const lastPoseRef = useRef<HeadPose | null>(null);
  const [cheatingProbability, setCheatingProbability] = useState<number>(0);

  const checkCheating = async (roll: number, pitch: number, yaw: number) => {
      const probability = await predictNewCheating(
        roll,
        pitch,
        yaw
      );
      // console.log("Cheating probability:", probability);
      setCheatingProbability(probability);
      // You can add further logic here based on the probability value
    };

  // Load BlazeFace model
  useEffect(() => {
    const loadModels = async () => {
      try {
        setIsLoading(true);
        setError("");

        // Initialize TensorFlow.js backend first
        await tf.ready();
        console.log("✓ TensorFlow.js backend initialized:", tf.getBackend());

        const face = await blazeface.load();
        setFaceModel(face);
        console.log("✓ BlazeFace model loaded successfully");
        setIsLoading(false);
      } catch (err) {
        console.error("Error initializing:", err);
        setError("Failed to initialize face detection.");
        setIsLoading(false);
      }
    };

    loadModels();

    return () => {
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current);
      }
    };
  }, []);

  // Setup webcam
  useEffect(() => {
    const setupWebcam = async () => {
      try {
        const stream = await navigator.mediaDevices.getUserMedia({
          video: { width: 640, height: 480, facingMode: "user" },
        });

        if (videoRef.current) {
          videoRef.current.srcObject = stream;
          videoRef.current.onloadedmetadata = () => {
            videoRef.current?.play();
          };
        }
      } catch (err) {
        console.error("Error accessing webcam:", err);
        setError("Failed to access webcam. Please grant camera permissions.");
      }
    };

    setupWebcam();

    return () => {
      if (videoRef.current?.srcObject) {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
        tracks.forEach((track) => track.stop());
      }
    };
  }, []);

  // Calculate head pose from face position and landmarks
  const calculatePoseFromFace = (
    face: any,
    videoWidth: number,
    videoHeight: number
  ): HeadPose => {
    const box = face.topLeft as number[];
    const bottomRight = face.bottomRight as number[];
    const keypoints = face.landmarks as number[][];

    const centerX = (box[0] + bottomRight[0]) / 2;
    const centerY = (box[1] + bottomRight[1]) / 2;
    const width = bottomRight[0] - box[0];
    const height = bottomRight[1] - box[1];

    // Extract key landmarks (BlazeFace provides 6 keypoints)
    // 0: right eye, 1: left eye, 2: nose, 3: mouth, 4: right ear, 5: left ear
    const rightEye = keypoints[0];
    const leftEye = keypoints[1];
    const nose = keypoints[2];
    const mouth = keypoints[3];

    // Calculate yaw (left-right rotation) from horizontal position and eye alignment
    const normalizedX = (centerX / videoWidth - 0.5) * 2;
    const eyeCenterX = (rightEye[0] + leftEye[0]) / 2;
    const noseOffsetX = nose[0] - eyeCenterX;
    const yaw = normalizedX * 45 + (noseOffsetX / width) * 30;

    // Calculate pitch (up-down rotation) from vertical position and nose-mouth distance
    const normalizedY = (centerY / videoHeight - 0.5) * 2;
    const noseToMouthDist = mouth[1] - nose[1];
    const expectedDist = height * 0.2;
    const pitchFromDist =
      ((noseToMouthDist - expectedDist) / expectedDist) * 20;
    const pitch = normalizedY * 30 + pitchFromDist;

    // Calculate roll (tilt) from eye alignment
    const eyeDx = leftEye[0] - rightEye[0];
    const eyeDy = leftEye[1] - rightEye[1];
    const roll = Math.atan2(eyeDy, eyeDx) * (180 / Math.PI);

    return {
      roll: Math.max(-90, Math.min(90, roll)),
      pitch: Math.max(-90, Math.min(90, pitch)),
      yaw: Math.max(-90, Math.min(90, yaw)),
      timestamp: Date.now(),
    };
  };

  // Check if pose has changed significantly
  const hasPoseChanged = (newPose: HeadPose, threshold = 2): boolean => {
    if (!lastPoseRef.current) return true;

    const rollDiff = Math.abs(newPose.roll - lastPoseRef.current.roll);
    const pitchDiff = Math.abs(newPose.pitch - lastPoseRef.current.pitch);
    const yawDiff = Math.abs(newPose.yaw - lastPoseRef.current.yaw);

    return rollDiff > threshold || pitchDiff > threshold || yawDiff > threshold;
  };

  // Export pose history as CSV
  const exportToCSV = () => {
    if (poseHistory.length === 0) {
      alert("No orientation data to export!");
      return;
    }

    const csvHeader = "Timestamp,Date,Roll,Pitch,Yaw\n";
    const csvRows = poseHistory
      .map((pose) => {
        const date = new Date(pose.timestamp).toISOString();
        return `${pose.timestamp},${date},${pose.roll.toFixed(
          2
        )},${pose.pitch.toFixed(2)},${pose.yaw.toFixed(2)}`;
      })
      .join("\n");

    const csv = csvHeader + csvRows;
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `head-pose-${Date.now()}.csv`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // Clear pose history
  const clearHistory = () => {
    setPoseHistory([]);
    lastPoseRef.current = null;
  };

  // Main prediction loop
  const predictPose = async () => {
    if (!videoRef.current || !canvasRef.current || !faceModel || isLoading) {
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
      const predictions = await faceModel.estimateFaces(video, false);

      if (predictions.length > 0) {
        const face = predictions[0];

        // Draw face box
        const box = face.topLeft as number[];
        const bottomRight = face.bottomRight as number[];
        ctx.strokeStyle = "#00ff00";
        ctx.lineWidth = 2;
        ctx.strokeRect(
          box[0],
          box[1],
          bottomRight[0] - box[0],
          bottomRight[1] - box[1]
        );

        // Draw landmarks (keypoints)
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
          // Draw landmark point
          ctx.fillStyle = "#00ff00";
          ctx.beginPath();
          ctx.arc(point[0], point[1], 4, 0, 2 * Math.PI);
          ctx.fill();

          // Draw landmark border
          ctx.strokeStyle = "#ffffff";
          ctx.lineWidth = 1;
          ctx.stroke();

          // Draw landmark label
          ctx.fillStyle = "#ffffff";
          ctx.font = "10px Arial";
          ctx.fillText(landmarkLabels[idx], point[0] + 8, point[1] - 8);
        });

        // Calculate pose
        const pose = calculatePoseFromFace(face, canvas.width, canvas.height);
        setHeadPose(pose);

        // Check for cheating
        await checkCheating(pose.roll, pose.pitch, pose.yaw);
        // Save pose if it has changed significantly
        if (hasPoseChanged(pose)) {
          setPoseHistory((prev) => [...prev, pose]);
          lastPoseRef.current = pose;
        }

        // Draw pose info on canvas
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
  };

  // Start prediction when model is ready
  useEffect(() => {
    if (faceModel && !isLoading) {
      predictPose();
    }
  }, [faceModel, isLoading]);

  return (
    <div className="min-h-screen bg-gradient-to-br w-full from-gray-900 via-blue-900 to-gray-900 text-white p-8">
      <div className="w-full px-[10vw] mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-4xl font-bold mb-2 bg-clip-text text-transparent bg-gradient-to-r from-blue-400 to-purple-400">
            Real-Time Head Pose Detection
          </h1>
          <p className="text-gray-300">Using BlazeFace Face Detection</p>
        </div>

        {error && (
          <div className="bg-red-900/50 border border-red-500 rounded-lg p-4 mb-6">
            <p className="text-red-200">{error}</p>
          </div>
        )}

        {isLoading && (
          <div className="text-center mb-6">
            <div className="inline-block animate-spin rounded-full h-12 w-12 border-b-2 border-blue-400 mb-4"></div>
            <p className="text-gray-300">
              Loading BlazeFace model and initializing camera...
            </p>
          </div>
        )}

        <div className="flex w-full gap-6 mb-6">
          {/* Video/Canvas Display */}
          <div className="bg-gray-800/50 w-full relative rounded-lg p-6 backdrop-blur-sm border border-gray-700">
            <h2 className="text-xl font-semibold mb-4">Live Feed</h2>
            <div className="relative bg-black rounded-lg overflow-hidden aspect-video">
              <div className="flex absolute top-[100px] right-[100px] flex-col gap-4">
                <div className="flex z-100 justify-center items-center border border-white p-6 font-[18px] ">
                  Cheating Probability: {(cheatingProbability * 100).toFixed(2)}
                  %
                </div>
                {/* <div className="flex  justify-center items-center border border-white p-6 font-[18px] ">
                  Pose Initialised: {initialPoseRef.current ? "Yes" : "No"}
                </div> */}
              </div>
              <video
                ref={videoRef}
                className="absolute inset-0 w-full h-full object-cover"
                autoPlay
                playsInline
                muted
              />
              <canvas ref={canvasRef} className="relative w-full h-full" />
              <div className="absolute top-2 right-2 bg-black/70 px-3 py-1 rounded text-sm">
                {fps} FPS
              </div>
              <div className="absolute bottom-2 left-2 bg-black/70 px-3 py-1 rounded text-xs">
                Recorded: {poseHistory.length} poses
              </div>
            </div>
          </div>
          {/* <div
            className="w-full bg-red-400/10 rounded-lg p-6 backdrop-blur-sm border border-red-300/10 flex items-center justify-center"
            
          >
            <ManHeadViewer />
          </div> */}

          {/* Pose Information */}
          <div className="bg-gray-800/50 rounded-lg p-6 backdrop-blur-sm border border-gray-700">
            <h2 className="text-xl font-semibold mb-4">Head Orientation</h2>

            {headPose ? (
              <div className="space-y-6">
                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-red-400 font-semibold">Roll</span>
                    <span className="text-2xl font-bold">
                      {headPose.roll.toFixed(1)}°
                    </span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-3">
                    <div
                      className="bg-gradient-to-r from-red-500 to-red-400 h-3 rounded-full transition-all duration-200"
                      style={{
                        width: `${Math.min(
                          (Math.abs(headPose.roll) / 90) * 100,
                          100
                        )}%`,
                      }}
                    />
                  </div>
                  <p className="text-sm text-gray-400 mt-1">
                    Side tilt (ear to shoulder)
                  </p>
                </div>

                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-green-400 font-semibold">Pitch</span>
                    <span className="text-2xl font-bold">
                      {headPose.pitch.toFixed(1)}°
                    </span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-3">
                    <div
                      className="bg-gradient-to-r from-green-500 to-green-400 h-3 rounded-full transition-all duration-200"
                      style={{
                        width: `${Math.min(
                          (Math.abs(headPose.pitch) / 90) * 100,
                          100
                        )}%`,
                      }}
                    />
                  </div>
                  <p className="text-sm text-gray-400 mt-1">
                    Up/Down (nodding)
                  </p>
                </div>

                <div>
                  <div className="flex justify-between items-center mb-2">
                    <span className="text-blue-400 font-semibold">Yaw</span>
                    <span className="text-2xl font-bold">
                      {headPose.yaw.toFixed(1)}°
                    </span>
                  </div>
                  <div className="w-full bg-gray-700 rounded-full h-3">
                    <div
                      className="bg-gradient-to-r from-blue-500 to-blue-400 h-3 rounded-full transition-all duration-200"
                      style={{
                        width: `${Math.min(
                          (Math.abs(headPose.yaw) / 90) * 100,
                          100
                        )}%`,
                      }}
                    />
                  </div>
                  <p className="text-sm text-gray-400 mt-1">
                    Left/Right (shaking head)
                  </p>
                </div>

                {/* Visual representation */}

                <div className="mt-8">
                  <h3 className="text-lg font-semibold mb-4 text-center">
                    3D Orientation
                  </h3>
                  <div className="relative w-48 h-48 mx-auto perspective-1000">
                    <div
                      className="absolute inset-0 bg-gradient-to-br from-blue-500/20 to-purple-500/20 rounded-full border-2 border-blue-400 transition-transform duration-200"
                      style={{
                        transform: `rotateX(${-headPose.pitch}deg) rotateY(${
                          headPose.yaw
                        }deg) rotateZ(${headPose.roll}deg)`,
                        transformStyle: "preserve-3d",
                      }}
                    >
                      <div className="absolute inset-0 flex items-center justify-center">
                        <div className="w-4 h-4 bg-white rounded-full shadow-lg"></div>
                      </div>
                      {/* Nose direction indicator */}
                      <div
                        className="absolute top-1/2 left-1/2 w-2 h-16 bg-yellow-400 -translate-x-1/2 origin-top"
                        style={{
                          transformStyle: "preserve-3d",
                          transform: "translateZ(20px)",
                        }}
                      />
                    </div>
                  </div>
                </div>
              </div>
            ) : (
              <div className="text-center py-12 text-gray-400">
                <svg
                  className="w-16 h-16 mx-auto mb-4 opacity-50"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z"
                  />
                </svg>
                <p>No face detected</p>
                <p className="text-sm mt-2">
                  Position your face in front of the camera
                </p>
              </div>
            )}
          </div>
        </div>

        {/* Export Controls */}
        <div className="bg-gray-800/50 rounded-lg p-6 backdrop-blur-sm border border-gray-700 mb-6">
          <div className="flex items-center justify-between flex-wrap gap-4">
            <div>
              <h3 className="text-lg font-semibold mb-1">
                Orientation History
              </h3>
              <p className="text-sm text-gray-400">
                {poseHistory.length} orientation changes recorded
              </p>
            </div>
            <div className="flex gap-3">
              <button
                onClick={clearHistory}
                disabled={poseHistory.length === 0}
                className="px-6 py-2 bg-red-600 hover:bg-red-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg font-semibold transition-colors"
              >
                Clear History
              </button>
              <button
                onClick={exportToCSV}
                disabled={poseHistory.length === 0}
                className="px-6 py-2 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed rounded-lg font-semibold transition-colors flex items-center gap-2"
              >
                <svg
                  className="w-5 h-5"
                  fill="none"
                  stroke="currentColor"
                  viewBox="0 0 24 24"
                >
                  <path
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth={2}
                    d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
                  />
                </svg>
                Export CSV
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HeadPoseDetector;
