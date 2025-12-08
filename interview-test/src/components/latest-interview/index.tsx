import React, { useRef, useEffect, useState, useCallback } from "react";
import Webcam from "react-webcam";
import * as tf from "@tensorflow/tfjs";
import * as faceapi from "face-api.js";

// Types for our data
interface HeadPoseData {
  timestamp: string;
  roll: number;
  pitch: number;
  yaw: number;
}

interface ScalerParams {
  mean: number[];
  scale: number[];
}

const HeadPoseTracker: React.FC = () => {
  const webcamRef = useRef<Webcam>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // State
  const [isRecording, setIsRecording] = useState(false);
  const [modelsLoaded, setModelsLoaded] = useState(false);
  const [recordedData, setRecordedData] = useState<HeadPoseData[]>([]);
  const [currentPose, setCurrentPose] = useState<{
    roll: number;
    pitch: number;
    yaw: number;
  } | null>(null);

  // Refs for heavyweight objects to avoid re-renders
  const customModelRef = useRef<tf.LayersModel | null>(null);
  const scalerParamsRef = useRef<ScalerParams | null>(null);
  const requestRef = useRef<number>();

  // 1. Load Resources
  useEffect(() => {
    const loadResources = async () => {
      try {
        // Load Face API models
        await faceapi.nets.tinyFaceDetector.loadFromUri("/models/face-api");
        await faceapi.nets.faceLandmark68Net.loadFromUri("/models/face-api");

        // Load your custom TFJS model
        customModelRef.current = await tf.loadLayersModel(
          "/models/headpose/model.json"
        );

        // Load Scaler Params
        const response = await fetch("/models/headpose/scaler_params.json");
        scalerParamsRef.current = await response.json();

        console.log("All models and params loaded");
        setModelsLoaded(true);
      } catch (error) {
        console.error("Failed to load models:", error);
      }
    };

    loadResources();
  }, []);

  // 2. Helper: Calculate Pairwise Distances (Matches your Python `compute_features`)
  const computeFeatures = (landmarks: faceapi.Point[]): number[] => {
    const features: number[] = [];
    // Calculate distance between every pair of the 68 points
    // This loops 68 * 67 / 2 times (~2278 features)
    for (let i = 0; i < 68; i++) {
      for (let j = i + 1; j < 68; j++) {
        const p1 = landmarks[i];
        const p2 = landmarks[j];
        // Euclidean distance
        const dist = Math.sqrt(
          Math.pow(p1.x - p2.x, 2) + Math.pow(p1.y - p2.y, 2)
        );
        features.push(dist);
      }
    }
    return features;
  };

  // 3. Helper: Standardize Features (Matches your Python `scaler.transform`)
  const standardizeFeatures = (features: number[]): number[] => {
    if (!scalerParamsRef.current) return features;
    const { mean, scale } = scalerParamsRef.current;

    return features.map((val, index) => {
      // (x - mean) / std_dev
      return (val - mean[index]) / scale[index];
    });
  };

  // 4. Main Detection Loop
  const detect = useCallback(async () => {
    if (
      typeof webcamRef.current !== "undefined" &&
      webcamRef.current !== null &&
      webcamRef.current.video?.readyState === 4 &&
      modelsLoaded &&
      customModelRef.current
    ) {
      // Get Video Properties
      const video = webcamRef.current.video;
      const videoWidth = video.videoWidth;
      const videoHeight = video.videoHeight;

      // Set canvas dimensions
      if (canvasRef.current) {
        canvasRef.current.width = videoWidth;
        canvasRef.current.height = videoHeight;
      }

      // A. Detect Face & Landmarks (using TinyFaceDetector for speed)
      const detection = await faceapi
        .detectSingleFace(video, new faceapi.TinyFaceDetectorOptions())
        .withFaceLandmarks();

      if (detection) {
        // B. Preprocessing
        const landmarks = detection.landmarks.positions; // 68 points
        const rawFeatures = computeFeatures(landmarks);
        const processedFeatures = standardizeFeatures(rawFeatures);

        // C. Prediction
        // Create tensor [1, 2278]
        const inputTensor = tf.tensor2d([processedFeatures]);
        const prediction = customModelRef.current.predict(
          inputTensor
        ) as tf.Tensor;
        const [roll, pitch, yaw] = (await prediction.data()) as Float32Array;

        // Cleanup tensor to avoid memory leaks
        inputTensor.dispose();
        prediction.dispose();

        // Update State
        setCurrentPose({ roll, pitch, yaw });

        if (isRecording) {
          setRecordedData((prev) => [
            ...prev,
            {
              timestamp: new Date().toISOString(),
              roll,
              pitch,
              yaw,
            },
          ]);
        }

        // D. Visualizing (Drawing landmarks + pose info)
        if (canvasRef.current) {
          const ctx = canvasRef.current.getContext("2d");
          if (ctx) {
            ctx.clearRect(0, 0, videoWidth, videoHeight);

            // Draw landmarks
            faceapi.draw.drawFaceLandmarks(canvasRef.current, detection);

            // Draw Pose Data text
            ctx.fillStyle = "red";
            ctx.font = "20px Arial";
            ctx.fillText(`Roll: ${roll.toFixed(2)}°`, 10, 30);
            ctx.fillText(`Pitch: ${pitch.toFixed(2)}°`, 10, 60);
            ctx.fillText(`Yaw: ${yaw.toFixed(2)}°`, 10, 90);
          }
        }
      }
    }

    requestRef.current = requestAnimationFrame(detect);
  }, [modelsLoaded, isRecording]);

  useEffect(() => {
    requestRef.current = requestAnimationFrame(detect);
    return () => {
      if (requestRef.current) cancelAnimationFrame(requestRef.current);
    };
  }, [detect]);

  // 5. CSV Export
  const downloadCSV = () => {
    if (recordedData.length === 0) {
      alert("No data recorded yet!");
      return;
    }

    const headers = ["Timestamp,Roll,Pitch,Yaw\n"];
    const csvContent = headers
      .concat(
        recordedData.map((d) => `${d.timestamp},${d.roll},${d.pitch},${d.yaw}`)
      )
      .join("\n");

    const blob = new Blob([csvContent], { type: "text/csv;charset=utf-8;" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.setAttribute("download", `head_pose_data_${new Date().getTime()}.csv`);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
  };

  return (
    <div className="flex flex-col items-center gap-4 p-4">
      <h2 className="text-2xl font-bold">Head Pose Estimator</h2>

      {!modelsLoaded && (
        <div className="text-blue-500">Loading Models... (Please wait)</div>
      )}

      <div className="relative border-4 border-gray-800 rounded-lg overflow-hidden">
        <Webcam
          ref={webcamRef}
          audio={false}
          width={640}
          height={480}
          className="block"
          screenshotFormat="image/jpeg"
        />
        <canvas ref={canvasRef} className="absolute top-0 left-0" />
      </div>

      <div className="flex gap-4 mt-4">
        <button
          onClick={() => setIsRecording(!isRecording)}
          className={`px-6 py-2 rounded-full font-bold text-white transition-colors ${
            isRecording
              ? "bg-red-500 hover:bg-red-600"
              : "bg-green-500 hover:bg-green-600"
          }`}
        >
          {isRecording ? "Stop Recording" : "Start Recording"}
        </button>

        <button
          onClick={downloadCSV}
          className="px-6 py-2 bg-blue-500 hover:bg-blue-600 text-white rounded-full font-bold transition-colors"
        >
          Export CSV ({recordedData.length} samples)
        </button>
      </div>

      <div className="mt-4 grid grid-cols-3 gap-4 text-center">
        <div className="bg-gray-100 p-4 rounded">
          <p className="text-gray-500">Roll</p>
          <p className="text-xl font-mono">
            {currentPose?.roll.toFixed(1) || "--"}°
          </p>
        </div>
        <div className="bg-gray-100 p-4 rounded">
          <p className="text-gray-500">Pitch</p>
          <p className="text-xl font-mono">
            {currentPose?.pitch.toFixed(1) || "--"}°
          </p>
        </div>
        <div className="bg-gray-100 p-4 rounded">
          <p className="text-gray-500">Yaw</p>
          <p className="text-xl font-mono">
            {currentPose?.yaw.toFixed(1) || "--"}°
          </p>
        </div>
      </div>
    </div>
  );
};

export default HeadPoseTracker;
