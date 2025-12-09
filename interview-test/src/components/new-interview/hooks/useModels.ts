// src/components/new-interview/hooks/useModels.ts
import { useState, useEffect } from "react";
import * as tf from "@tensorflow/tfjs";
import "@tensorflow/tfjs-backend-webgl";
import "@tensorflow/tfjs-backend-cpu";
import * as blazeface from "@tensorflow-models/blazeface";
import * as cocoSsd from "@tensorflow-models/coco-ssd";

export function useModels() {
  const [faceModel, setFaceModel] = useState<blazeface.BlazeFaceModel | null>(
    null
  );
  const [objectModel, setObjectModel] =
    useState<cocoSsd.ObjectDetection | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    const loadModels = async () => {
      try {
        setIsLoading(true);
        setError("");

        await tf.ready();
        console.log("✓ TensorFlow.js backend initialized:", tf.getBackend());

        const [face, object] = await Promise.all([
          blazeface.load(),
          cocoSsd.load(),
        ]);

        setFaceModel(face);
        setObjectModel(object);
        console.log("✓ BlazeFace and COCO-SSD models loaded");
        setIsLoading(false);
      } catch (err) {
        console.error("Error initializing:", err);
        setError("Failed to initialize models.");
        setIsLoading(false);
      }
    };

    loadModels();
  }, []);

  return { faceModel, objectModel, isLoading, error };
}
