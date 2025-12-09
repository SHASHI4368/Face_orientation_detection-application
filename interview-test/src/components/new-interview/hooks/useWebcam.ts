/* eslint-disable react-hooks/exhaustive-deps */
// src/components/new-interview/hooks/useWebcam.ts
import { useEffect, useRef } from "react";

export function useWebcam(onError: (error: string) => void) {
  const videoRef = useRef<HTMLVideoElement>(null);

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
        onError("Failed to access webcam. Please grant camera permissions.");
      }
    };

    setupWebcam();

    return () => {
      if (videoRef.current?.srcObject) {
        const tracks = (videoRef.current.srcObject as MediaStream).getTracks();
        tracks.forEach((track) => track.stop());
      }
    };
  }, [onError]);

  return videoRef;
}
