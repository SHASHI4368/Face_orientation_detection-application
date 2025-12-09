/* eslint-disable @typescript-eslint/no-explicit-any */
// src/components/new-interview/utils/poseCalculation.ts
import { HeadPose } from "../types";

export function calculatePoseFromFace(
  face: any,
  videoWidth: number,
  videoHeight: number
): HeadPose {
  const box = face.topLeft as number[];
  const bottomRight = face.bottomRight as number[];
  const keypoints = face.landmarks as number[][];

  const centerX = (box[0] + bottomRight[0]) / 2;
  const centerY = (box[1] + bottomRight[1]) / 2;
  const width = bottomRight[0] - box[0];
  const height = bottomRight[1] - box[1];

  const rightEye = keypoints[0];
  const leftEye = keypoints[1];
  const nose = keypoints[2];
  const mouth = keypoints[3];

  const normalizedX = (centerX / videoWidth - 0.5) * 2;
  const eyeCenterX = (rightEye[0] + leftEye[0]) / 2;
  const noseOffsetX = nose[0] - eyeCenterX;
  const yaw = normalizedX * 45 + (noseOffsetX / width) * 30;

  const normalizedY = (centerY / videoHeight - 0.5) * 2;
  const noseToMouthDist = mouth[1] - nose[1];
  const expectedDist = height * 0.2;
  const pitchFromDist = ((noseToMouthDist - expectedDist) / expectedDist) * 20;
  const pitch = normalizedY * 30 + pitchFromDist;

  const eyeDx = leftEye[0] - rightEye[0];
  const eyeDy = leftEye[1] - rightEye[1];
  const roll = Math.atan2(eyeDy, eyeDx) * (180 / Math.PI);

  return {
    roll: Math.max(-90, Math.min(90, roll)),
    pitch: Math.max(-90, Math.min(90, pitch)),
    yaw: Math.max(-90, Math.min(90, yaw)),
    timestamp: Date.now(),
  };
}

export function hasPoseChanged(
  newPose: HeadPose,
  lastPose: HeadPose | null,
  threshold = 2
): boolean {
  if (!lastPose) return true;

  const rollDiff = Math.abs(newPose.roll - lastPose.roll);
  const pitchDiff = Math.abs(newPose.pitch - lastPose.pitch);
  const yawDiff = Math.abs(newPose.yaw - lastPose.yaw);

  return rollDiff > threshold || pitchDiff > threshold || yawDiff > threshold;
}
