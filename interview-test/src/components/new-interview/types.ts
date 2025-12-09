// src/components/new-interview/types.ts
export interface HeadPose {
  roll: number;
  pitch: number;
  yaw: number;
  timestamp: number;
}

export interface Detection {
  class: string;
  score: number;
  bbox: [number, number, number, number];
}

export interface ViolationAlert {
  type: "phone" | "multiple_faces" | "suspicious_object";
  message: string;
  timestamp: number;
}

export interface CheatingProbability {
  probability: number;
  timestamp: number;
}
