// src/components/new-interview/utils/exportUtils.ts
import { HeadPose, CheatingProbability } from "../types";

export function exportPoseToCSV(poseHistory: HeadPose[]) {
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

  downloadCSV(csvHeader + csvRows, `head-pose-${Date.now()}.csv`);
}

export function exportCheatingProbabilitiesToCSV(
  cheatingProbabilityList: CheatingProbability[]
) {
  if (cheatingProbabilityList.length === 0) {
    alert("No cheating probability data to export!");
    return;
  }

  const csvHeader = "Timestamp,Date,Cheating Probability\n";
  const csvRows = cheatingProbabilityList
    .map((entry) => {
      const date = new Date(entry.timestamp).toISOString();
      return `${entry.timestamp},${date},${entry.probability.toFixed(4)}`;
    })
    .join("\n");

  downloadCSV(csvHeader + csvRows, `cheating-probabilities-${Date.now()}.csv`);
}

function downloadCSV(csv: string, filename: string) {
  const blob = new Blob([csv], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  document.body.removeChild(a);
  URL.revokeObjectURL(url);
}
