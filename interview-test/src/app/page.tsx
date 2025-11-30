"use client";

import dynamic from "next/dynamic";

const InterviewMonitor = dynamic(() => import("../components/interview"), {
  ssr: false,
  loading: () => <div>Loading...</div>,
});

export default InterviewMonitor;
