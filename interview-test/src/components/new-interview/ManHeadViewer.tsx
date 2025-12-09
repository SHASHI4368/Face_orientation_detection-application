"use client";

import React, { Suspense } from "react";
import { Canvas } from "@react-three/fiber";
import { OrbitControls, Stage, useGLTF } from "@react-three/drei";

function ManHeadModel() {
  const { scene } = useGLTF("/Man_Head.glb");

  return <primitive object={scene} scale={1} />;
}

export default function ManHeadViewer() {
  return (
    <div className="w-full h-[500px]">
      <Canvas camera={{ position: [0, 0, 3], fov: 45 }} shadows>
        <Suspense
          fallback={
            <mesh>
              <boxGeometry />
              <meshStandardMaterial color="red" />
            </mesh>
          }
        >
          {/* Adds lighting + soft shadows */}
          <Stage
            intensity={0.8}
            environment="city"
            preset={"soft"}
            shadows={false}
            adjustCamera={false}
          >
            <ManHeadModel />
          </Stage>

          {/* Allows rotating the model */}
          <OrbitControls enableZoom={true} />
        </Suspense>
      </Canvas>
    </div>
  );
}

// Fixed: Added the leading slash
useGLTF.preload("/Man_Head.glb");
