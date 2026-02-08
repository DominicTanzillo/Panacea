import { useRef, useMemo } from 'react';
import { Canvas, useFrame, useLoader } from '@react-three/fiber';
import { OrbitControls, Stars } from '@react-three/drei';
import * as THREE from 'three';
import { SatelliteLayer } from './SatelliteLayer';
import type { SatellitePosition } from '../lib/types';

const EARTH_TEXTURE_URL = 'https://unpkg.com/three-globe@2.31.0/example/img/earth-blue-marble.jpg';
const EARTH_BUMP_URL = 'https://unpkg.com/three-globe@2.31.0/example/img/earth-topology.png';
const EARTH_SPEC_URL = 'https://unpkg.com/three-globe@2.31.0/example/img/earth-water.png';

function Earth() {
  const meshRef = useRef<THREE.Mesh>(null);
  const [colorMap, bumpMap, specMap] = useLoader(THREE.TextureLoader, [
    EARTH_TEXTURE_URL,
    EARTH_BUMP_URL,
    EARTH_SPEC_URL,
  ]);

  useFrame((_, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.y += delta * 0.02; // Slow rotation
    }
  });

  return (
    <mesh ref={meshRef}>
      <sphereGeometry args={[1, 64, 64]} />
      <meshPhongMaterial
        map={colorMap}
        bumpMap={bumpMap}
        bumpScale={0.02}
        specularMap={specMap}
        specular={new THREE.Color(0x333333)}
        shininess={15}
      />
    </mesh>
  );
}

function Atmosphere() {
  const shaderRef = useRef<THREE.ShaderMaterial>(null);

  const vertexShader = `
    varying vec3 vNormal;
    void main() {
      vNormal = normalize(normalMatrix * normal);
      gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
    }
  `;

  const fragmentShader = `
    varying vec3 vNormal;
    void main() {
      float intensity = pow(0.65 - dot(vNormal, vec3(0.0, 0.0, 1.0)), 2.0);
      gl_FragColor = vec4(0.3, 0.6, 1.0, 1.0) * intensity;
    }
  `;

  return (
    <mesh scale={[1.08, 1.08, 1.08]}>
      <sphereGeometry args={[1, 64, 64]} />
      <shaderMaterial
        ref={shaderRef}
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        blending={THREE.AdditiveBlending}
        side={THREE.BackSide}
        transparent
      />
    </mesh>
  );
}

interface SceneProps {
  satellites: SatellitePosition[];
  onSelectSatellite: (sat: SatellitePosition | null) => void;
  selectedSatellite: SatellitePosition | null;
}

function Scene({ satellites, onSelectSatellite, selectedSatellite }: SceneProps) {
  return (
    <>
      <ambientLight intensity={0.15} />
      <directionalLight position={[5, 3, 5]} intensity={1.5} />
      <pointLight position={[-10, -5, -10]} intensity={0.3} color="#4f8aff" />

      <Earth />
      <Atmosphere />
      <SatelliteLayer
        satellites={satellites}
        onSelect={onSelectSatellite}
        selected={selectedSatellite}
      />
      <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />

      <OrbitControls
        enablePan={false}
        minDistance={1.5}
        maxDistance={15}
        rotateSpeed={0.5}
        zoomSpeed={0.8}
      />
    </>
  );
}

interface GlobeProps {
  satellites: SatellitePosition[];
  onSelectSatellite: (sat: SatellitePosition | null) => void;
  selectedSatellite: SatellitePosition | null;
}

export function Globe({ satellites, onSelectSatellite, selectedSatellite }: GlobeProps) {
  const cameraConfig = useMemo(() => ({
    position: [0, 0, 3.5] as [number, number, number],
    fov: 45,
    near: 0.01,
    far: 200,
  }), []);

  return (
    <Canvas camera={cameraConfig} gl={{ antialias: true, alpha: false }}>
      <color attach="background" args={['#050510']} />
      <fog attach="fog" args={['#050510', 15, 50]} />
      <Scene
        satellites={satellites}
        onSelectSatellite={onSelectSatellite}
        selectedSatellite={selectedSatellite}
      />
    </Canvas>
  );
}
