import { useRef, useMemo, useEffect } from 'react';
import { Canvas, useFrame, useLoader, useThree } from '@react-three/fiber';
import { OrbitControls, Stars } from '@react-three/drei';
import * as THREE from 'three';
import * as satellite from 'satellite.js';
import { SatelliteLayer } from './SatelliteLayer';
import { CountryBorders } from './CountryBorders';
import type { SatellitePosition } from '../lib/types';
import { EARTH_RADIUS_KM } from '../lib/types';

const SCALE = 1 / EARTH_RADIUS_KM;

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

  // Rotate Earth to match GMST so continents align with ECI satellite positions
  useFrame(() => {
    if (meshRef.current) {
      const gmst = satellite.gstime(new Date());
      meshRef.current.rotation.y = gmst;
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
        vertexShader={vertexShader}
        fragmentShader={fragmentShader}
        blending={THREE.AdditiveBlending}
        side={THREE.BackSide}
        transparent
      />
    </mesh>
  );
}

// Smoothly move camera to look at the selected satellite
function CameraController({ target }: { target: SatellitePosition | null }) {
  const { camera } = useThree();
  const controlsRef = useRef<any>(null);
  const targetPos = useRef<THREE.Vector3 | null>(null);
  const animating = useRef(false);
  const prevTargetId = useRef<number | null>(null);

  useEffect(() => {
    // Store ref to OrbitControls — find it via the canvas parent
    const controls = (camera as any).__orbitControls;
    if (controls) controlsRef.current = controls;
  });

  useEffect(() => {
    if (!target || target.noradId === prevTargetId.current) return;
    prevTargetId.current = target.noradId;
    // Compute satellite scene position
    const dt = (Date.now() - target.propagatedAt) / 1000;
    const sx = (target.x + target.vx * dt) * SCALE;
    const sy = (target.z + target.vz * dt) * SCALE;
    const sz = -(target.y + target.vy * dt) * SCALE;
    targetPos.current = new THREE.Vector3(sx, sy, sz);
    animating.current = true;
  }, [target]);

  useFrame(() => {
    if (!animating.current || !targetPos.current) return;

    // Smoothly move camera to look at the satellite from a closer distance
    const satPos = targetPos.current;
    const dir = satPos.clone().normalize();
    const desiredCamPos = dir.clone().multiplyScalar(satPos.length() + 1.2);

    camera.position.lerp(desiredCamPos, 0.04);

    // After close enough, stop animating
    if (camera.position.distanceTo(desiredCamPos) < 0.01) {
      animating.current = false;
    }
  });

  return null;
}

interface SceneProps {
  satellites: SatellitePosition[];
  onSelectSatellite: (sat: SatellitePosition | null) => void;
  selectedSatellite: SatellitePosition | null;
  showBorders: boolean;
}

function Scene({ satellites, onSelectSatellite, selectedSatellite, showBorders }: SceneProps) {
  return (
    <>
      <ambientLight intensity={0.15} />
      <directionalLight position={[5, 3, 5]} intensity={1.5} />
      <pointLight position={[-10, -5, -10]} intensity={0.3} color="#4f8aff" />

      <Earth />
      <Atmosphere />
      <CountryBorders visible={showBorders} />
      <CameraController target={selectedSatellite} />
      {satellites.length > 0 && (
        <SatelliteLayer
          satellites={satellites}
          onSelect={onSelectSatellite}
          selected={selectedSatellite}
        />
      )}
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
  showBorders?: boolean;
}

export function Globe({ satellites, onSelectSatellite, selectedSatellite, showBorders = false }: GlobeProps) {
  const cameraConfig = useMemo(() => ({
    position: [0, 0, 3.5] as [number, number, number],
    fov: 45,
    near: 0.01,
    far: 200,
  }), []);

  // Log WebGL context loss
  useEffect(() => {
    const canvas = document.querySelector('canvas');
    if (!canvas) return;
    const onLost = (e: Event) => {
      console.error('WebGL CONTEXT LOST', e);
      e.preventDefault();
    };
    const onRestored = () => console.log('WebGL context restored');
    canvas.addEventListener('webglcontextlost', onLost);
    canvas.addEventListener('webglcontextrestored', onRestored);
    return () => {
      canvas.removeEventListener('webglcontextlost', onLost);
      canvas.removeEventListener('webglcontextrestored', onRestored);
    };
  }, []);

  return (
    <Canvas
      camera={cameraConfig}
      gl={{ antialias: false, alpha: false, powerPreference: 'high-performance' }}
      onCreated={({ gl }) => {
        console.log('R3F Canvas created, renderer:', gl.info);
        gl.setPixelRatio(Math.min(window.devicePixelRatio, 2));
      }}
    >
      <color attach="background" args={['#050510']} />
      <fog attach="fog" args={['#050510', 15, 50]} />
      <Scene
        satellites={satellites}
        onSelectSatellite={onSelectSatellite}
        selectedSatellite={selectedSatellite}
        showBorders={showBorders}
      />
    </Canvas>
  );
}
