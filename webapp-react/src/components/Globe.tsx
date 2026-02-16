import { useRef, useMemo, useEffect } from 'react';
import { Canvas, useFrame, useLoader, useThree } from '@react-three/fiber';
import { OrbitControls, Stars } from '@react-three/drei';
import type { OrbitControls as OrbitControlsImpl } from 'three-stdlib';
import * as THREE from 'three';
import * as satellite from 'satellite.js';
import { SatelliteLayer } from './SatelliteLayer';
import { CountryBorders } from './CountryBorders';
import type { SatellitePosition, ProjectedPair } from '../lib/types';
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

// Zoom-out threshold: if camera is farther than this, deselect
const DESELECT_DISTANCE = 6.0;

// Smoothly move camera to look at the selected satellite
// Stops animation on user interaction; deselects on zoom out
function CameraController({
  target,
  pairTarget,
  controlsRef,
  onDeselect,
}: {
  target: SatellitePosition | null;
  pairTarget: ProjectedPair | null;
  controlsRef: React.RefObject<OrbitControlsImpl | null>;
  onDeselect: () => void;
}) {
  const { camera, gl } = useThree();
  const targetPos = useRef<THREE.Vector3 | null>(null);
  const animating = useRef(false);
  const prevTargetId = useRef<number | null>(null);
  const prevDistance = useRef(camera.position.length());
  // Pair mode: track previous satellite position so we can apply rigid deltas
  const prevSatPos = useRef<THREE.Vector3 | null>(null);
  const pairFocused = useRef(false);

  // Cancel animation on any user interaction (mouse drag, scroll, touch)
  useEffect(() => {
    const canvas = gl.domElement;
    const cancelAnimation = () => {
      if (animating.current) {
        animating.current = false;
      }
    };
    canvas.addEventListener('pointerdown', cancelAnimation);
    canvas.addEventListener('wheel', cancelAnimation);
    return () => {
      canvas.removeEventListener('pointerdown', cancelAnimation);
      canvas.removeEventListener('wheel', cancelAnimation);
    };
  }, [gl]);

  // Focus on individual satellite
  useEffect(() => {
    if (pairTarget) return; // pair takes priority
    if (!target) {
      animating.current = false;
      prevTargetId.current = null;
      targetPos.current = null;
      return;
    }
    if (target.noradId === prevTargetId.current) return;
    prevTargetId.current = target.noradId;
    const dt = (Date.now() - target.propagatedAt) / 1000;
    const sx = (target.x + target.vx * dt) * SCALE;
    const sy = (target.z + target.vz * dt) * SCALE;
    const sz = -(target.y + target.vy * dt) * SCALE;
    targetPos.current = new THREE.Vector3(sx, sy, sz);
    animating.current = true;
  }, [target, pairTarget]);

  // Focus on projected pair — satellite-fixated camera
  // The satellite stays centered in the viewport; Earth moves underneath.
  // OrbitControls pivot = satellite position so user can orbit around the satellite.
  // On slider movement: delta-translate camera + pivot together (no lerp, no lag).
  useEffect(() => {
    if (!pairTarget) {
      pairFocused.current = false;
      prevSatPos.current = null;
      if (controlsRef.current) {
        controlsRef.current.target.set(0, 0, 0);
        controlsRef.current.update();
      }
      return;
    }

    const { sat1 } = pairTarget;
    const newPos = new THREE.Vector3(sat1.x * SCALE, sat1.z * SCALE, -sat1.y * SCALE);

    if (!pairFocused.current) {
      // First focus: start animation toward the satellite
      targetPos.current = newPos.clone();
      animating.current = true;
      pairFocused.current = true;
    } else if (animating.current) {
      // Still animating initial focus — update the target to track latest position
      targetPos.current = newPos.clone();
    } else if (prevSatPos.current) {
      // Animation done, slider moved: rigid delta translate
      // This keeps the satellite exactly centered and Earth moves in the background
      const delta = newPos.clone().sub(prevSatPos.current);
      camera.position.add(delta);
      if (controlsRef.current) {
        controlsRef.current.target.add(delta);
        controlsRef.current.update();
      }
    }

    prevSatPos.current = newPos.clone();
  }, [pairTarget, controlsRef, camera]);

  useFrame(() => {
    const currentDist = camera.position.length();

    // Deselect when user zooms out past threshold (only for individual, not pair)
    if (target && !pairTarget && !animating.current) {
      if (currentDist > DESELECT_DISTANCE && prevDistance.current <= DESELECT_DISTANCE) {
        onDeselect();
      }
    }
    prevDistance.current = currentDist;

    if (!animating.current || !targetPos.current) return;

    const satPos = targetPos.current;
    const dir = satPos.clone().normalize();
    const desiredCamPos = dir.clone().multiplyScalar(satPos.length() + 0.5);

    camera.position.lerp(desiredCamPos, 0.04);

    // In pair mode: lerp orbit pivot TO the satellite (so it becomes the center)
    if (pairTarget && controlsRef.current) {
      controlsRef.current.target.lerp(satPos, 0.04);
      controlsRef.current.update();
    }

    if (camera.position.distanceTo(desiredCamPos) < 0.01) {
      animating.current = false;
      // Snap controls target exactly to satellite position
      if (pairTarget && controlsRef.current) {
        controlsRef.current.target.copy(satPos);
        controlsRef.current.update();
      }
    }
  });

  return null;
}

/** Convert ECI [x,y,z] km to scene coordinates */
function eciToScene(x: number, y: number, z: number): [number, number, number] {
  return [x * SCALE, z * SCALE, -y * SCALE];
}

/** Build a THREE.BufferGeometry line from ECI trail points */
function buildTrailGeometry(trail: [number, number, number][]): THREE.BufferGeometry {
  const positions = new Float32Array(trail.length * 3);
  for (let i = 0; i < trail.length; i++) {
    const [sx, sy, sz] = eciToScene(trail[i][0], trail[i][1], trail[i][2]);
    positions[i * 3] = sx;
    positions[i * 3 + 1] = sy;
    positions[i * 3 + 2] = sz;
  }
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
  geo.computeBoundingSphere();
  return geo;
}

/** Two highlighted rings, orbital trail lines, and TCA line for a conjunction pair */
function PairHighlight({ pair }: { pair: ProjectedPair }) {
  const group1Ref = useRef<THREE.Group>(null);
  const group2Ref = useRef<THREE.Group>(null);
  const ring1Ref = useRef<THREE.Mesh>(null);
  const ring2Ref = useRef<THREE.Mesh>(null);
  const { camera } = useThree();

  // Dynamic connecting line (updated per frame)
  const connectLine = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(new Float32Array(6), 3));
    const mat = new THREE.LineDashedMaterial({ color: '#ffffff', dashSize: 0.02, gapSize: 0.01, transparent: true, opacity: 0.3 });
    const line = new THREE.Line(geo, mat);
    line.frustumCulled = false;
    return line;
  }, []);

  // Static trail line objects (rebuilt when trail data changes)
  const sat1TrailLine = useMemo(() => {
    if (!pair.sat1Trail || pair.sat1Trail.length < 2) return null;
    const geo = buildTrailGeometry(pair.sat1Trail);
    const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: '#4f8aff', transparent: true, opacity: 0.4 }));
    line.frustumCulled = false;
    return line;
  }, [pair.sat1Trail]);

  const sat2TrailLine = useMemo(() => {
    if (!pair.sat2Trail || pair.sat2Trail.length < 2) return null;
    const geo = buildTrailGeometry(pair.sat2Trail);
    const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: '#ff8c42', transparent: true, opacity: 0.4 }));
    line.frustumCulled = false;
    return line;
  }, [pair.sat2Trail]);

  // TCA line object (constant red line at closest approach)
  const tcaLine = useMemo(() => {
    if (!pair.sat1AtTCA || !pair.sat2AtTCA) return null;
    const [s1x, s1y, s1z] = eciToScene(pair.sat1AtTCA.x, pair.sat1AtTCA.y, pair.sat1AtTCA.z);
    const [s2x, s2y, s2z] = eciToScene(pair.sat2AtTCA.x, pair.sat2AtTCA.y, pair.sat2AtTCA.z);
    const positions = new Float32Array([s1x, s1y, s1z, s2x, s2y, s2z]);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
    const line = new THREE.Line(geo, new THREE.LineBasicMaterial({ color: '#ff4f5a', transparent: true, opacity: 0.8 }));
    line.frustumCulled = false;
    return line;
  }, [pair.sat1AtTCA, pair.sat2AtTCA]);

  useFrame(() => {
    const { sat1, sat2 } = pair;
    const [p1x, p1y, p1z] = eciToScene(sat1.x, sat1.y, sat1.z);
    const [p2x, p2y, p2z] = eciToScene(sat2.x, sat2.y, sat2.z);

    // Position and billboard the ring groups
    if (group1Ref.current) {
      group1Ref.current.position.set(p1x, p1y, p1z);
      group1Ref.current.quaternion.copy(camera.quaternion);
    }
    if (group2Ref.current) {
      group2Ref.current.position.set(p2x, p2y, p2z);
      group2Ref.current.quaternion.copy(camera.quaternion);
    }
    if (ring1Ref.current) ring1Ref.current.rotation.z += 0.03;
    if (ring2Ref.current) ring2Ref.current.rotation.z -= 0.03;

    // Update dynamic connecting line
    const connectGeo = connectLine.geometry;
    const posArr = connectGeo.getAttribute('position').array as Float32Array;
    posArr[0] = p1x; posArr[1] = p1y; posArr[2] = p1z;
    posArr[3] = p2x; posArr[4] = p2y; posArr[5] = p2z;
    (connectGeo.getAttribute('position') as THREE.BufferAttribute).needsUpdate = true;
    connectLine.computeLineDistances();
  });

  return (
    <group>
      {/* Sat 1 orbital trail — blue line */}
      {sat1TrailLine && <primitive object={sat1TrailLine} />}

      {/* Sat 2 orbital trail — orange line */}
      {sat2TrailLine && <primitive object={sat2TrailLine} />}

      {/* TCA line — constant red solid line at closest approach */}
      {tcaLine && <primitive object={tcaLine} />}

      {/* Sat 1 — blue ring */}
      <group ref={group1Ref}>
        <mesh ref={ring1Ref}>
          <ringGeometry args={[0.02, 0.028, 32]} />
          <meshBasicMaterial color="#4f8aff" side={THREE.DoubleSide} transparent opacity={0.9} />
        </mesh>
        <mesh>
          <circleGeometry args={[0.008, 16]} />
          <meshBasicMaterial color="#4f8aff" transparent opacity={0.6} />
        </mesh>
      </group>

      {/* Sat 2 — orange ring */}
      <group ref={group2Ref}>
        <mesh ref={ring2Ref}>
          <ringGeometry args={[0.02, 0.028, 32]} />
          <meshBasicMaterial color="#ff8c42" side={THREE.DoubleSide} transparent opacity={0.9} />
        </mesh>
        <mesh>
          <circleGeometry args={[0.008, 16]} />
          <meshBasicMaterial color="#ff8c42" transparent opacity={0.6} />
        </mesh>
      </group>

      {/* Current connecting line — white dashed */}
      <primitive object={connectLine} />
    </group>
  );
}

interface SceneProps {
  satellites: SatellitePosition[];
  onSelectSatellite: (sat: SatellitePosition | null) => void;
  selectedSatellite: SatellitePosition | null;
  projectedPair: ProjectedPair | null;
  showBorders: boolean;
}

function Scene({ satellites, onSelectSatellite, selectedSatellite, projectedPair, showBorders }: SceneProps) {
  const controlsRef = useRef<OrbitControlsImpl>(null);

  return (
    <>
      <ambientLight intensity={0.15} />
      <directionalLight position={[5, 3, 5]} intensity={1.5} />
      <pointLight position={[-10, -5, -10]} intensity={0.3} color="#4f8aff" />

      <Earth />
      <Atmosphere />
      <CountryBorders visible={showBorders} />
      <CameraController
        target={selectedSatellite}
        pairTarget={projectedPair}
        controlsRef={controlsRef}
        onDeselect={() => onSelectSatellite(null)}
      />
      {/* Hide all satellites when viewing a conjunction pair */}
      {!projectedPair && satellites.length > 0 && (
        <SatelliteLayer
          satellites={satellites}
          onSelect={onSelectSatellite}
          selected={selectedSatellite}
        />
      )}
      {projectedPair && <PairHighlight pair={projectedPair} />}
      <Stars radius={100} depth={50} count={5000} factor={4} saturation={0} fade speed={1} />

      <OrbitControls
        ref={controlsRef}
        enablePan={false}
        minDistance={0.3}
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
  projectedPair?: ProjectedPair | null;
  showBorders?: boolean;
}

export function Globe({ satellites, onSelectSatellite, selectedSatellite, projectedPair = null, showBorders = false }: GlobeProps) {
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
        projectedPair={projectedPair}
        showBorders={showBorders}
      />
    </Canvas>
  );
}
