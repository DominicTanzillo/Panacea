import { useRef, useMemo, useCallback } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import type { SatellitePosition } from '../lib/types';
import { TYPE_COLORS, EARTH_RADIUS_KM } from '../lib/types';

interface SatelliteLayerProps {
  satellites: SatellitePosition[];
  onSelect: (sat: SatellitePosition | null) => void;
  selected: SatellitePosition | null;
}

// Pre-parse color map once at module level
const PARSED_COLORS: Record<string, [number, number, number]> = {};
for (const [key, hex] of Object.entries(TYPE_COLORS)) {
  const c = new THREE.Color(hex);
  PARSED_COLORS[key] = [c.r, c.g, c.b];
}
const WHITE_RGB: [number, number, number] = [1, 1, 1];

export function SatelliteLayer({ satellites, onSelect }: SatelliteLayerProps) {
  const pointsRef = useRef<THREE.Points>(null);
  const { raycaster, pointer, camera } = useThree();

  // Build typed arrays for positions and colors from satellite data
  const { positionArray, colorArray } = useMemo(() => {
    const n = satellites.length;
    const positions = new Float32Array(n * 3);
    const colors = new Float32Array(n * 3);
    const scale = 1 / EARTH_RADIUS_KM;

    for (let i = 0; i < n; i++) {
      const sat = satellites[i];
      const i3 = i * 3;

      // ECI km to scene units (Earth radii), Y-up coordinate swap
      positions[i3] = sat.x * scale;
      positions[i3 + 1] = sat.z * scale;   // ECI Z -> scene Y
      positions[i3 + 2] = -sat.y * scale;  // ECI Y -> scene -Z

      const rgb = PARSED_COLORS[sat.type] || WHITE_RGB;
      colors[i3] = rgb[0];
      colors[i3 + 1] = rgb[1];
      colors[i3 + 2] = rgb[2];
    }

    return { positionArray: positions, colorArray: colors, count: n };
  }, [satellites]);

  // Build geometry from typed arrays
  const geometry = useMemo(() => {
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.Float32BufferAttribute(positionArray, 3));
    geo.setAttribute('color', new THREE.Float32BufferAttribute(colorArray, 3));
    return geo;
  }, [positionArray, colorArray]);

  // Click handler — find nearest point to click
  const handleClick = useCallback(() => {
    if (!pointsRef.current) return;

    raycaster.setFromCamera(pointer, camera);
    raycaster.params.Points = { threshold: 0.02 };
    const intersects = raycaster.intersectObject(pointsRef.current);

    if (intersects.length > 0 && intersects[0].index !== undefined) {
      const idx = intersects[0].index;
      if (idx < satellites.length) {
        onSelect(satellites[idx]);
        return;
      }
    }
    onSelect(null);
  }, [satellites, onSelect, raycaster, pointer, camera]);

  return (
    <points
      ref={pointsRef}
      geometry={geometry}
      onClick={handleClick}
      frustumCulled={false}
    >
      <pointsMaterial
        size={2.5}
        vertexColors
        sizeAttenuation={false}
        transparent
        opacity={0.9}
        depthWrite={false}
        toneMapped={false}
      />
    </points>
  );
}

// Ring around selected satellite
export function SelectionRing({ satellite }: { satellite: SatellitePosition }) {
  const meshRef = useRef<THREE.Mesh>(null);

  const position = useMemo((): [number, number, number] => {
    const scale = 1 / EARTH_RADIUS_KM;
    return [
      satellite.x * scale,
      satellite.z * scale,
      -satellite.y * scale,
    ];
  }, [satellite]);

  useFrame((_, delta) => {
    if (meshRef.current) {
      meshRef.current.rotation.z += delta * 2;
    }
  });

  return (
    <mesh ref={meshRef} position={position}>
      <ringGeometry args={[0.02, 0.025, 32]} />
      <meshBasicMaterial color="#4f8aff" side={THREE.DoubleSide} transparent opacity={0.8} />
    </mesh>
  );
}
