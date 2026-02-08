import { useRef, useMemo, useCallback, useEffect } from 'react';
import { useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import type { SatellitePosition } from '../lib/types';
import { TYPE_COLORS, EARTH_RADIUS_KM } from '../lib/types';

interface SatelliteLayerProps {
  satellites: SatellitePosition[];
  onSelect: (sat: SatellitePosition | null) => void;
  selected: SatellitePosition | null;
}

const POINT_SIZE = 0.008;

// Pre-parse color map once
const COLOR_MAP: Record<string, THREE.Color> = {};
for (const [key, hex] of Object.entries(TYPE_COLORS)) {
  COLOR_MAP[key] = new THREE.Color(hex);
}
const WHITE = new THREE.Color('#ffffff');

export function SatelliteLayer({ satellites, onSelect }: SatelliteLayerProps) {
  const meshRef = useRef<THREE.InstancedMesh>(null);
  const { raycaster, pointer, camera } = useThree();

  // Reusable objects (allocated once)
  const dummy = useMemo(() => new THREE.Object3D(), []);

  const count = satellites.length;

  // Apply instance data when satellites change (once, not per-frame)
  useEffect(() => {
    const mesh = meshRef.current;
    if (!mesh) return;

    for (let i = 0; i < count; i++) {
      const sat = satellites[i];
      const scale = 1 / EARTH_RADIUS_KM;

      dummy.position.set(
        sat.x * scale,
        sat.z * scale,
        -sat.y * scale
      );
      dummy.scale.setScalar(1);
      dummy.updateMatrix();
      mesh.setMatrixAt(i, dummy.matrix);

      const c = COLOR_MAP[sat.type] || WHITE;
      mesh.setColorAt(i, c);
    }

    mesh.count = count;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }, [satellites, count, dummy]);

  // Click handler
  const handleClick = useCallback(() => {
    if (!meshRef.current) return;

    raycaster.setFromCamera(pointer, camera);
    const intersects = raycaster.intersectObject(meshRef.current);

    if (intersects.length > 0 && intersects[0].instanceId !== undefined) {
      const idx = intersects[0].instanceId;
      if (idx < satellites.length) {
        onSelect(satellites[idx]);
        return;
      }
    }
    onSelect(null);
  }, [satellites, onSelect, raycaster, pointer, camera]);

  // Allocate enough instances — re-create mesh if count grows
  const maxCount = useMemo(() => Math.max(satellites.length, 100), [satellites.length]);

  return (
    <instancedMesh
      ref={meshRef}
      args={[undefined, undefined, maxCount]}
      onClick={handleClick}
      frustumCulled={false}
    >
      <sphereGeometry args={[POINT_SIZE, 6, 6]} />
      <meshBasicMaterial toneMapped={false} />
    </instancedMesh>
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
