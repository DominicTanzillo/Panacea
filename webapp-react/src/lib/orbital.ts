import * as satellite from 'satellite.js';
import type { TLERecord, SatellitePosition, ObjectType } from './types';
import { EARTH_RADIUS_KM } from './types';

const MU = 3.986004418e14; // m^3/s^2

export const SCENE_SCALE = 1; // 1 unit = 1 Earth radius

export function meanMotionToAltitude(nRevPerDay: number): number {
  const nRadPerSec = nRevPerDay * 2 * Math.PI / 86400;
  const aMeters = Math.pow(MU / (nRadPerSec * nRadPerSec), 1 / 3);
  return aMeters / 1000 - EARTH_RADIUS_KM;
}

export function propagateSatellite(
  tle: TLERecord,
  date: Date,
  group: string,
  type: ObjectType
): SatellitePosition | null {
  let satrec: satellite.SatRec;
  try {
    satrec = satellite.json2satrec(tle as unknown as satellite.OMMJsonObject);
  } catch {
    return null;
  }

  let posVel: satellite.PositionAndVelocity | null;
  try {
    posVel = satellite.propagate(satrec, date);
  } catch {
    return null;
  }

  if (!posVel || !posVel.position || typeof posVel.position === 'boolean') return null;

  const pos = posVel.position as satellite.EciVec3<number>;
  const gmst = satellite.gstime(date);
  const geo = satellite.eciToGeodetic(pos, gmst);

  const lat = satellite.degreesLat(geo.latitude);
  const lon = satellite.degreesLong(geo.longitude);
  const alt = geo.height;

  return {
    name: tle.OBJECT_NAME,
    noradId: tle.NORAD_CAT_ID,
    objectId: tle.OBJECT_ID,
    x: pos.x,
    y: pos.y,
    z: pos.z,
    lat,
    lon,
    alt,
    type,
    group,
  };
}

export function eciToScene(x: number, y: number, z: number): [number, number, number] {
  const scale = SCENE_SCALE / EARTH_RADIUS_KM;
  return [x * scale, z * scale, -y * scale];
}
