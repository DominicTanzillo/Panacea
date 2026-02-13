export interface TLERecord {
  OBJECT_NAME: string;
  OBJECT_ID: string;
  EPOCH: string;
  MEAN_MOTION: number;
  ECCENTRICITY: number;
  INCLINATION: number;
  RA_OF_ASC_NODE: number;
  ARG_OF_PERICENTER: number;
  MEAN_ANOMALY: number;
  EPHEMERIS_TYPE: number;
  CLASSIFICATION_TYPE: string;
  NORAD_CAT_ID: number;
  ELEMENT_SET_NO: number;
  REV_AT_EPOCH: number;
  BSTAR: number;
  MEAN_MOTION_DOT: number;
  MEAN_MOTION_DDOT: number;
}

export interface SatellitePosition {
  name: string;
  noradId: number;
  objectId: string;
  x: number; // ECI km
  y: number;
  z: number;
  vx: number; // ECI km/s
  vy: number;
  vz: number;
  lat: number;
  lon: number;
  alt: number; // km
  type: ObjectType;
  group: string;
  propagatedAt: number; // Date.now() when this position was computed
}

export type ObjectType = 'active' | 'debris' | 'rocket_body' | 'station' | 'constellation';

export interface ObjectGroup {
  id: string;
  label: string;
  type: ObjectType;
  color: string;
  url: string;
  enabled: boolean;
}

export const OBJECT_GROUPS: ObjectGroup[] = [
  { id: 'stations', label: 'Space Stations', type: 'station', color: '#ffffff', url: 'https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=json', enabled: true },
  { id: 'active', label: 'Active Satellites', type: 'active', color: '#4fff8a', url: 'https://celestrak.org/NORAD/elements/gp.php?GROUP=active&FORMAT=json', enabled: true },
  { id: 'starlink', label: 'Starlink', type: 'constellation', color: '#4f8aff', url: 'https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=json', enabled: true },
  { id: 'cosmos-2251-debris', label: 'Cosmos 2251 Debris', type: 'debris', color: '#ff4f5a', url: 'https://celestrak.org/NORAD/elements/gp.php?GROUP=cosmos-2251-debris&FORMAT=json', enabled: true },
  { id: 'iridium-33-debris', label: 'Iridium 33 Debris', type: 'debris', color: '#ff6b6b', url: 'https://celestrak.org/NORAD/elements/gp.php?GROUP=iridium-33-debris&FORMAT=json', enabled: true },
  { id: 'fengyun-1c-debris', label: 'Fengyun 1C Debris', type: 'debris', color: '#ff8787', url: 'https://celestrak.org/NORAD/elements/gp.php?GROUP=fengyun-1c-debris&FORMAT=json', enabled: true },
];

export const TYPE_COLORS: Record<ObjectType, string> = {
  active: '#4fff8a',
  debris: '#ff4f5a',
  rocket_body: '#ffb84f',
  station: '#ffffff',
  constellation: '#4f8aff',
};

export const EARTH_RADIUS_KM = 6371;
