import { useState, useMemo, useEffect, useCallback } from 'react';
import * as satellite from 'satellite.js';
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ReferenceLine, ResponsiveContainer,
} from 'recharts';
import type { ScreeningPair, TrajectoryPoint } from '../lib/api';
import type { TLERecord, ProjectedPair } from '../lib/types';

interface AlertDetailProps {
  pair: ScreeningPair;
  tles: TLERecord[];
  onBack: () => void;
  onProjection: (pair: ProjectedPair | null) => void;
}

interface ChartPoint {
  hourFromTCA: number;
  hourAbs: number;
  distance: number;
  sat1: { x: number; y: number; z: number };
  sat2: { x: number; y: number; z: number };
}

const HOURS_FORWARD = 120;
const STEP_MINUTES = 20;

function createSatrec(tle: TLERecord): satellite.SatRec | null {
  try {
    const satrec = satellite.json2satrec(tle as unknown as satellite.OMMJsonObject);
    const test = satellite.propagate(satrec, new Date()) as { position: satellite.EciVec3<number> | false | null };
    if (!test || !test.position || typeof test.position === 'boolean') return null;
    return satrec;
  } catch { return null; }
}

function trajectoryToChartPoints(trajectory: TrajectoryPoint[]): ChartPoint[] {
  if (trajectory.length === 0) return [];
  let tcaIdx = 0;
  for (let i = 1; i < trajectory.length; i++) {
    if (trajectory[i].d < trajectory[tcaIdx].d) tcaIdx = i;
  }
  const tcaHour = trajectory[tcaIdx].h;
  return trajectory.map(p => ({
    hourFromTCA: p.h - tcaHour, hourAbs: p.h,
    distance: Math.round(p.d * 10) / 10,
    sat1: { x: p.s1[0], y: p.s1[1], z: p.s1[2] },
    sat2: { x: p.s2[0], y: p.s2[1], z: p.s2[2] },
  }));
}

function propagateTrajectory(tle1: TLERecord, tle2: TLERecord): ChartPoint[] {
  const satrec1 = createSatrec(tle1);
  const satrec2 = createSatrec(tle2);
  if (!satrec1 || !satrec2) return [];
  const now = new Date();
  const rawPoints: { hourAbs: number; distance: number; s1: satellite.EciVec3<number>; s2: satellite.EciVec3<number> }[] = [];
  const nSteps = Math.floor(HOURS_FORWARD * 60 / STEP_MINUTES) + 1;
  for (let i = 0; i < nSteps; i++) {
    const futureDate = new Date(now.getTime() + i * STEP_MINUTES * 60_000);
    const pv1 = satellite.propagate(satrec1, futureDate) as { position: satellite.EciVec3<number> | false | null } | null;
    const pv2 = satellite.propagate(satrec2, futureDate) as { position: satellite.EciVec3<number> | false | null } | null;
    if (!pv1?.position || typeof pv1.position === 'boolean' || !pv2?.position || typeof pv2.position === 'boolean') continue;
    const p1 = pv1.position as satellite.EciVec3<number>;
    const p2 = pv2.position as satellite.EciVec3<number>;
    if (!isFinite(p1.x) || !isFinite(p2.x)) continue;
    const dist = Math.sqrt((p1.x-p2.x)**2 + (p1.y-p2.y)**2 + (p1.z-p2.z)**2);
    if (!isFinite(dist)) continue;
    rawPoints.push({ hourAbs: i * STEP_MINUTES / 60, distance: dist, s1: p1, s2: p2 });
  }
  if (rawPoints.length === 0) return [];
  let tcaIdx = 0;
  for (let i = 1; i < rawPoints.length; i++) { if (rawPoints[i].distance < rawPoints[tcaIdx].distance) tcaIdx = i; }
  const tcaHour = rawPoints[tcaIdx].hourAbs;
  return rawPoints.map(p => ({
    hourFromTCA: p.hourAbs - tcaHour, hourAbs: p.hourAbs,
    distance: Math.round(p.distance * 10) / 10,
    sat1: { x: p.s1.x, y: p.s1.y, z: p.s1.z },
    sat2: { x: p.s2.x, y: p.s2.y, z: p.s2.z },
  }));
}

import { riskTier } from './ConjunctionAlerts';

const TIER_COLORS: Record<string, string> = { HIGH: '#ef4444', MODERATE: '#f59e0b', LOW: '#22c55e' };
const OBJ_TYPE_LABELS: Record<string, string> = { PAYLOAD: 'Payload', ROCKET_BODY: 'Rocket Body', DEBRIS: 'Debris', UNKNOWN: 'Unknown' };

function formatPcDisplay(pc: number): string {
  if (!pc || pc <= 0) return '--';
  if (pc >= 0.01) return `${(pc * 100).toFixed(1)}%`;
  if (pc >= 1e-4) return `1 in ${Math.round(1 / pc).toLocaleString()}`;
  return pc.toExponential(1);
}

export function AlertDetail({ pair, tles, onBack, onProjection }: AlertDetailProps) {
  const hasPrecomputed = !!(pair.trajectory && pair.trajectory.length > 0);
  const hasTrail = !!(pair.trail && pair.trail.length > 0);

  const tle1 = useMemo(() => {
    if (hasPrecomputed) return null;
    return tles.find(t => t.NORAD_CAT_ID === pair.norad_1) || (pair.tle1 as TLERecord | undefined) || null;
  }, [tles, pair, hasPrecomputed]);
  const tle2 = useMemo(() => {
    if (hasPrecomputed) return null;
    return tles.find(t => t.NORAD_CAT_ID === pair.norad_2) || (pair.tle2 as TLERecord | undefined) || null;
  }, [tles, pair, hasPrecomputed]);

  const chartPoints = useMemo(() => {
    if (hasPrecomputed) return trajectoryToChartPoints(pair.trajectory!);
    if (!tle1 || !tle2) return [];
    return propagateTrajectory(tle1, tle2);
  }, [hasPrecomputed, pair.trajectory, tle1, tle2]);

  const simPoints = useMemo((): ChartPoint[] => {
    if (hasTrail) {
      const trail = pair.trail!;
      let tcaI = 0;
      for (let i = 1; i < trail.length; i++) { if (trail[i].d < trail[tcaI].d) tcaI = i; }
      const tcaH = trail[tcaI].h;
      return trail.map(p => ({
        hourFromTCA: p.h - tcaH, hourAbs: p.h,
        distance: Math.round(p.d * 10) / 10,
        sat1: { x: p.s1[0], y: p.s1[1], z: p.s1[2] },
        sat2: { x: p.s2[0], y: p.s2[1], z: p.s2[2] },
      }));
    }
    return chartPoints;
  }, [hasTrail, pair.trail, chartPoints]);

  const simTcaIdx = useMemo(() => {
    if (simPoints.length === 0) return 0;
    let idx = 0;
    for (let i = 1; i < simPoints.length; i++) { if (Math.abs(simPoints[i].hourFromTCA) < Math.abs(simPoints[idx].hourFromTCA)) idx = i; }
    return idx;
  }, [simPoints]);

  const [sliderIdx, setSliderIdx] = useState(0);
  const [autoPlay, setAutoPlay] = useState(false);

  useEffect(() => { if (simPoints.length > 0) setSliderIdx(simTcaIdx); }, [simTcaIdx, simPoints.length]);
  useEffect(() => {
    if (!autoPlay || simPoints.length === 0) return;
    const timer = setInterval(() => {
      setSliderIdx(prev => { const next = prev + 1; if (next >= simPoints.length) { setAutoPlay(false); return prev; } return next; });
    }, 60);
    return () => clearInterval(timer);
  }, [autoPlay, simPoints.length]);

  const currentPoint = simPoints[sliderIdx] || null;

  const trailData = useMemo(() => {
    if (!pair.trail || pair.trail.length === 0) return null;
    const sat1Trail: [number, number, number][] = [];
    const sat2Trail: [number, number, number][] = [];
    for (const pt of pair.trail) { sat1Trail.push([pt.s1[0], pt.s1[1], pt.s1[2]]); sat2Trail.push([pt.s2[0], pt.s2[1], pt.s2[2]]); }
    return { sat1Trail, sat2Trail };
  }, [pair.trail]);

  const tcaPositions = useMemo(() => {
    if (simPoints.length === 0) return null;
    const tca = simPoints[simTcaIdx];
    return { sat1AtTCA: tca.sat1, sat2AtTCA: tca.sat2 };
  }, [simPoints, simTcaIdx]);

  const emitProjection = useCallback((point: ChartPoint | null) => {
    if (!point) { onProjection(null); return; }
    onProjection({
      sat1: { ...point.sat1, name: pair.name_1, noradId: pair.norad_1 },
      sat2: { ...point.sat2, name: pair.name_2, noradId: pair.norad_2 },
      distanceKm: point.distance,
      missDistanceKm: pair.miss_distance_km ?? undefined,
      ...(trailData || {}), ...(tcaPositions || {}),
    });
  }, [onProjection, pair, trailData, tcaPositions]);

  useEffect(() => { emitProjection(currentPoint); }, [currentPoint, emitProjection]);
  useEffect(() => { return () => onProjection(null); }, [onProjection]);

  const handleBack = useCallback(() => { onProjection(null); onBack(); }, [onBack, onProjection]);

  const tcaPoint = simPoints.length > 0 ? simPoints[simTcaIdx] : null;
  const tier = riskTier(pair);
  const color = TIER_COLORS[tier];
  const hasData = chartPoints.length > 0 || simPoints.length > 0;
  // No trajectory = show CDM-only view (not "loading forever")
  const noTrajectory = !hasPrecomputed && !hasTrail && !tle1 && !tle2;

  const isCDM = pair.source === 'cdm';
  const tcaDate = pair.cdm_tca ? new Date(pair.cdm_tca) : null;
  const hoursToTCA = tcaDate ? (tcaDate.getTime() - Date.now()) / 3600000 : null;
  const isPast = hoursToTCA != null && hoursToTCA < -1;

  return (
    <div style={{
      position: 'absolute', right: 24, top: 72, width: 420,
      maxHeight: 'calc(100vh - 96px)',
      background: 'rgba(12,12,18,0.95)', backdropFilter: 'blur(16px)',
      border: '1px solid #1e1e2c', borderRadius: 12,
      boxShadow: '0 16px 48px rgba(0,0,0,0.5)',
      zIndex: 20, display: 'flex', flexDirection: 'column',
      overflow: 'hidden',
      animation: 'overlayIn 200ms ease-out',
    }}>
      {/* Header */}
      <div style={{ padding: '16px 20px', borderBottom: '1px solid #1e1e2c', flexShrink: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 12 }}>
          <button onClick={handleBack} style={{
            width: 28, height: 28, display: 'flex', alignItems: 'center', justifyContent: 'center',
            borderRadius: 6, border: 'none', background: '#1a1a24', color: '#7c7c96', cursor: 'pointer', fontFamily: 'inherit', fontSize: 14,
          }}>&larr;</button>
          <span style={{ flex: 1, fontSize: 14, fontWeight: 600, color: '#e8e8f0' }}>Conjunction Detail</span>
          <span style={{ fontSize: 12, fontWeight: 700, color, background: `${color}12`, padding: '2px 10px', borderRadius: 6 }}>{tier}</span>
        </div>

        {/* Pair names */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginBottom: 12 }}>
          <div style={{ padding: '8px 10px', background: '#111118', borderRadius: 6, borderLeft: '3px solid #3b82f6' }}>
            <div style={{ fontSize: 12, color: '#55556a', marginBottom: 2 }}>Object 1</div>
            <div style={{ fontSize: 13, fontWeight: 500, color: '#e8e8f0', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{pair.name_1}</div>
            <div style={{ fontSize: 12, fontFamily: "'JetBrains Mono', monospace", color: '#55556a' }}>{pair.norad_1}</div>
          </div>
          <div style={{ padding: '8px 10px', background: '#111118', borderRadius: 6, borderLeft: '3px solid #f59e0b' }}>
            <div style={{ fontSize: 12, color: '#55556a', marginBottom: 2 }}>Object 2</div>
            <div style={{ fontSize: 13, fontWeight: 500, color: '#e8e8f0', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{pair.name_2}</div>
            <div style={{ fontSize: 12, fontFamily: "'JetBrains Mono', monospace", color: '#55556a' }}>{pair.norad_2}</div>
          </div>
        </div>

        {/* Key metrics row */}
        <div style={{ display: 'flex', gap: 16, fontSize: 13 }}>
          {isCDM && pair.pc != null && (
            <div><span style={{ color: '#55556a' }}>Pc </span><span style={{ fontWeight: 600, fontFamily: "'JetBrains Mono', monospace", color }}>{formatPcDisplay(pair.pc)}</span></div>
          )}
          {pair.miss_distance_km != null && (
            <div><span style={{ color: '#55556a' }}>Miss </span><span style={{ fontWeight: 600, fontFamily: "'JetBrains Mono', monospace", color: '#e8e8f0' }}>{pair.miss_distance_km.toFixed(1)} km</span></div>
          )}
          {pair.altitude_km > 0 && (
            <div><span style={{ color: '#55556a' }}>Alt </span><span style={{ color: '#e8e8f0' }}>{pair.altitude_km.toFixed(0)} km</span></div>
          )}
          {hoursToTCA != null && (
            <div style={{ marginLeft: 'auto' }}><span style={{ color: '#55556a' }}>TCA </span><span style={{ color: '#e8e8f0' }}>
              {isPast ? `${Math.abs(hoursToTCA) < 24 ? `${Math.abs(hoursToTCA).toFixed(0)}h ago` : `${Math.abs(hoursToTCA / 24).toFixed(0)}d ago`}` : hoursToTCA < 24 ? `in ${hoursToTCA.toFixed(1)}h` : `in ${(hoursToTCA / 24).toFixed(1)}d`}
            </span></div>
          )}
        </div>

        {/* CDM type badges */}
        {isCDM && (
          <div style={{ display: 'flex', gap: 6, marginTop: 8, flexWrap: 'wrap', fontSize: 12 }}>
            <span style={{ color: '#3b82f6', background: 'rgba(59,130,246,0.08)', padding: '2px 8px', borderRadius: 4, fontWeight: 600 }}>CDM</span>
            {pair.sat1_type && <span style={{ color: '#55556a', background: '#111118', padding: '2px 8px', borderRadius: 4 }}>{OBJ_TYPE_LABELS[pair.sat1_type.toUpperCase()] || pair.sat1_type}</span>}
            {pair.sat2_type && <span style={{ color: '#55556a', background: '#111118', padding: '2px 8px', borderRadius: 4 }}>{OBJ_TYPE_LABELS[pair.sat2_type.toUpperCase()] || pair.sat2_type}</span>}
            {pair.emergency_reportable === 'Y' && <span style={{ color: '#ef4444', background: 'rgba(239,68,68,0.08)', padding: '2px 8px', borderRadius: 4, fontWeight: 700 }}>EMERGENCY</span>}
          </div>
        )}
      </div>

      {/* Body */}
      <div style={{ flex: 1, overflow: 'auto', padding: '16px 20px' }}>
        {hasData ? (
          <>
            {/* Chart */}
            <div style={{ marginBottom: 16 }}>
              <div style={{ fontSize: 12, color: '#55556a', marginBottom: 8 }}>
                {isCDM ? 'Separation Distance (SGP4 approximate — CDM miss distance is authoritative)' : 'Separation Distance'}
              </div>
              <div style={{ height: 180 }}>
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={chartPoints} margin={{ top: 4, right: 4, left: -10, bottom: 0 }}>
                    <defs>
                      <linearGradient id="distGrad" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.2} />
                        <stop offset="100%" stopColor="#3b82f6" stopOpacity={0.02} />
                      </linearGradient>
                    </defs>
                    <XAxis dataKey="hourFromTCA" tick={{ fontSize: 12, fill: '#55556a' }} tickFormatter={(h: number) => { const s = h < 0 ? '-' : '+'; const a = Math.abs(h); return a < 24 ? `${s}${a.toFixed(0)}h` : `${s}${(a/24).toFixed(0)}d`; }} axisLine={{ stroke: '#1e1e2c' }} tickLine={false} interval="preserveStartEnd" />
                    <YAxis tick={{ fontSize: 12, fill: '#55556a' }} tickFormatter={(v: number) => v >= 1000 ? `${(v/1000).toFixed(0)}k` : `${v.toFixed(0)}`} axisLine={false} tickLine={false} width={45} />
                    <Tooltip contentStyle={{ background: '#111118', border: '1px solid #2a2a3a', borderRadius: 6, fontSize: 12 }} formatter={(v: unknown) => [`${Number(v).toFixed(1)} km`, 'Separation']} labelFormatter={(h: unknown) => { const v = Number(h); return `TCA ${v < 0 ? '' : '+'}${Math.abs(v) < 24 ? v.toFixed(1) + 'h' : (v/24).toFixed(1) + 'd'}`; }} />
                    <ReferenceLine x={0} stroke="#ef4444" strokeDasharray="4 4" strokeWidth={1} />
                    {currentPoint && <ReferenceLine x={currentPoint.hourFromTCA} stroke="#3b82f6" strokeWidth={1.5} strokeOpacity={0.6} />}
                    <Area type="monotone" dataKey="distance" stroke="#3b82f6" strokeWidth={1.5} fill="url(#distGrad)" dot={false} activeDot={{ r: 3, fill: '#3b82f6' }} />
                  </AreaChart>
                </ResponsiveContainer>
              </div>
            </div>

            {/* Slider */}
            {simPoints.length > 5 && (
              <div style={{ marginBottom: 16 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                  <button onClick={() => { if (!autoPlay && sliderIdx >= simPoints.length - 1) setSliderIdx(0); setAutoPlay(!autoPlay); }}
                    style={{ width: 28, height: 28, display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px solid #2a2a3a', borderRadius: 6, background: 'transparent', color: '#7c7c96', cursor: 'pointer', fontSize: 12, flexShrink: 0 }}>
                    {autoPlay ? '\u23F8' : '\u25B6'}
                  </button>
                  <div style={{ flex: 1 }}>
                    <input type="range" min={0} max={simPoints.length - 1} value={sliderIdx}
                      onChange={e => { setAutoPlay(false); setSliderIdx(Number(e.target.value)); }}
                      style={{ width: '100%', height: 6, appearance: 'none', cursor: 'pointer', borderRadius: 3, background: `linear-gradient(to right, #3b82f6 ${(sliderIdx / Math.max(simPoints.length - 1, 1)) * 100}%, #1a1a24 ${(sliderIdx / Math.max(simPoints.length - 1, 1)) * 100}%)` }} />
                    <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12, color: '#3a3a4a', marginTop: 4 }}>
                      <span>-30 min</span><span style={{ fontWeight: 600, color: '#7c7c96' }}>TCA</span><span>+30 min</span>
                    </div>
                  </div>
                </div>
                {currentPoint && (
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 13, marginTop: 8 }}>
                    <span style={{ color: '#55556a' }}>TCA {currentPoint.hourFromTCA < 0 ? '' : '+'}{Math.abs(currentPoint.hourFromTCA * 60).toFixed(0)} min</span>
                    <span style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 600, color: currentPoint.distance < 10 ? '#ef4444' : currentPoint.distance < 50 ? '#f59e0b' : '#e8e8f0' }}>
                      {currentPoint.distance.toFixed(1)} km
                    </span>
                  </div>
                )}
              </div>
            )}

            {/* TCA summary */}
            {isCDM && pair.miss_distance_km != null && (
              <div style={{ padding: '10px 14px', background: '#111118', borderRadius: 8, marginBottom: 12, fontSize: 13 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
                  <div style={{ width: 6, height: 6, borderRadius: '50%', background: '#ef4444' }} />
                  <span style={{ fontWeight: 600, color: '#e8e8f0' }}>Closest Approach (CDM)</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', color: '#7c7c96' }}>
                  <span>{pair.cdm_tca ? new Date(pair.cdm_tca).toISOString().slice(5, 16).replace('T', ' ') + ' UTC' : ''}</span>
                  <span style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 600, color: '#e8e8f0' }}>{pair.miss_distance_km.toFixed(1)} km</span>
                </div>
                {tcaPoint && Math.abs(tcaPoint.distance - pair.miss_distance_km) > 10 && (
                  <div style={{ fontSize: 12, color: '#3a3a4a', marginTop: 4, fontStyle: 'italic' }}>
                    SGP4 shows {tcaPoint.distance.toFixed(1)} km — TLE propagation is less accurate for debris
                  </div>
                )}
              </div>
            )}

            {/* Recommendation */}
            {tier !== 'LOW' && (
              <div style={{ padding: '10px 14px', background: `${color}06`, border: `1px solid ${color}20`, borderRadius: 8, fontSize: 13 }}>
                <div style={{ fontWeight: 600, color, marginBottom: 4 }}>
                  {isPast ? 'Post-Event Analysis' : 'Recommendation'}
                </div>
                <div style={{ color: '#7c7c96', lineHeight: 1.5 }}>
                  {isPast ? (
                    <>TCA has passed. {isCDM && pair.miss_distance_km != null ? `Objects passed within ${pair.miss_distance_km.toFixed(1)} km.` : 'Event is no longer actionable.'}{' '}
                    {tier === 'HIGH' ? 'Check for debris generation in follow-up TLEs.' : 'Monitor for updated CDMs.'}</>
                  ) : tier === 'HIGH' ? (
                    <>Collision avoidance maneuver recommended. Pc of {pair.pc != null ? pair.pc.toExponential(1) : '>1e-4'} exceeds the 1e-4 threshold.
                    {hoursToTCA != null && hoursToTCA > 0 && <> Time to TCA: <strong style={{ color: '#e8e8f0' }}>{hoursToTCA < 24 ? `${hoursToTCA.toFixed(1)} hours` : `${(hoursToTCA / 24).toFixed(1)} days`}</strong>.</>}</>
                  ) : (
                    <>Monitor conjunction. {isCDM ? 'Request updated CDM to refine Pc estimate.' : 'Refine with higher-fidelity orbit determination.'}</>
                  )}
                </div>
              </div>
            )}
          </>
        ) : (
          /* No trajectory data — show CDM data only */
          <div style={{ textAlign: 'center', padding: '32px 16px' }}>
            {noTrajectory ? (
              <>
                <div style={{ fontSize: 13, color: '#55556a', lineHeight: 1.6, marginBottom: 16 }}>
                  No orbital elements available for this pair. Objects may not be in the CelesTrak catalog (common for small debris and rocket bodies).
                </div>
                {isCDM && pair.miss_distance_km != null && (
                  <div style={{ padding: '12px 16px', background: '#111118', borderRadius: 8, textAlign: 'left', fontSize: 13 }}>
                    <div style={{ fontWeight: 600, color: '#e8e8f0', marginBottom: 6 }}>CDM Data (authoritative)</div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', color: '#7c7c96', marginBottom: 4 }}>
                      <span>Collision Probability</span>
                      <span style={{ fontFamily: "'JetBrains Mono', monospace", fontWeight: 600, color }}>{formatPcDisplay(pair.pc ?? 0)}</span>
                    </div>
                    <div style={{ display: 'flex', justifyContent: 'space-between', color: '#7c7c96', marginBottom: 4 }}>
                      <span>Miss Distance</span>
                      <span style={{ fontFamily: "'JetBrains Mono', monospace", color: '#e8e8f0' }}>{pair.miss_distance_km.toFixed(1)} km</span>
                    </div>
                    {pair.cdm_tca && (
                      <div style={{ display: 'flex', justifyContent: 'space-between', color: '#7c7c96' }}>
                        <span>TCA</span>
                        <span style={{ fontFamily: "'JetBrains Mono', monospace", color: '#e8e8f0' }}>{new Date(pair.cdm_tca).toISOString().slice(0, 16).replace('T', ' ')} UTC</span>
                      </div>
                    )}
                  </div>
                )}
              </>
            ) : (
              <div style={{ color: '#55556a', fontSize: 13 }}>
                <div style={{ width: 20, height: 20, border: '2px solid #3b82f6', borderTop: '2px solid transparent', borderRadius: '50%', animation: 'spin 1s linear infinite', margin: '0 auto 12px' }} />
                Computing trajectory...
              </div>
            )}
          </div>
        )}
      </div>

      {/* Footer */}
      <div style={{ padding: '8px 20px', borderTop: '1px solid #1e1e2c', textAlign: 'center', fontSize: 12, color: '#3a3a4a', flexShrink: 0 }}>
        {hasData ? (hasPrecomputed ? 'Pre-computed trajectory' : 'SGP4 propagation') + (isCDM ? ' (approx.)' : '') : 'CDM data only'}
      </div>
    </div>
  );
}
