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

// Forecast data for cross-referencing alerts with model predictions
interface ForecastMatch {
  exceedance_probability: number;
  ensemble_probability?: number | null;
  forecast_pc: number;
  risk_direction: string;
  n_updates: number;
  time_series: { time_to_tca_hours: number; log10_pc: number; pc: number }[];
  forecast_steps?: { step: number; predicted_log10_pc: number; uncertainty_log10_pc: number; hours_ahead: number }[];
  predicted_max_log10_pc?: number | null;
}

export function AlertDetail({ pair, tles, onBack, onProjection }: AlertDetailProps) {
  const hasPrecomputed = !!(pair.trajectory && pair.trajectory.length > 0);
  const hasTrail = !!(pair.trail && pair.trail.length > 0);
  // Timeout: if trajectory hasn't computed in 5s, give up and show CDM-only view
  const [timedOut, setTimedOut] = useState(false);
  useEffect(() => {
    setTimedOut(false);
    const t = setTimeout(() => setTimedOut(true), 5000);
    return () => clearTimeout(t);
  }, [pair.norad_1, pair.norad_2]);

  // Fetch forecast data to show model predictions when no trajectory available
  const [forecastMatch, setForecastMatch] = useState<ForecastMatch | null>(null);
  useEffect(() => {
    fetch('./cdm_forecast.json')
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (!data?.pairs) return;
        const n1 = pair.norad_1, n2 = pair.norad_2;
        const match = data.pairs.find((p: Record<string, unknown>) =>
          (p.sat1_norad === n1 && p.sat2_norad === n2) ||
          (p.sat1_norad === n2 && p.sat2_norad === n1)
        );
        setForecastMatch(match || null);
      })
      .catch(() => setForecastMatch(null));
  }, [pair.norad_1, pair.norad_2]);

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
  const noTrajectory = (!hasPrecomputed && !hasTrail && !tle1 && !tle2) || timedOut;

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
          <button onClick={handleBack} aria-label="Close detail view" style={{
            width: 28, height: 28, display: 'flex', alignItems: 'center', justifyContent: 'center',
            borderRadius: 6, border: 'none', background: '#1a1a24', color: '#7c7c96', cursor: 'pointer', fontFamily: 'inherit', fontSize: 14,
          }}>&larr;</button>
          <span style={{ flex: 1, fontSize: 14, fontWeight: 600, color: '#e8e8f0' }}>Conjunction Detail</span>
          <span style={{ fontSize: 12, fontWeight: 700, color, background: `${color}12`, padding: '2px 10px', borderRadius: 6 }}>{tier === 'HIGH' ? '\u26A0 ' : tier === 'MODERATE' ? '-- ' : '\u2713 '}{tier}</span>
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
                {isCDM ? 'Separation Distance (SGP4 estimate — orbit prediction algorithm using public data; CDM miss distance is authoritative)' : 'Separation Distance'}
              </div>
              <div style={{ height: 180 }} role="img" aria-label={`Separation distance chart for ${pair.name_1} vs ${pair.name_2}. Shows distance in km relative to time of closest approach.`}>
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
                    aria-label={autoPlay ? 'Pause orbital approach animation' : 'Play orbital approach animation'}
                    style={{ width: 28, height: 28, display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px solid #2a2a3a', borderRadius: 6, background: 'transparent', color: '#7c7c96', cursor: 'pointer', fontSize: 12, flexShrink: 0 }}>
                    {autoPlay ? '\u23F8' : '\u25B6'}
                  </button>
                  <div style={{ flex: 1 }}>
                    <input type="range" min={0} max={simPoints.length - 1} value={sliderIdx}
                      aria-label="Orbital approach timeline slider"
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
                    SGP4 (orbit prediction algorithm using public data) shows {tcaPoint.distance.toFixed(1)} km — TLE propagation is less accurate for debris
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
                    <>Collision avoidance maneuver recommended. The collision probability ({pair.pc != null ? formatPcDisplay(pair.pc) : 'elevated'}) has reached the screening threshold — this event warrants immediate attention and active monitoring.
                    {hoursToTCA != null && hoursToTCA > 0 && <> Time to closest approach: <strong style={{ color: '#e8e8f0' }}>{hoursToTCA < 24 ? `${hoursToTCA.toFixed(1)} hours` : `${(hoursToTCA / 24).toFixed(1)} days`}</strong>.</>}</>
                  ) : (
                    <>Monitor conjunction. {isCDM ? 'Updated data messages may refine the collision probability estimate.' : 'Higher-fidelity orbit data would improve this estimate.'}</>
                  )}
                </div>
              </div>
            )}
          </>
        ) : (
          /* No trajectory — show forecast model data + CDM data */
          <div style={{ padding: '16px' }}>
            {noTrajectory ? (
              <>
                {/* ML forecast chart if available */}
                {forecastMatch && forecastMatch.time_series && forecastMatch.time_series.length >= 2 && (() => {
                  const sorted = forecastMatch.time_series.slice().sort((a, b) => b.time_to_tca_hours - a.time_to_tca_hours);
                  type FcPoint = { hoursToTCA: number; 'log10(Pc)': number | undefined; forecast?: number; forecastUpper?: number; forecastLower?: number };
                  const pts: FcPoint[] = sorted.map(u => ({ hoursToTCA: -Math.round(u.time_to_tca_hours), 'log10(Pc)': u.log10_pc }));
                  // Add forecast steps
                  if (pts.length > 0 && forecastMatch.forecast_steps && forecastMatch.forecast_steps.length > 0) {
                    const last = pts[pts.length - 1];
                    last.forecast = last['log10(Pc)'];
                    const span = Math.abs(last.hoursToTCA);
                    const steps = forecastMatch.forecast_steps;
                    const stepSpan = span / (steps.length + 1);
                    for (let i = 0; i < steps.length; i++) {
                      const fs = steps[i];
                      const h = Math.round(last.hoursToTCA + (i + 1) * stepSpan);
                      pts.push({ hoursToTCA: h, 'log10(Pc)': undefined, forecast: fs.predicted_log10_pc,
                        forecastUpper: fs.predicted_log10_pc + (fs.uncertainty_log10_pc || 0),
                        forecastLower: fs.predicted_log10_pc - (fs.uncertainty_log10_pc || 0) });
                    }
                    const lastFs = steps[steps.length - 1];
                    pts.push({ hoursToTCA: 0, 'log10(Pc)': undefined, forecast: lastFs.predicted_log10_pc,
                      forecastUpper: lastFs.predicted_log10_pc + (lastFs.uncertainty_log10_pc || 0) * 1.2,
                      forecastLower: lastFs.predicted_log10_pc - (lastFs.uncertainty_log10_pc || 0) * 1.2 });
                  } else if (pts.length > 0 && forecastMatch.forecast_pc > 0) {
                    const last = pts[pts.length - 1];
                    last.forecast = last['log10(Pc)'];
                    pts.push({ hoursToTCA: 0, 'log10(Pc)': undefined, forecast: Math.log10(Math.max(forecastMatch.forecast_pc, 1e-20)) });
                  }
                  return (
                    <div style={{ marginBottom: 12 }}>
                      <div style={{ fontSize: 12, color: '#55556a', fontWeight: 600, marginBottom: 6 }}>ML Forecast — Pc Evolution</div>
                      <ResponsiveContainer width="100%" height={160}>
                        <AreaChart data={pts} margin={{ top: 4, right: 8, left: -10, bottom: 4 }}>
                          <XAxis dataKey="hoursToTCA" type="number" domain={['dataMin', 0]} tick={{ fontSize: 11, fill: '#55556a' }} tickFormatter={(v: number) => `${v}h`} axisLine={{ stroke: '#2a2a3a' }} tickLine={false} />
                          <YAxis tick={{ fontSize: 11, fill: '#55556a' }} domain={['auto', 'auto']} tickFormatter={(v: number) => v.toFixed(1)} axisLine={false} tickLine={false} />
                          <Tooltip contentStyle={{ background: '#111118', border: '1px solid #2a2a3a', borderRadius: 6, fontSize: 12 }} labelFormatter={(v) => `TCA ${v}h`} />
                          <ReferenceLine y={-3.3} stroke="#ef4444" strokeDasharray="4 3" strokeWidth={1} strokeOpacity={0.5} />
                          <Area type="monotone" dataKey="log10(Pc)" stroke={color} strokeWidth={2} fill={`${color}15`} dot={{ r: 2, fill: color }} connectNulls={false} />
                          <Area type="monotone" dataKey="forecastUpper" stroke="none" fill={color} fillOpacity={0.08} connectNulls={false} />
                          <Area type="monotone" dataKey="forecastLower" stroke="none" fill="#0c0c12" fillOpacity={0.9} connectNulls={false} />
                          <Area type="monotone" dataKey="forecast" stroke={color} strokeWidth={2} strokeDasharray="4 3" fill="none" dot={{ r: 3, fill: color, stroke: '#111118', strokeWidth: 2 }} connectNulls={false} />
                        </AreaChart>
                      </ResponsiveContainer>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 6, marginTop: 8, fontSize: 12 }}>
                        <div style={{ color: '#7c7c96' }}>P(Exceed) <span style={{ color: '#e8e8f0', fontWeight: 600 }}>{((forecastMatch.ensemble_probability ?? forecastMatch.exceedance_probability) * 100).toFixed(1)}%</span></div>
                        <div style={{ color: '#7c7c96' }}>Trend <span style={{ color: forecastMatch.risk_direction === 'escalating' ? '#ef4444' : forecastMatch.risk_direction === 'de-escalating' ? '#22c55e' : '#7c7c96', fontWeight: 600 }}>{forecastMatch.risk_direction}</span></div>
                        <div style={{ color: '#7c7c96' }}>CDM Updates <span style={{ color: '#e8e8f0' }}>{forecastMatch.n_updates}</span></div>
                        {forecastMatch.predicted_max_log10_pc != null && (
                          <div style={{ color: '#7c7c96' }}>Pred Max <span style={{ color: '#f59e0b' }}>10^{forecastMatch.predicted_max_log10_pc.toFixed(1)}</span></div>
                        )}
                      </div>
                    </div>
                  );
                })()}

                {/* CDM data section */}
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

                {!forecastMatch && (
                  <div style={{ fontSize: 12, color: '#3a3a4a', marginTop: 8 }}>
                    No orbital elements or forecast data available for this pair.
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
        {hasData ? (hasPrecomputed ? 'Pre-computed trajectory' : 'SGP4 propagation (orbit prediction algorithm using public data)') + (isCDM ? ' — approximate' : '') : 'CDM data only'}
      </div>
    </div>
  );
}
