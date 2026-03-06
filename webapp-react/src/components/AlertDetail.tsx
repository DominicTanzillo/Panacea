import { useState, useMemo, useEffect, useCallback } from 'react';
import * as satellite from 'satellite.js';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
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
  /** Hours relative to TCA (negative = before, positive = after) */
  hourFromTCA: number;
  /** Absolute hour from now */
  hourAbs: number;
  distance: number;
  /** ECI positions for globe projection */
  sat1: { x: number; y: number; z: number };
  sat2: { x: number; y: number; z: number };
}

const HOURS_FORWARD = 120; // 5 days
const STEP_MINUTES = 20;   // every 20 min

function createSatrec(tle: TLERecord): satellite.SatRec | null {
  try {
    const satrec = satellite.json2satrec(tle as unknown as satellite.OMMJsonObject);
    const test = satellite.propagate(satrec, new Date()) as { position: satellite.EciVec3<number> | false | null };
    if (!test || !test.position || typeof test.position === 'boolean') {
      console.warn(`[AlertDetail] Satrec for ${tle.OBJECT_NAME} (${tle.NORAD_CAT_ID}) failed test propagation`);
      return null;
    }
    return satrec;
  } catch (e) {
    console.warn(`[AlertDetail] json2satrec failed for ${tle.OBJECT_NAME}:`, e);
    return null;
  }
}

/** Convert pre-computed trajectory data to chart points */
function trajectoryToChartPoints(trajectory: TrajectoryPoint[]): ChartPoint[] {
  if (trajectory.length === 0) return [];

  // Find TCA (minimum distance)
  let tcaIdx = 0;
  for (let i = 1; i < trajectory.length; i++) {
    if (trajectory[i].d < trajectory[tcaIdx].d) tcaIdx = i;
  }
  const tcaHour = trajectory[tcaIdx].h;

  return trajectory.map(p => ({
    hourFromTCA: p.h - tcaHour,
    hourAbs: p.h,
    distance: Math.round(p.d * 10) / 10,
    sat1: { x: p.s1[0], y: p.s1[1], z: p.s1[2] },
    sat2: { x: p.s2[0], y: p.s2[1], z: p.s2[2] },
  }));
}

/** Fallback: propagate trajectory from TLEs using satellite.js */
function propagateTrajectory(tle1: TLERecord, tle2: TLERecord): ChartPoint[] {
  const satrec1 = createSatrec(tle1);
  const satrec2 = createSatrec(tle2);

  if (!satrec1 || !satrec2) {
    console.warn('[AlertDetail] Failed to create satrecs', { tle1: tle1.OBJECT_NAME, tle2: tle2.OBJECT_NAME });
    return [];
  }

  const now = new Date();
  const rawPoints: { hourAbs: number; distance: number; s1: satellite.EciVec3<number>; s2: satellite.EciVec3<number> }[] = [];
  const nSteps = Math.floor(HOURS_FORWARD * 60 / STEP_MINUTES) + 1;

  for (let i = 0; i < nSteps; i++) {
    const futureDate = new Date(now.getTime() + i * STEP_MINUTES * 60_000);

    const pv1 = satellite.propagate(satrec1, futureDate) as { position: satellite.EciVec3<number> | false | null } | null;
    const pv2 = satellite.propagate(satrec2, futureDate) as { position: satellite.EciVec3<number> | false | null } | null;

    if (
      !pv1 || !pv1.position || typeof pv1.position === 'boolean' ||
      !pv2 || !pv2.position || typeof pv2.position === 'boolean'
    ) continue;

    const p1 = pv1.position as satellite.EciVec3<number>;
    const p2 = pv2.position as satellite.EciVec3<number>;

    if (!isFinite(p1.x) || !isFinite(p2.x)) continue;

    const dx = p1.x - p2.x;
    const dy = p1.y - p2.y;
    const dz = p1.z - p2.z;
    const dist = Math.sqrt(dx * dx + dy * dy + dz * dz);

    if (!isFinite(dist)) continue;

    rawPoints.push({
      hourAbs: i * STEP_MINUTES / 60,
      distance: dist,
      s1: p1,
      s2: p2,
    });
  }

  if (rawPoints.length === 0) {
    console.warn('[AlertDetail] No valid propagation points');
    return [];
  }

  let tcaIdx = 0;
  for (let i = 1; i < rawPoints.length; i++) {
    if (rawPoints[i].distance < rawPoints[tcaIdx].distance) tcaIdx = i;
  }
  const tcaHour = rawPoints[tcaIdx].hourAbs;

  return rawPoints.map(p => ({
    hourFromTCA: p.hourAbs - tcaHour,
    hourAbs: p.hourAbs,
    distance: Math.round(p.distance * 10) / 10,
    sat1: { x: p.s1.x, y: p.s1.y, z: p.s1.z },
    sat2: { x: p.s2.x, y: p.s2.y, z: p.s2.z },
  }));
}

import { riskTier } from './ConjunctionAlerts';

const TIER_COLORS: Record<string, string> = {
  HIGH: '#ff4f5a',
  MODERATE: '#ffb84f',
  LOW: '#4fff8a',
};

const OBJ_TYPE_LABELS: Record<string, string> = {
  PAYLOAD: 'Payload',
  ROCKET_BODY: 'Rocket Body',
  DEBRIS: 'Debris',
  UNKNOWN: 'Unknown',
};

export function AlertDetail({ pair, tles, onBack, onProjection }: AlertDetailProps) {
  // Use pre-computed trajectory if available, fall back to TLE propagation
  const hasPrecomputed = !!(pair.trajectory && pair.trajectory.length > 0);
  const hasTrail = !!(pair.trail && pair.trail.length > 0);

  const tle1 = useMemo(() => hasPrecomputed ? null : tles.find(t => t.NORAD_CAT_ID === pair.norad_1), [tles, pair.norad_1, hasPrecomputed]);
  const tle2 = useMemo(() => hasPrecomputed ? null : tles.find(t => t.NORAD_CAT_ID === pair.norad_2), [tles, pair.norad_2, hasPrecomputed]);

  // Sparse chart data (20-min steps, full 5-day window) — for the distance chart
  const chartPoints = useMemo(() => {
    if (hasPrecomputed) {
      return trajectoryToChartPoints(pair.trajectory!);
    }
    if (!tle1 || !tle2) return [];
    return propagateTrajectory(tle1, tle2);
  }, [hasPrecomputed, pair.trajectory, tle1, tle2]);

  // Dense simulation data (15-sec steps, ±30 min around TCA) — for the slider
  const simPoints = useMemo((): ChartPoint[] => {
    if (hasTrail) {
      // Use pre-computed dense trail
      const trail = pair.trail!;
      // Find TCA within trail (minimum distance)
      let tcaI = 0;
      for (let i = 1; i < trail.length; i++) {
        if (trail[i].d < trail[tcaI].d) tcaI = i;
      }
      const tcaH = trail[tcaI].h;
      return trail.map(p => ({
        hourFromTCA: p.h - tcaH,
        hourAbs: p.h,
        distance: Math.round(p.d * 10) / 10,
        sat1: { x: p.s1[0], y: p.s1[1], z: p.s1[2] },
        sat2: { x: p.s2[0], y: p.s2[1], z: p.s2[2] },
      }));
    }
    // Fallback: use sparse chart data for slider too
    return chartPoints;
  }, [hasTrail, pair.trail, chartPoints]);

  // Find TCA index in simulation data (for default slider position)
  const simTcaIdx = useMemo(() => {
    if (simPoints.length === 0) return 0;
    let idx = 0;
    for (let i = 1; i < simPoints.length; i++) {
      if (Math.abs(simPoints[i].hourFromTCA) < Math.abs(simPoints[idx].hourFromTCA)) idx = i;
    }
    return idx;
  }, [simPoints]);

  const [sliderIdx, setSliderIdx] = useState(0);

  useEffect(() => {
    if (simPoints.length > 0) {
      setSliderIdx(simTcaIdx);
    }
  }, [simTcaIdx, simPoints.length]);

  const currentPoint = simPoints[sliderIdx] || null;

  // Build trail arrays for globe orbital paths
  const trailData = useMemo(() => {
    if (!pair.trail || pair.trail.length === 0) return null;
    const sat1Trail: [number, number, number][] = [];
    const sat2Trail: [number, number, number][] = [];
    for (const pt of pair.trail) {
      sat1Trail.push([pt.s1[0], pt.s1[1], pt.s1[2]]);
      sat2Trail.push([pt.s2[0], pt.s2[1], pt.s2[2]]);
    }
    return { sat1Trail, sat2Trail };
  }, [pair.trail]);

  // TCA positions for the constant red line on the globe
  const tcaPositions = useMemo(() => {
    if (simPoints.length === 0) return null;
    const tca = simPoints[simTcaIdx];
    return {
      sat1AtTCA: tca.sat1,
      sat2AtTCA: tca.sat2,
    };
  }, [simPoints, simTcaIdx]);

  // Project positions to globe whenever slider changes
  const emitProjection = useCallback((point: ChartPoint | null) => {
    if (!point) {
      onProjection(null);
      return;
    }
    onProjection({
      sat1: { ...point.sat1, name: pair.name_1, noradId: pair.norad_1 },
      sat2: { ...point.sat2, name: pair.name_2, noradId: pair.norad_2 },
      distanceKm: point.distance,
      ...(trailData || {}),
      ...(tcaPositions || {}),
    });
  }, [onProjection, pair, trailData, tcaPositions]);

  useEffect(() => {
    emitProjection(currentPoint);
  }, [currentPoint, emitProjection]);

  useEffect(() => {
    return () => onProjection(null);
  }, [onProjection]);

  const handleBack = useCallback(() => {
    onProjection(null);
    onBack();
  }, [onBack, onProjection]);

  const tcaPoint = simPoints.length > 0 ? simPoints[simTcaIdx] : null;

  const tier = riskTier(pair);
  const color = TIER_COLORS[tier];
  const hasData = chartPoints.length > 0 || simPoints.length > 0;
  const isLoading = !hasPrecomputed && !hasTrail && !tle1 && !tle2;

  return (
    <div className="absolute right-4 top-16 w-96 max-h-[85vh] rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)]/95 backdrop-blur-md shadow-2xl z-20 flex flex-col">
      {/* Header */}
      <div className="p-3 border-b border-[var(--color-border)]">
        <div className="flex items-center gap-2 mb-2">
          <button
            onClick={handleBack}
            className="w-7 h-7 flex items-center justify-center rounded-lg text-[var(--color-text-muted)] hover:text-[var(--color-text)] hover:bg-[var(--color-surface-2)] transition-colors text-sm"
          >
            &larr;
          </button>
          <h3 className="font-semibold text-sm flex-1">Conjunction Detail</h3>
          <span
            className="px-1.5 py-0.5 rounded text-[10px] font-bold"
            style={{ background: color + '22', color }}
          >
            {tier}
          </span>
        </div>

        {/* Pair info */}
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="rounded-lg bg-[var(--color-surface-2)] p-2 border-l-2 border-[#4f8aff]">
            <div className="text-[10px] text-[var(--color-text-muted)] mb-0.5">Object 1</div>
            <div className="font-medium truncate">{pair.name_1}</div>
            <div className="text-[var(--color-text-muted)] font-mono">{pair.norad_1}</div>
          </div>
          <div className="rounded-lg bg-[var(--color-surface-2)] p-2 border-l-2 border-[#ff8c42]">
            <div className="text-[10px] text-[var(--color-text-muted)] mb-0.5">Object 2</div>
            <div className="font-medium truncate">{pair.name_2}</div>
            <div className="text-[var(--color-text-muted)] font-mono">{pair.norad_2}</div>
          </div>
        </div>

        {/* CDM metadata panel */}
        {pair.source === 'cdm' && (
          <div className="mt-2 rounded-lg bg-[#8b5cf6]/8 border border-[#8b5cf6]/20 p-2 text-xs">
            <div className="flex items-center gap-1.5 mb-1.5">
              <span className="px-1.5 py-0.5 rounded text-[9px] font-bold bg-[#8b5cf6]/15 text-[#8b5cf6]">
                CDM
              </span>
              <span className="text-[var(--color-text-muted)]">Space-Track Conjunction Data Message</span>
              {pair.emergency_reportable === 'Y' && (
                <span className="ml-auto px-1.5 py-0.5 rounded text-[9px] font-bold bg-[#ff4f5a]/15 text-[#ff4f5a]">
                  EMERGENCY
                </span>
              )}
            </div>
            <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-[var(--color-text-muted)]">
              {pair.pc != null && (
                <div>Pc: <span className="font-mono font-semibold" style={{ color }}>{pair.pc.toExponential(2)}</span></div>
              )}
              {pair.miss_distance_km != null && (
                <div>Miss dist: <span className="text-[var(--color-text)]">{pair.miss_distance_km.toFixed(1)} km</span></div>
              )}
              {pair.sat1_type && (
                <div>Obj 1: <span className="text-[var(--color-text)]">{OBJ_TYPE_LABELS[pair.sat1_type.toUpperCase()] || pair.sat1_type}</span></div>
              )}
              {pair.sat2_type && (
                <div>Obj 2: <span className="text-[var(--color-text)]">{OBJ_TYPE_LABELS[pair.sat2_type.toUpperCase()] || pair.sat2_type}</span></div>
              )}
              {pair.cdm_tca && (
                <div className="col-span-2">CDM TCA: <span className="text-[var(--color-text)] font-mono">{new Date(pair.cdm_tca).toISOString().slice(0, 19).replace('T', ' ')} UTC</span></div>
              )}
            </div>
          </div>
        )}

        {/* Key metrics — CDM miss distance is authoritative; SGP4 is approximate */}
        <div className="flex justify-between mt-2 text-xs text-[var(--color-text-muted)]">
          <span>Alt: <span className="text-[var(--color-text)]">{pair.altitude_km.toFixed(0)} km</span></span>
          {pair.source === 'cdm' && pair.miss_distance_km != null ? (
            <span>CDM miss: <span className="text-[var(--color-text)] font-semibold">{pair.miss_distance_km.toFixed(1)} km</span></span>
          ) : tcaPoint ? (
            <span>Min sep: <span className="text-[var(--color-text)]">{tcaPoint.distance.toFixed(1)} km</span></span>
          ) : null}
          {pair.source === 'cdm' && pair.cdm_tca ? (
            <span>TCA: <span className="text-[var(--color-text)]">
              {(() => {
                const h = (new Date(pair.cdm_tca).getTime() - Date.now()) / 3600000;
                if (h < -24) return `${Math.abs(h / 24).toFixed(0)}d ago`;
                if (h < 0) return `${Math.abs(h).toFixed(0)}h ago`;
                if (h < 24) return `in ${h.toFixed(1)}h`;
                return `in ${(h / 24).toFixed(1)}d`;
              })()}
            </span></span>
          ) : tcaPoint ? (
            <span>TCA: <span className="text-[var(--color-text)]">
              {tcaPoint.hourAbs < 24
                ? `${tcaPoint.hourAbs.toFixed(1)}h`
                : `${(tcaPoint.hourAbs / 24).toFixed(1)}d`}
            </span></span>
          ) : null}
        </div>
      </div>

      {/* Chart */}
      <div className="p-3 flex-1 min-h-0">
        {hasData ? (
          <>
            <div className="text-[10px] text-[var(--color-text-muted)] mb-1.5">
              {pair.source === 'cdm'
                ? 'SGP4 approximate trajectory (CDM miss distance is authoritative)'
                : 'Separation distance — centered on closest approach'}
            </div>
            <div className="h-48">
              <ResponsiveContainer width="100%" height="100%">
                <AreaChart
                  data={chartPoints}
                  margin={{ top: 4, right: 4, left: -10, bottom: 0 }}
                >
                  <defs>
                    <linearGradient id="distGradient" x1="0" y1="0" x2="0" y2="1">
                      <stop offset="0%" stopColor="#4f8aff" stopOpacity={0.3} />
                      <stop offset="100%" stopColor="#4f8aff" stopOpacity={0.02} />
                    </linearGradient>
                  </defs>
                  <XAxis
                    dataKey="hourFromTCA"
                    tick={{ fontSize: 10, fill: 'var(--color-text-muted)' }}
                    tickFormatter={(h: number) => {
                      const sign = h < 0 ? '-' : '+';
                      const abs = Math.abs(h);
                      if (abs < 24) return `${sign}${abs.toFixed(0)}h`;
                      return `${sign}${(abs / 24).toFixed(0)}d`;
                    }}
                    axisLine={{ stroke: 'var(--color-border)' }}
                    tickLine={false}
                    interval="preserveStartEnd"
                  />
                  <YAxis
                    tick={{ fontSize: 10, fill: 'var(--color-text-muted)' }}
                    tickFormatter={(v: number) => v >= 1000 ? `${(v / 1000).toFixed(0)}k` : `${v.toFixed(0)}`}
                    axisLine={false}
                    tickLine={false}
                    width={45}
                    label={{ value: 'km', position: 'insideTopLeft', fontSize: 9, fill: 'var(--color-text-muted)', offset: 10 }}
                  />
                  <Tooltip
                    contentStyle={{
                      background: 'var(--color-surface)',
                      border: '1px solid var(--color-border)',
                      borderRadius: '8px',
                      fontSize: '11px',
                      padding: '6px 10px',
                    }}
                    formatter={(value: unknown) => [`${Number(value).toFixed(1)} km`, 'Separation']}
                    labelFormatter={(h: unknown) => {
                      const v = Number(h);
                      const sign = v < 0 ? '' : '+';
                      if (Math.abs(v) < 24) return `TCA ${sign}${v.toFixed(1)} hours`;
                      return `TCA ${sign}${(v / 24).toFixed(1)} days`;
                    }}
                  />
                  <ReferenceLine
                    x={0}
                    stroke="#ff4f5a"
                    strokeDasharray="4 4"
                    strokeWidth={1}
                    label={{
                      value: 'TCA',
                      position: 'top',
                      fontSize: 9,
                      fill: '#ff4f5a',
                    }}
                  />
                  {currentPoint && (
                    <ReferenceLine
                      x={currentPoint.hourFromTCA}
                      stroke="var(--color-accent)"
                      strokeWidth={1.5}
                      strokeOpacity={0.7}
                    />
                  )}
                  <Area
                    type="monotone"
                    dataKey="distance"
                    stroke="#4f8aff"
                    strokeWidth={1.5}
                    fill="url(#distGradient)"
                    dot={false}
                    activeDot={{ r: 3, fill: '#4f8aff' }}
                  />
                </AreaChart>
              </ResponsiveContainer>
            </div>

            {/* Time slider — uses dense simulation data (15-sec steps around TCA) */}
            <div className="mt-3 space-y-1.5">
              <div className="flex items-center justify-between text-[9px] text-[var(--color-text-muted)] mb-0.5">
                <span>-30 min</span>
                <span className="font-medium text-[var(--color-text)]">Closest Approach</span>
                <span>+30 min</span>
              </div>
              <input
                type="range"
                min={0}
                max={simPoints.length - 1}
                value={sliderIdx}
                onChange={e => setSliderIdx(Number(e.target.value))}
                className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
                style={{
                  background: `linear-gradient(to right, #4f8aff ${(sliderIdx / Math.max(simPoints.length - 1, 1)) * 100}%, var(--color-surface-2) ${(sliderIdx / Math.max(simPoints.length - 1, 1)) * 100}%)`,
                }}
              />
              {currentPoint && (
                <div className="flex justify-between text-[10px] text-[var(--color-text-muted)]">
                  <span>
                    TCA {currentPoint.hourFromTCA < 0 ? '' : '+'}
                    {Math.abs(currentPoint.hourFromTCA * 60).toFixed(0)} min
                  </span>
                  <span className="font-mono font-semibold" style={{
                    color: currentPoint.distance < 10 ? '#ff4f5a' : currentPoint.distance < 50 ? '#ffb84f' : 'var(--color-text)',
                  }}>
                    {currentPoint.distance.toFixed(1)} km
                  </span>
                </div>
              )}
            </div>

            {/* TCA summary — use CDM data when available */}
            {pair.source === 'cdm' && pair.miss_distance_km != null ? (
              <div className="mt-2 rounded-lg bg-[var(--color-surface-2)] p-2 text-xs">
                <div className="flex items-center gap-1.5 mb-1">
                  <div className="w-1.5 h-1.5 rounded-full bg-[#ff4f5a]" />
                  <span className="font-medium">Closest Approach (CDM)</span>
                </div>
                <div className="flex justify-between text-[var(--color-text-muted)]">
                  <span>
                    {pair.cdm_tca ? new Date(pair.cdm_tca).toISOString().slice(5, 16).replace('T', ' ') + ' UTC' : ''}
                  </span>
                  <span className="font-mono font-semibold">{pair.miss_distance_km.toFixed(1)} km</span>
                </div>
                {tcaPoint && Math.abs(tcaPoint.distance - pair.miss_distance_km) > 10 && (
                  <div className="mt-1 text-[10px] text-[var(--color-text-muted)] italic">
                    SGP4 shows {tcaPoint.distance.toFixed(1)} km — TLE-based propagation is less accurate for debris
                  </div>
                )}
              </div>
            ) : tcaPoint ? (
              <div className="mt-2 rounded-lg bg-[var(--color-surface-2)] p-2 text-xs">
                <div className="flex items-center gap-1.5 mb-1">
                  <div className="w-1.5 h-1.5 rounded-full bg-[#ff4f5a]" />
                  <span className="font-medium">Closest Approach</span>
                </div>
                <div className="flex justify-between text-[var(--color-text-muted)]">
                  <span>
                    T+{tcaPoint.hourAbs < 24
                      ? `${tcaPoint.hourAbs.toFixed(1)} hours`
                      : `${(tcaPoint.hourAbs / 24).toFixed(1)} days`}
                  </span>
                  <span className="font-mono">{tcaPoint.distance.toFixed(1)} km</span>
                </div>
              </div>
            ) : null}

            {/* Maneuver recommendation */}
            {tier !== 'LOW' && (() => {
              const isCDM = pair.source === 'cdm';
              const tcaDate = pair.cdm_tca ? new Date(pair.cdm_tca) : null;
              const hoursToTCA = tcaDate ? (tcaDate.getTime() - Date.now()) / 3600000 : tcaPoint?.hourAbs ?? null;
              const isPast = hoursToTCA != null && hoursToTCA < -1;

              return (
                <div className="mt-2 rounded-lg border p-2 text-xs" style={{ borderColor: color + '44', background: color + '08' }}>
                  <div className="flex items-center gap-1.5 mb-1.5">
                    <span className="font-semibold" style={{ color }}>
                      {isPast ? 'Post-Event Analysis' : 'Recommendation'}
                    </span>
                  </div>
                  {isPast ? (
                    <div className="space-y-1 text-[var(--color-text-muted)]">
                      <p>TCA has passed. {isCDM && pair.miss_distance_km != null
                        ? `Objects passed within ${pair.miss_distance_km.toFixed(1)} km.`
                        : 'Event is no longer actionable.'}</p>
                      <p>{tier === 'HIGH'
                        ? 'This was a high-risk event. Check for debris generation or orbit changes in follow-up TLEs.'
                        : 'Monitor for updated CDMs on this pair in subsequent screening cycles.'}</p>
                    </div>
                  ) : tier === 'HIGH' ? (
                    <div className="space-y-1 text-[var(--color-text-muted)]">
                      <p><span className="font-semibold text-[var(--color-text)]">Collision avoidance maneuver recommended.</span></p>
                      <p>Pc of {pair.pc != null ? pair.pc.toExponential(1) : '>1e-4'} exceeds the 1e-4 operational threshold
                        {isCDM ? ' per USSPACECOM conjunction assessment.' : '.'}</p>
                      {hoursToTCA != null && hoursToTCA > 0 && (
                        <p>Time to TCA: <span className="font-semibold text-[var(--color-text)]">
                          {hoursToTCA < 24 ? `${hoursToTCA.toFixed(1)} hours` : `${(hoursToTCA / 24).toFixed(1)} days`}
                        </span> &mdash; {hoursToTCA < 12 ? 'execute maneuver urgently' : 'plan maneuver within next screening cycle'}.</p>
                      )}
                    </div>
                  ) : (
                    <div className="space-y-1 text-[var(--color-text-muted)]">
                      <p>Monitor conjunction. {isCDM ? 'Request updated CDM to refine Pc estimate.' : 'Refine with higher-fidelity orbit determination.'}</p>
                      <p>Pre-plan avoidance maneuver as contingency if Pc increases above 1e-4.</p>
                    </div>
                  )}
                </div>
              );
            })()}
          </>
        ) : (
          <div className="h-48 flex flex-col items-center justify-center text-xs text-[var(--color-text-muted)] gap-2">
            {isLoading ? (
              <>
                <div className="w-5 h-5 border-2 border-[var(--color-accent)] border-t-transparent rounded-full animate-spin" />
                <span>Loading trajectory data...</span>
              </>
            ) : (
              <>
                <div className="w-5 h-5 border-2 border-[var(--color-accent)] border-t-transparent rounded-full animate-spin" />
                <span>Computing SGP4 trajectory...</span>
              </>
            )}
          </div>
        )}
      </div>

      {/* Footer */}
      <div className="p-2 border-t border-[var(--color-border)] text-center">
        <span className="text-[10px] text-[var(--color-text-muted)]">
          {hasPrecomputed ? 'Pre-computed trajectory' : 'SGP4 propagation'}
          {pair.source === 'cdm' ? ' (approx.)' : ''}
          &middot; drag slider to simulate approach
        </span>
      </div>
    </div>
  );
}
