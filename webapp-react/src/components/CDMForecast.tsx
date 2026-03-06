import { useState, useEffect, useMemo } from 'react';
import {
  XAxis, YAxis, CartesianGrid, Tooltip,
  ReferenceLine, ResponsiveContainer, Area, AreaChart,
} from 'recharts';

interface CDMUpdate {
  update_idx: number;
  log10_pc: number;
  pc: number;
  miss_distance_km: number;
  time_to_tca_hours: number;
}

interface ForecastPair {
  sat1_name: string;
  sat2_name: string;
  sat1_norad: number;
  sat2_norad: number;
  tca: string;
  current_pc: number;
  forecast_pc: number;
  exceedance_probability?: number;
  current_miss_km: number;
  forecast_miss_km: number;
  risk_direction: 'escalating' | 'stable' | 'de-escalating' | 'unknown';
  pc_trend: number;
  confidence: number;
  action_recommended: boolean;
  n_updates: number;
  time_series: CDMUpdate[];
}

interface ModelMetrics {
  mode: string;
  n_training_pairs: number;
  n_test_pairs: number;
  n_total_pairs: number;
  positive_rate: number;
  test_accuracy: number;
  test_precision: number;
  test_recall: number;
  test_f1: number;
  test_auc_pr: number;
}

interface ForecastData {
  generated_at: string;
  model: string;
  prediction_task: string;
  n_pairs: number;
  n_actionable: number;
  n_escalating: number;
  model_metrics?: ModelMetrics;
  pairs: ForecastPair[];
}

const DIRECTION_COLORS: Record<string, string> = {
  escalating: '#ff4f5a',
  stable: '#ffb84f',
  'de-escalating': '#4fff8a',
  unknown: '#666',
};

const DIRECTION_ICONS: Record<string, string> = {
  escalating: '\u2191',
  stable: '\u2192',
  'de-escalating': '\u2193',
  unknown: '?',
};

function formatPc(pc: number): string {
  if (pc >= 0.01) return `${(pc * 100).toFixed(1)}%`;
  return pc.toExponential(1);
}

function formatPct(v: number): string {
  return `${(v * 100).toFixed(1)}%`;
}

function ModelMetricsBanner({ metrics, task }: { metrics: ModelMetrics; task: string }) {
  const hasTest = metrics.test_accuracy > 0;

  return (
    <div className="px-4 py-3 border-b border-[var(--color-border)] space-y-2">
      <div className="text-[11px] text-[var(--color-text-muted)] leading-snug">
        <span className="font-semibold text-[var(--color-text)]">Prediction Task:</span>{' '}
        {task || 'Predict whether Pc will exceed 1e-4 before TCA'}
      </div>

      {hasTest ? (
        <div className="grid grid-cols-5 gap-2">
          <div className="text-center">
            <div className="text-sm font-bold text-[#4fff8a]">{formatPct(metrics.test_accuracy)}</div>
            <div className="text-[9px] text-[var(--color-text-muted)]">Accuracy</div>
          </div>
          <div className="text-center">
            <div className="text-sm font-bold text-[#4f8aff]">{metrics.test_f1.toFixed(3)}</div>
            <div className="text-[9px] text-[var(--color-text-muted)]">F1 Score</div>
          </div>
          <div className="text-center">
            <div className="text-sm font-bold text-[#8b5cf6]">{metrics.test_auc_pr.toFixed(3)}</div>
            <div className="text-[9px] text-[var(--color-text-muted)]">AUC-PR</div>
          </div>
          <div className="text-center">
            <div className="text-sm font-bold text-[var(--color-text)]">{metrics.n_training_pairs}</div>
            <div className="text-[9px] text-[var(--color-text-muted)]">Train Pairs</div>
          </div>
          <div className="text-center">
            <div className="text-sm font-bold text-[var(--color-text)]">{metrics.n_test_pairs}</div>
            <div className="text-[9px] text-[var(--color-text-muted)]">Test Pairs</div>
          </div>
        </div>
      ) : (
        <div className="text-[10px] text-[var(--color-text-muted)]">
          Mode: {metrics.mode} &middot; {metrics.n_total_pairs} pairs tracked &middot;
          Accuracy improves as more CDM data accumulates daily
        </div>
      )}

      <div className="text-[9px] text-[var(--color-text-muted)]">
        Validated on temporal test split (train on older pairs, test on recent resolved pairs)
        &middot; {formatPct(metrics.positive_rate)} of pairs exceed action threshold
      </div>
    </div>
  );
}

function PairTimeSeries({ pair }: { pair: ForecastPair }) {
  const chartData = useMemo(() => {
    return pair.time_series.map(u => ({
      update: u.update_idx,
      'log10(Pc)': u.log10_pc,
      'Miss (km)': u.miss_distance_km,
      'Time to TCA (h)': u.time_to_tca_hours,
    }));
  }, [pair.time_series]);

  const dirColor = DIRECTION_COLORS[pair.risk_direction];
  const thresholdLog = -3.3; // log10(5e-4) maneuver planning threshold
  const excProb = pair.exceedance_probability;

  return (
    <div className="rounded-lg bg-[var(--color-surface-2)] p-3 space-y-2">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="min-w-0">
          <div className="font-medium text-xs truncate">{pair.sat1_name} vs {pair.sat2_name}</div>
          <div className="text-[10px] text-[var(--color-text-muted)] font-mono">
            {pair.sat1_norad} / {pair.sat2_norad}
          </div>
        </div>
        <div className="flex items-center gap-2 shrink-0">
          <span className="text-lg" style={{ color: dirColor }}>
            {DIRECTION_ICONS[pair.risk_direction]}
          </span>
          {pair.action_recommended && (
            <span className="px-1.5 py-0.5 rounded text-[9px] font-bold bg-[#ff4f5a]/15 text-[#ff4f5a]">
              ACTION
            </span>
          )}
        </div>
      </div>

      {/* Metrics row */}
      <div className="grid grid-cols-5 gap-2 text-[10px]">
        <div>
          <div className="text-[var(--color-text-muted)]">Current Pc</div>
          <div className="font-mono font-semibold">{formatPc(pair.current_pc)}</div>
        </div>
        <div>
          <div className="text-[var(--color-text-muted)]">Forecast Pc</div>
          <div className="font-mono font-semibold" style={{ color: pair.forecast_pc >= 5e-4 ? '#ff4f5a' : '#4fff8a' }}>
            {formatPc(pair.forecast_pc)}
          </div>
        </div>
        {excProb != null && (
          <div>
            <div className="text-[var(--color-text-muted)]">P(Exceed)</div>
            <div className="font-mono font-semibold" style={{ color: excProb >= 0.5 ? '#ff4f5a' : '#4fff8a' }}>
              {formatPct(excProb)}
            </div>
          </div>
        )}
        <div>
          <div className="text-[var(--color-text-muted)]">Trend</div>
          <div className="font-semibold capitalize" style={{ color: dirColor }}>
            {pair.risk_direction}
          </div>
        </div>
        <div>
          <div className="text-[var(--color-text-muted)]">Updates</div>
          <div className="font-semibold">{pair.n_updates}</div>
        </div>
      </div>

      {/* Pc evolution chart */}
      {pair.time_series.length >= 2 && (
        <div className="h-28">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 4, right: 4, left: -15, bottom: 0 }}>
              <defs>
                <linearGradient id={`pcGrad-${pair.sat1_norad}`} x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor={dirColor} stopOpacity={0.3} />
                  <stop offset="100%" stopColor={dirColor} stopOpacity={0.02} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#333" />
              <XAxis
                dataKey="update"
                tick={{ fontSize: 9, fill: 'var(--color-text-muted)' }}
                label={{ value: 'CDM Update #', position: 'insideBottom', offset: -2, fill: '#666', fontSize: 8 }}
              />
              <YAxis
                tick={{ fontSize: 9, fill: 'var(--color-text-muted)' }}
                domain={['auto', 'auto']}
                label={{ value: 'log10(Pc)', position: 'insideTopLeft', fontSize: 8, fill: '#666', offset: 10 }}
              />
              <Tooltip
                contentStyle={{ background: '#1a1a2e', border: '1px solid #333', borderRadius: 8, fontSize: 11, padding: '6px 10px' }}
                formatter={(v: unknown, name?: string) => {
                  if (name === 'log10(Pc)') return [`${Number(v).toFixed(2)}`, name];
                  return [`${v}`, name ?? ''];
                }}
              />
              <ReferenceLine
                y={thresholdLog}
                stroke="#ff4f5a"
                strokeDasharray="4 4"
                strokeWidth={1}
                label={{ value: 'Maneuver threshold (5e-4)', position: 'right', fontSize: 8, fill: '#ff4f5a' }}
              />
              <Area
                type="monotone"
                dataKey="log10(Pc)"
                stroke={dirColor}
                strokeWidth={2}
                fill={`url(#pcGrad-${pair.sat1_norad})`}
                dot={{ r: 3, fill: dirColor }}
              />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Miss distance + TCA info */}
      <div className="flex justify-between text-[10px] text-[var(--color-text-muted)]">
        <span>Miss: {pair.current_miss_km.toFixed(0)} km (current) / {Math.max(0, pair.forecast_miss_km).toFixed(0)} km (forecast)</span>
        {pair.tca && (
          <span>TCA: {pair.tca.slice(5, 16).replace('T', ' ')}</span>
        )}
      </div>
    </div>
  );
}

export function CDMForecast({ visible, onClose }: { visible: boolean; onClose: () => void }) {
  const [data, setData] = useState<ForecastData | null>(null);
  const [filter, setFilter] = useState<'all' | 'escalating' | 'action'>('all');

  useEffect(() => {
    if (!visible) return;
    fetch('./cdm_forecast.json')
      .then(r => r.ok ? r.json() : null)
      .then(setData)
      .catch(() => setData(null));
  }, [visible]);

  const filtered = useMemo(() => {
    if (!data) return [];
    switch (filter) {
      case 'escalating': return data.pairs.filter(p => p.risk_direction === 'escalating');
      case 'action': return data.pairs.filter(p => p.action_recommended);
      default: return data.pairs;
    }
  }, [data, filter]);

  if (!visible) return null;

  return (
    <div className="absolute bottom-12 left-4 right-4 max-h-[60vh] rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)]/95 backdrop-blur-md shadow-2xl z-20 flex flex-col">
      <div className="flex items-center justify-between px-4 py-2 border-b border-[var(--color-border)]">
        <div className="flex items-center gap-3">
          <h3 className="font-semibold text-sm">Pc Escalation Forecast</h3>
          <span className="text-[10px] px-1.5 py-0.5 rounded bg-[#8b5cf6]/15 text-[#8b5cf6] font-medium">
            Will Pc exceed 5e-4?
          </span>
        </div>
        <button
          onClick={onClose}
          className="text-[var(--color-text-muted)] hover:text-[var(--color-text)] transition-colors text-lg leading-none"
        >
          x
        </button>
      </div>

      {!data ? (
        <div className="p-6 text-xs text-[var(--color-text-muted)] text-center">
          CDM forecast data not available. Run the daily pipeline to generate predictions.
        </div>
      ) : (
        <>
          {/* Model metrics banner */}
          {data.model_metrics && (
            <ModelMetricsBanner
              metrics={data.model_metrics}
              task={data.prediction_task}
            />
          )}

          {/* Summary stats */}
          <div className="grid grid-cols-4 gap-2 px-4 py-2 border-b border-[var(--color-border)]">
            <div className="text-center">
              <div className="text-lg font-bold text-[#8b5cf6]">{data.n_pairs}</div>
              <div className="text-[10px] text-[var(--color-text-muted)]">Pairs Tracked</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold text-[#ff4f5a]">{data.n_escalating}</div>
              <div className="text-[10px] text-[var(--color-text-muted)]">Escalating</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold text-[#ffb84f]">{data.n_actionable}</div>
              <div className="text-[10px] text-[var(--color-text-muted)]">Actionable</div>
            </div>
            <div className="text-center">
              <div className="text-lg font-bold text-[var(--color-text)]">
                {data.pairs.filter(p => p.n_updates >= 3).length}
              </div>
              <div className="text-[10px] text-[var(--color-text-muted)]">3+ Updates</div>
            </div>
          </div>

          {/* Filter tabs */}
          <div className="flex gap-2 px-4 py-2 border-b border-[var(--color-border)]">
            {(['all', 'action', 'escalating'] as const).map(f => (
              <button
                key={f}
                onClick={() => setFilter(f)}
                className={`px-3 py-1 rounded-full text-[11px] font-medium border transition-all ${
                  filter === f
                    ? 'border-[var(--color-accent)] bg-[var(--color-accent)]/10 text-[var(--color-text)]'
                    : 'border-[var(--color-border)] text-[var(--color-text-muted)] hover:text-[var(--color-text)]'
                }`}
              >
                {f === 'all' ? `All (${data.pairs.length})` :
                 f === 'action' ? `Action Required (${data.n_actionable})` :
                 `Escalating (${data.n_escalating})`}
              </button>
            ))}
          </div>

          {/* Pair list */}
          <div className="overflow-y-auto flex-1 p-3 space-y-2">
            {filtered.length === 0 ? (
              <div className="text-xs text-[var(--color-text-muted)] text-center py-4">
                No pairs match this filter.
              </div>
            ) : (
              filtered.map(pair => (
                <PairTimeSeries
                  key={`${pair.sat1_norad}-${pair.sat2_norad}`}
                  pair={pair}
                />
              ))
            )}
          </div>

          {/* Footer */}
          <div className="px-4 py-2 border-t border-[var(--color-border)] text-center">
            <span className="text-[10px] text-[var(--color-text-muted)]">
              Logistic regression on CDM time series features &middot;
              Continuously retrained daily on resolved pairs &middot;
              Source: Space-Track 18th SDS CDMs
            </span>
          </div>
        </>
      )}
    </div>
  );
}
