import { useMemo, useState, useEffect } from 'react';
import * as Tabs from '@radix-ui/react-tabs';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from 'recharts';
import type { ModelComparisonResult, StalenessResults, PipelineStats } from '../lib/api';
import type { SatellitePosition } from '../lib/types';
import { Overlay } from './Overlay';

interface RiskDashboardProps {
  modelComparison: ModelComparisonResult[] | null;
  experimentResults: StalenessResults | null;
  satellites: SatellitePosition[];
  visible: boolean;
  onClose: () => void;
}

/* ── Density tab ────────────────────────────────────────────── */

function DensityTab({ satellites }: { satellites: SatellitePosition[] }) {
  const BIN_WIDTH = 50;
  const chartData = useMemo(() => {
    const bins: Record<number, number> = {};
    for (const sat of satellites) {
      const bin = Math.round(sat.alt / BIN_WIDTH) * BIN_WIDTH;
      if (bin >= 0 && bin <= 2000) bins[bin] = (bins[bin] || 0) + 1;
    }
    return Object.entries(bins)
      .map(([alt, count]) => ({ altitude: `${alt}`, count }))
      .sort((a, b) => parseInt(a.altitude) - parseInt(b.altitude));
  }, [satellites]);

  return (
    <div className="p-4">
      <div className="text-xs text-[var(--color-text-muted)] mb-3">
        Object density by altitude ({BIN_WIDTH}km bins, LEO)
      </div>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#222" />
          <XAxis dataKey="altitude" tick={{ fill: '#666', fontSize: 9 }} interval={3} />
          <YAxis tick={{ fill: '#666', fontSize: 10 }} />
          <Tooltip
            contentStyle={{ background: '#14141e', border: '1px solid #2a2a3a', borderRadius: 4, fontSize: 11 }}
            labelFormatter={(val) => `${val} km`}
          />
          <Bar dataKey="count" fill="#8b5cf6" radius={[1, 1, 0, 0]} />
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

/* ── Pipeline tab ───────────────────────────────────────────── */

function PipelineTab() {
  const [stats, setStats] = useState<PipelineStats | null>(null);

  useEffect(() => {
    fetch('./pipeline_stats.json')
      .then(r => r.ok ? r.json() : null)
      .then(setStats)
      .catch(() => setStats(null));
  }, []);

  if (!stats) {
    return (
      <div className="p-6 text-xs text-[var(--color-text-muted)] text-center">
        Pipeline stats not yet generated. Runs daily at 00:00 UTC.
      </div>
    );
  }

  const fm = stats.forecast_model;
  const testMetrics = fm?.test;

  const deduped = Object.values(
    stats.daily_history.reduce<Record<string, typeof stats.daily_history[0]>>((acc, d) => {
      acc[d.date] = d; return acc;
    }, {})
  ).sort((a, b) => a.date.localeCompare(b.date));

  const dailyData = deduped.slice(-14).map(d => ({
    date: d.date.slice(5, 10),
    screened: Math.round((d.n_satellites_screened ?? 0) / 1000),
    predictions: d.n_predictions_logged ?? 0,
  }));

  return (
    <div className="p-4 space-y-5">
      {stats.generated_at && (
        <div className="text-[9px] text-[var(--color-text-muted)] text-right uppercase tracking-wider">
          Updated {new Date(stats.generated_at).toLocaleDateString()}
        </div>
      )}

      {/* Forecast model card */}
      {fm && (
        <div className="bg-[var(--color-surface-2)] p-4" style={{ borderRadius: 4 }}>
          <div className="text-xs font-semibold text-[var(--color-text)] mb-1">
            Pc Escalation Forecast
          </div>
          <div className="text-[10px] text-[var(--color-text-muted)] mb-3">
            Predict Pc &gt; 5e-4 before TCA
          </div>
          {testMetrics ? (
            <>
              <div className="grid grid-cols-4 gap-4">
                <MetricCell value={`${(testMetrics.accuracy * 100).toFixed(0)}%`} label="Accuracy" color="#4fff8a" />
                <MetricCell value={testMetrics.f1.toFixed(3)} label="F1" color="#4f8aff" />
                <MetricCell value={testMetrics.auc_pr.toFixed(3)} label="AUC-PR" color="#8b5cf6" />
                <MetricCell value={`${fm.n_test ?? 0}`} label="Test Pairs" color="var(--color-text)" />
              </div>
              {/* Confusion matrix */}
              {testMetrics.tp != null && (
                <div className="mt-4 pt-3 border-t border-[var(--color-border)]/20">
                  <div className="text-[9px] text-[var(--color-text-muted)] mb-2 uppercase tracking-wider">Confusion Matrix</div>
                  <div className="grid grid-cols-2 gap-1 max-w-[140px]">
                    <CMCell value={testMetrics.tp ?? 0} label="TP" good />
                    <CMCell value={testMetrics.fp ?? 0} label="FP" good={false} />
                    <CMCell value={testMetrics.fn ?? 0} label="FN" good={false} />
                    <CMCell value={testMetrics.tn ?? 0} label="TN" good />
                  </div>
                </div>
              )}
            </>
          ) : (
            <div className="text-[10px] text-[var(--color-text-muted)]">
              {fm.mode} mode &middot; {fm.n_pairs_total ?? 0} pairs
            </div>
          )}
        </div>
      )}

      {/* CDM stats */}
      <div className="grid grid-cols-3 gap-3">
        <StatCard value={stats.cdm_stats.pc_high} label="Pc >= 5e-4" color="#ff4f5a" />
        <StatCard value={stats.cdm_stats.pc_moderate} label="1e-4 <= Pc < 5e-4" color="#ffb84f" />
        <StatCard value={stats.cdm_stats.emergency_count} label="Emergency" color="#8b5cf6" />
      </div>
      <div className="text-[9px] text-[var(--color-text-muted)] text-center">
        {stats.cdm_stats.total_cdms.toLocaleString()} CDMs from Space-Track 18th SDS
      </div>

      {/* Activity chart */}
      {dailyData.length > 0 && (
        <div>
          <div className="text-[10px] text-[var(--color-text-muted)] mb-2 uppercase tracking-wider">
            14-Day Pipeline Activity
          </div>
          <ResponsiveContainer width="100%" height={100}>
            <BarChart data={dailyData} margin={{ top: 2, right: 4, left: -12, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#222" />
              <XAxis dataKey="date" tick={{ fill: '#555', fontSize: 8 }} />
              <YAxis tick={{ fill: '#555', fontSize: 8 }} />
              <Tooltip
                contentStyle={{ background: '#14141e', border: '1px solid #2a2a3a', borderRadius: 4, fontSize: 10 }}
              />
              <Bar dataKey="screened" fill="#4f8aff" radius={[1, 1, 0, 0]} name="Sats (K)" />
              <Bar dataKey="predictions" fill="#4fff8a" radius={[1, 1, 0, 0]} name="Predictions" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
}

function MetricCell({ value, label, color }: { value: string; label: string; color: string }) {
  return (
    <div>
      <div className="text-lg font-bold font-mono" style={{ color }}>{value}</div>
      <div className="text-[9px] text-[var(--color-text-muted)] uppercase tracking-wider">{label}</div>
    </div>
  );
}

function CMCell({ value, label, good }: { value: number; label: string; good: boolean }) {
  return (
    <div
      className="p-2 text-center text-[10px]"
      style={{ background: good ? 'rgba(79,255,138,0.08)' : 'rgba(255,79,90,0.08)', borderRadius: 3 }}
    >
      <div className="font-bold" style={{ color: good ? '#4fff8a' : '#ff4f5a' }}>{value}</div>
      <div className="text-[8px] text-[var(--color-text-muted)]">{label}</div>
    </div>
  );
}

function StatCard({ value, label, color }: { value: number; label: string; color: string }) {
  return (
    <div className="bg-[var(--color-surface-2)] p-3 text-center" style={{ borderRadius: 4 }}>
      <div className="text-xl font-bold font-mono" style={{ color }}>{value}</div>
      <div className="text-[9px] text-[var(--color-text-muted)] mt-0.5">{label}</div>
    </div>
  );
}

/* ── Main Dashboard ─────────────────────────────────────────── */

export function RiskDashboard({
  modelComparison: _mc,
  experimentResults: _er,
  satellites,
  visible,
  onClose,
}: RiskDashboardProps) {
  void _mc; void _er;

  return (
    <Overlay visible={visible} onClose={onClose} title="Dashboard" subtitle="Pipeline status and orbital density" maxWidth="900px">
      <Tabs.Root defaultValue="pipeline" className="flex flex-col">
        <Tabs.List className="flex border-b border-[var(--color-border-subtle)] px-6">
          {['pipeline', 'density'].map(tab => (
            <Tabs.Trigger
              key={tab}
              value={tab}
              className="px-4 py-3 text-xs font-medium text-[var(--color-text-muted)] border-b-2 border-transparent
                data-[state=active]:text-[var(--color-text)] data-[state=active]:border-[var(--color-accent)]
                transition-colors capitalize tracking-wide"
            >
              {tab}
            </Tabs.Trigger>
          ))}
        </Tabs.List>

        <div>
          <Tabs.Content value="pipeline"><PipelineTab /></Tabs.Content>
          <Tabs.Content value="density"><DensityTab satellites={satellites} /></Tabs.Content>
        </div>
      </Tabs.Root>
    </Overlay>
  );
}
