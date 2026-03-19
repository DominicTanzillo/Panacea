import { useState, useEffect, useMemo } from 'react';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from 'recharts';
import type { ModelComparisonResult, StalenessResults, PipelineStats } from '../lib/api';
import type { SatellitePosition } from '../lib/types';
import { FullPagePanel } from './Overlay';

interface RiskDashboardProps {
  modelComparison: ModelComparisonResult[] | null;
  experimentResults: StalenessResults | null;
  satellites: SatellitePosition[];
  visible: boolean;
  onClose: () => void;
}

const TOOLTIP_STYLE = {
  background: '#111118', border: '1px solid #2a2a3a', borderRadius: 6, fontSize: 12, padding: '8px 12px',
};

export function RiskDashboard({
  modelComparison: _mc, experimentResults: _er, satellites, visible, onClose,
}: RiskDashboardProps) {
  void _mc; void _er;
  const [stats, setStats] = useState<PipelineStats | null>(null);

  useEffect(() => {
    if (!visible) return;
    fetch('./pipeline_stats.json')
      .then(r => r.ok ? r.json() : null)
      .then(setStats)
      .catch(() => setStats(null));
  }, [visible]);

  const fm = stats?.forecast_model;
  const testMetrics = fm?.test;

  const deduped = useMemo(() => {
    if (!stats) return [];
    return Object.values(
      stats.daily_history.reduce<Record<string, typeof stats.daily_history[0]>>((acc, d) => {
        acc[d.date] = d; return acc;
      }, {})
    ).sort((a, b) => a.date.localeCompare(b.date));
  }, [stats]);

  const dailyData = useMemo(() => {
    return deduped.slice(-14).map(d => ({
      date: d.date.slice(5, 10),
      screened: Math.round((d.n_satellites_screened ?? 0) / 1000),
      predictions: d.n_predictions_logged ?? 0,
    }));
  }, [deduped]);

  // Density data
  const densityData = useMemo(() => {
    const BIN = 50;
    const bins: Record<number, number> = {};
    for (const sat of satellites) {
      const bin = Math.round(sat.alt / BIN) * BIN;
      if (bin >= 0 && bin <= 2000) bins[bin] = (bins[bin] || 0) + 1;
    }
    return Object.entries(bins)
      .map(([alt, count]) => ({ altitude: `${alt}`, count }))
      .sort((a, b) => parseInt(a.altitude) - parseInt(b.altitude));
  }, [satellites]);

  return (
    <FullPagePanel
      visible={visible}
      onClose={onClose}
      title="Pipeline"
      subtitle={stats?.generated_at ? `Last run: ${new Date(stats.generated_at).toLocaleString()}` : 'Daily automated pipeline'}
      maxWidth={1000}
    >
      {!stats ? (
        <div style={{ padding: 48, textAlign: 'center', fontSize: 14, color: '#7c7c96' }}>
          Pipeline stats not yet generated. The pipeline runs daily at 00:00 UTC.
        </div>
      ) : (
        <>
          {/* ── How it works ─────────────────────────────────── */}
          <div style={{ padding: '24px 0 20px', borderBottom: '1px solid #1e1e2c' }}>
            <SectionHeader>How the Pipeline Works</SectionHeader>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 12 }}>
              <StepCard step={1} title="Fetch CDMs" desc="Download latest conjunction data messages from Space-Track (18th SDS)" />
              <StepCard step={2} title="Feature Engineering" desc="Extract 20 features per pair: Pc trend, miss distance slope, RCS, debris flags" />
              <StepCard step={3} title="Model Inference" desc="Ensemble prediction: 40% LR + 30% BiLSTM + 30% regression signal" />
              <StepCard step={4} title="Export & Deploy" desc="Write forecasts to JSON, deploy webapp to GitHub Pages automatically" />
            </div>
          </div>

          {/* ── Forecast model metrics ───────────────────────── */}
          {fm && testMetrics && (
            <div style={{ padding: '24px 0', borderBottom: '1px solid #1e1e2c' }}>
              <SectionHeader>Forecast Model Performance</SectionHeader>
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 24 }}>
                <MetricCard value={`${(testMetrics.accuracy * 100).toFixed(0)}%`} label="Accuracy" />
                <MetricCard value={testMetrics.f1.toFixed(3)} label="F1 Score" />
                <MetricCard value={testMetrics.auc_pr.toFixed(3)} label="AUC-PR" />
                <MetricCard value={`${fm.n_test ?? 0}`} label="Test Pairs" />
              </div>

              {/* Confusion matrix */}
              {testMetrics.tp != null && (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: 4, maxWidth: 400 }}>
                  <CMCell value={testMetrics.tp ?? 0} label="TP" good />
                  <CMCell value={testMetrics.fp ?? 0} label="FP" good={false} />
                  <CMCell value={testMetrics.fn ?? 0} label="FN" good={false} />
                  <CMCell value={testMetrics.tn ?? 0} label="TN" good />
                </div>
              )}
            </div>
          )}

          {/* ── CDM stats ────────────────────────────────────── */}
          <div style={{ padding: '24px 0', borderBottom: '1px solid #1e1e2c' }}>
            <SectionHeader>CDM Statistics</SectionHeader>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(4, 1fr)', gap: 16, marginBottom: 8 }}>
              <MetricCard value={stats.cdm_stats.total_cdms.toLocaleString()} label="Total CDMs" />
              <MetricCard value={String(stats.cdm_stats.pc_high)} label="Pc \u2265 5e-4" color="#ef4444" />
              <MetricCard value={String(stats.cdm_stats.pc_moderate)} label="1e-4 \u2264 Pc < 5e-4" color="#f59e0b" />
              <MetricCard value={String(stats.cdm_stats.emergency_count)} label="Emergency Reportable" color="#ef4444" />
            </div>
          </div>

          {/* ── 14-day activity ───────────────────────────────── */}
          {dailyData.length > 0 && (
            <div style={{ padding: '24px 0', borderBottom: '1px solid #1e1e2c' }}>
              <SectionHeader>14-Day Pipeline Activity</SectionHeader>
              <ResponsiveContainer width="100%" height={180}>
                <BarChart data={dailyData} margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e1e2c" />
                  <XAxis dataKey="date" tick={{ fill: '#55556a', fontSize: 12 }} axisLine={{ stroke: '#2a2a3a' }} tickLine={false} />
                  <YAxis tick={{ fill: '#55556a', fontSize: 12 }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={TOOLTIP_STYLE} />
                  <Bar dataKey="screened" fill="#3b82f6" radius={[2, 2, 0, 0]} name="Satellites (K)" opacity={0.7} />
                  <Bar dataKey="predictions" fill="#e8e8f0" radius={[2, 2, 0, 0]} name="Predictions" opacity={0.5} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* ── Orbital density ───────────────────────────────── */}
          {densityData.length > 0 && (
            <div style={{ paddingTop: 24 }}>
              <SectionHeader>Object Density by Altitude</SectionHeader>
              <p style={{ fontSize: 13, color: '#55556a', marginBottom: 16 }}>
                50 km bins, LEO region. Shows congestion in major orbital shells.
              </p>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={densityData} margin={{ top: 8, right: 16, left: 0, bottom: 4 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e1e2c" />
                  <XAxis dataKey="altitude" tick={{ fill: '#55556a', fontSize: 12 }} interval={3} axisLine={{ stroke: '#2a2a3a' }} tickLine={false} />
                  <YAxis tick={{ fill: '#55556a', fontSize: 12 }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={TOOLTIP_STYLE} labelFormatter={val => `${val} km`} />
                  <Bar dataKey="count" fill="#3b82f6" radius={[2, 2, 0, 0]} opacity={0.6} />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}
        </>
      )}
    </FullPagePanel>
  );
}

/* ── Sub-components ───────────────────────────────────────── */
function SectionHeader({ children }: { children: React.ReactNode }) {
  return (
    <div style={{ fontSize: 13, color: '#55556a', textTransform: 'uppercase', letterSpacing: '0.06em', fontWeight: 600, marginBottom: 16 }}>
      {children}
    </div>
  );
}

function StepCard({ step, title, desc }: { step: number; title: string; desc: string }) {
  return (
    <div style={{ padding: '16px', background: '#111118', borderRadius: 8, border: '1px solid #1e1e2c' }}>
      <div style={{ fontSize: 12, color: '#3b82f6', fontWeight: 700, fontFamily: "'JetBrains Mono', monospace", marginBottom: 6 }}>
        {String(step).padStart(2, '0')}
      </div>
      <div style={{ fontSize: 14, fontWeight: 600, color: '#e8e8f0', marginBottom: 4 }}>{title}</div>
      <div style={{ fontSize: 13, color: '#55556a', lineHeight: 1.5 }}>{desc}</div>
    </div>
  );
}

function MetricCard({ value, label, color }: { value: string; label: string; color?: string }) {
  return (
    <div style={{ padding: '14px 16px', background: '#111118', borderRadius: 8, border: '1px solid #1e1e2c' }}>
      <div style={{ fontSize: 24, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace", color: color ?? '#e8e8f0', lineHeight: 1.2 }}>
        {value}
      </div>
      <div style={{ fontSize: 12, color: '#55556a', marginTop: 4 }}>{label}</div>
    </div>
  );
}

function CMCell({ value, label, good }: { value: number; label: string; good: boolean }) {
  return (
    <div style={{
      padding: '12px 8px',
      textAlign: 'center',
      background: good ? 'rgba(34,197,94,0.05)' : 'rgba(239,68,68,0.05)',
      borderRadius: 6,
    }}>
      <div style={{ fontSize: 20, fontWeight: 700, fontFamily: "'JetBrains Mono', monospace", color: good ? '#22c55e' : '#ef4444' }}>{value}</div>
      <div style={{ fontSize: 12, color: '#55556a', marginTop: 2 }}>{label}</div>
    </div>
  );
}
