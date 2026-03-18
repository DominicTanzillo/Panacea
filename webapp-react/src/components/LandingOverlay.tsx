import { useState, useEffect } from 'react';
import { PanaceaLogo } from './Header';

interface LandingOverlayProps {
  loading: boolean;
  nSatellites: number;
  nPairs: number;
  onEnter: () => void;
  onExploreModels?: () => void;
}

const MODELS = [
  { name: 'Logistic Regression', metric: 'F1 0.854', role: 'Production baseline' },
  { name: 'BiLSTM Ensemble', metric: '98.7% recall', role: 'Transfer learning + focal loss' },
  { name: 'Graph Neural Net', metric: 'F1 0.783', role: 'Conjunction network topology' },
  { name: 'Autoregressive', metric: 'r=0.931', role: 'Next-CDM forecasting' },
  { name: 'Conformal', metric: '90% coverage', role: 'Distribution-free UQ' },
  { name: 'Event Summarizer', metric: 'NLP', role: 'Human-readable risk reports' },
];

export function LandingOverlay({ loading, nSatellites, onEnter, onExploreModels }: LandingOverlayProps) {
  const [show, setShow] = useState(true);
  const [fadeOut, setFadeOut] = useState(false);
  const [phase, setPhase] = useState(0); // 0=title, 1=stats, 2=models

  useEffect(() => {
    if (!show) return;
    const t1 = setTimeout(() => setPhase(1), 800);
    const t2 = setTimeout(() => setPhase(2), 1400);
    return () => { clearTimeout(t1); clearTimeout(t2); };
  }, [show]);

  const handleEnter = () => {
    setFadeOut(true);
    setTimeout(() => { setShow(false); onEnter(); }, 500);
  };

  if (!show) return null;

  return (
    <div
      className={`fixed inset-0 flex flex-col items-center justify-center transition-opacity duration-500
        ${fadeOut ? 'opacity-0 pointer-events-none' : 'opacity-100'}`}
      style={{
        zIndex: 9999,
        background: 'linear-gradient(180deg, rgba(10,10,15,0.92) 0%, rgba(10,10,15,0.88) 50%, rgba(10,10,15,0.95) 100%)',
        backdropFilter: 'blur(6px)',
      }}
    >
      {/* Ambient glow */}
      <div className="absolute inset-0 pointer-events-none" style={{
        background: 'radial-gradient(ellipse 600px 400px at 50% 40%, rgba(79,138,255,0.06) 0%, transparent 100%)',
      }} />

      <div className="relative max-w-2xl w-full px-6">
        {/* Title block */}
        <div className="text-center mb-10">
          <div className="inline-block mb-5">
            <PanaceaLogo size={44} />
          </div>
          <h1
            className="text-5xl sm:text-6xl font-extrabold tracking-tight mb-3"
            style={{
              background: 'linear-gradient(135deg, #e0e0e8 20%, #4f8aff 60%, #8b5cf6 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              lineHeight: 1.1,
            }}
          >
            PANACEA
          </h1>
          <p className="text-sm text-[var(--color-text-muted)] tracking-wide">
            Predictive Assessment Network for Automated Conjunction Evaluation & Avoidance
          </p>
        </div>

        {/* Stats strip */}
        <div className={`flex justify-center gap-10 mb-10 transition-all duration-600
          ${phase >= 1 ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-3'}`}>
          <StatBlock value={nSatellites > 0 ? `${(nSatellites / 1000).toFixed(0)}K` : '--'} label="Objects" />
          <div className="w-px bg-[var(--color-border)]" />
          <StatBlock value="670" label="CDM Pairs" />
          <div className="w-px bg-[var(--color-border)]" />
          <StatBlock value="6" label="Models" />
          <div className="w-px bg-[var(--color-border)]" />
          <StatBlock value="90%" label="Coverage" />
        </div>

        {/* Model grid */}
        <div className={`grid grid-cols-3 gap-2 mb-8 transition-all duration-600
          ${phase >= 2 ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-3'}`}
        >
          {MODELS.map((m, i) => (
            <div
              key={i}
              className="bg-[var(--color-surface)]/70 border border-[var(--color-border)]/60 px-4 py-3.5 hover:bg-[var(--color-surface-2)] hover:border-[var(--color-border)] transition-colors"
              style={{ borderRadius: 4 }}
            >
              <div className="text-[11px] font-semibold text-[var(--color-text)] mb-1">{m.name}</div>
              <div className="text-[10px] font-mono text-[var(--color-accent)]">{m.metric}</div>
              <div className="text-[9px] text-[var(--color-text-muted)] mt-1 leading-snug">{m.role}</div>
            </div>
          ))}
        </div>

        {/* CTA */}
        <div className={`text-center transition-all duration-500
          ${phase >= 2 ? 'opacity-100' : 'opacity-0'}`}>
          <button
            onClick={handleEnter}
            disabled={loading}
            className={`group px-7 py-2.5 text-sm font-medium tracking-wide transition-all duration-200
              ${loading
                ? 'bg-[var(--color-surface-2)] text-[var(--color-text-muted)] cursor-wait'
                : 'bg-[var(--color-accent)] text-white hover:brightness-110 active:brightness-95'
              }`}
            style={{ borderRadius: 4 }}
          >
            {loading ? 'Loading orbital data...' : 'Enter Dashboard'}
            {!loading && (
              <span className="ml-2 inline-block group-hover:translate-x-0.5 transition-transform">&rarr;</span>
            )}
          </button>
          {onExploreModels && (
            <button
              onClick={onExploreModels}
              className="ml-4 px-5 py-2.5 text-sm font-medium tracking-wide text-[var(--color-text-muted)] border border-[var(--color-border)] hover:text-[var(--color-text)] hover:border-[var(--color-text-muted)] transition-colors"
              style={{ borderRadius: 4 }}
            >
              Explore Models
            </button>
          )}
          <p className="mt-6 text-[9px] text-[var(--color-text-muted)]/50 tracking-wider uppercase">
            AIPI 540 &middot; Duke University &middot; Space-Track CDMs + CelesTrak TLEs
          </p>
        </div>
      </div>
    </div>
  );
}

function StatBlock({ value, label }: { value: string; label: string }) {
  return (
    <div className="text-center min-w-[60px]">
      <div className="text-2xl font-bold text-[var(--color-text)] tabular-nums">{value}</div>
      <div className="text-[9px] text-[var(--color-text-muted)] uppercase tracking-wider mt-0.5">{label}</div>
    </div>
  );
}
