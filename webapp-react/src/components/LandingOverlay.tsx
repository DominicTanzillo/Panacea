import { useState, useEffect } from 'react';
import { PanaceaLogo } from './Header';

interface LandingOverlayProps {
  loading: boolean;
  nSatellites: number;
  nPairs: number;
  onEnter: () => void;
}

const FEATURES = [
  { icon: '\u{1F6F0}', label: 'Real-Time Tracking', desc: '25,000+ objects tracked via CelesTrak & Space-Track' },
  { icon: '\u{1F9E0}', label: '6 ML Models', desc: 'LogReg, BiLSTM, GNN, Ensemble, Conformal, Autoregressive' },
  { icon: '\u{1F4CA}', label: 'Calibrated Uncertainty', desc: 'Conformal prediction with 90% coverage guarantee' },
  { icon: '\u{1F310}', label: 'Graph Intelligence', desc: 'GNN captures congested orbital neighborhoods' },
];

export function LandingOverlay({ loading, nSatellites, onEnter }: LandingOverlayProps) {
  const [show, setShow] = useState(true);
  const [fadeOut, setFadeOut] = useState(false);
  const [typedText, setTypedText] = useState('');
  const [statsVisible, setStatsVisible] = useState(false);

  const fullText = 'Orbital Debris Collision Prediction System';

  // Typing animation
  useEffect(() => {
    if (!show) return;
    let i = 0;
    const timer = setInterval(() => {
      if (i < fullText.length) {
        setTypedText(fullText.slice(0, i + 1));
        i++;
      } else {
        clearInterval(timer);
        setTimeout(() => setStatsVisible(true), 300);
      }
    }, 35);
    return () => clearInterval(timer);
  }, [show]);

  const handleEnter = () => {
    setFadeOut(true);
    setTimeout(() => {
      setShow(false);
      onEnter();
    }, 600);
  };

  if (!show) return null;

  return (
    <div
      className={`fixed inset-0 z-50 flex flex-col items-center justify-center
        transition-opacity duration-700 ${fadeOut ? 'opacity-0' : 'opacity-100'}`}
      style={{
        background: 'radial-gradient(ellipse at 50% 50%, rgba(10,10,15,0.85) 0%, rgba(10,10,15,0.97) 70%)',
        backdropFilter: 'blur(4px)',
      }}
    >
      {/* Glow effect behind logo */}
      <div
        className="absolute"
        style={{
          width: 400,
          height: 400,
          background: 'radial-gradient(circle, rgba(79,138,255,0.08) 0%, transparent 70%)',
          top: '50%',
          left: '50%',
          transform: 'translate(-50%, -60%)',
          pointerEvents: 'none',
        }}
      />

      {/* Logo + Title */}
      <div className="relative flex flex-col items-center mb-8">
        <div className="mb-4 animate-pulse">
          <PanaceaLogo size={56} />
        </div>
        <h1
          className="text-4xl sm:text-5xl font-extrabold tracking-tighter mb-2"
          style={{
            background: 'linear-gradient(135deg, #e0e0e8 0%, #4f8aff 50%, #8b5cf6 100%)',
            WebkitBackgroundClip: 'text',
            WebkitTextFillColor: 'transparent',
          }}
        >
          PANACEA
        </h1>
        <p className="text-[var(--color-text-muted)] text-sm sm:text-base font-mono h-6">
          {typedText}
          <span className="animate-pulse">|</span>
        </p>
      </div>

      {/* Stats row */}
      <div
        className={`flex gap-8 mb-10 transition-all duration-500
          ${statsVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}`}
      >
        <Stat value={nSatellites > 0 ? `${(nSatellites / 1000).toFixed(0)}K+` : '...'} label="Objects Tracked" />
        <Stat value="670" label="Conjunction Pairs" />
        <Stat value="90%" label="Conformal Coverage" />
        <Stat value="98.7%" label="Ensemble Recall" />
      </div>

      {/* Feature grid */}
      <div
        className={`grid grid-cols-2 sm:grid-cols-4 gap-3 max-w-3xl mb-10 transition-all duration-500 delay-200
          ${statsVisible ? 'opacity-100 translate-y-0' : 'opacity-0 translate-y-4'}`}
      >
        {FEATURES.map((f, i) => (
          <div
            key={i}
            className="rounded-xl border border-[var(--color-border)] bg-[var(--color-surface)]/60 backdrop-blur-sm
              p-3 text-center hover:border-[var(--color-accent)]/50 transition-colors"
          >
            <div className="text-2xl mb-1">{f.icon}</div>
            <div className="text-[11px] font-semibold text-[var(--color-text)] mb-0.5">{f.label}</div>
            <div className="text-[9px] text-[var(--color-text-muted)] leading-tight">{f.desc}</div>
          </div>
        ))}
      </div>

      {/* Enter button */}
      <button
        onClick={handleEnter}
        disabled={loading}
        className={`group relative px-8 py-3 rounded-full font-semibold text-sm tracking-wide
          transition-all duration-300 ${statsVisible ? 'opacity-100 scale-100' : 'opacity-0 scale-95'}
          ${loading
            ? 'bg-[var(--color-surface-2)] text-[var(--color-text-muted)] cursor-wait'
            : 'bg-[var(--color-accent)] text-white hover:shadow-lg hover:shadow-[var(--color-accent)]/25 hover:scale-105 active:scale-100'
          }`}
      >
        {loading ? 'Loading orbital data...' : 'Launch Dashboard'}
        {!loading && (
          <span className="ml-2 inline-block group-hover:translate-x-1 transition-transform">&rarr;</span>
        )}
      </button>

      {/* Credit line */}
      <p className="mt-6 text-[10px] text-[var(--color-text-muted)]/60">
        AIPI 540 &middot; Duke University &middot; Built with Space-Track CDMs + CelesTrak TLEs
      </p>
    </div>
  );
}

function Stat({ value, label }: { value: string; label: string }) {
  return (
    <div className="text-center">
      <div className="text-xl sm:text-2xl font-bold text-[var(--color-text)]">{value}</div>
      <div className="text-[10px] text-[var(--color-text-muted)] tracking-wide uppercase">{label}</div>
    </div>
  );
}
