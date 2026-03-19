import { useState, useEffect } from 'react';
import { PanaceaLogo } from './Header';

interface LandingOverlayProps {
  loading: boolean;
  nSatellites: number;
  nPairs: number;
  onEnter: () => void;
  onExploreModels?: () => void;
}

export function LandingOverlay({ loading, nSatellites, onEnter, onExploreModels }: LandingOverlayProps) {
  const [show, setShow] = useState(true);
  const [fadeOut, setFadeOut] = useState(false);
  const [phase, setPhase] = useState(0); // 0=title, 1=tagline, 2=stats, 3=cta

  useEffect(() => {
    if (!show) return;
    const t1 = setTimeout(() => setPhase(1), 600);
    const t2 = setTimeout(() => setPhase(2), 1200);
    const t3 = setTimeout(() => setPhase(3), 1800);
    return () => { clearTimeout(t1); clearTimeout(t2); clearTimeout(t3); };
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
        zIndex: 500,
        background: 'linear-gradient(180deg, rgba(8,8,12,0.92) 0%, rgba(8,8,12,0.85) 40%, rgba(8,8,12,0.95) 100%)',
        backdropFilter: 'blur(4px)',
      }}
    >
      {/* Subtle ambient glow */}
      <div className="absolute inset-0 pointer-events-none" style={{
        background: 'radial-gradient(ellipse 800px 500px at 50% 38%, rgba(59,130,246,0.05) 0%, transparent 100%)',
      }} />

      <div className="relative max-w-xl w-full px-8">
        {/* Logo + Title */}
        <div className="text-center" style={{ marginBottom: 32 }}>
          <div style={{ display: 'inline-block', marginBottom: 20 }}>
            <PanaceaLogo size={48} />
          </div>
          <h1
            style={{
              fontSize: 56,
              fontWeight: 800,
              letterSpacing: '-0.03em',
              lineHeight: 1.05,
              background: 'linear-gradient(135deg, #e8e8f0 30%, #3b82f6 100%)',
              WebkitBackgroundClip: 'text',
              WebkitTextFillColor: 'transparent',
              marginBottom: 12,
            }}
          >
            PANACEA
          </h1>
          <p style={{
            fontSize: 13,
            color: '#55556a',
            letterSpacing: '0.08em',
            textTransform: 'uppercase',
            fontWeight: 500,
          }}>
            Predictive Assessment Network for Automated Conjunction Evaluation & Avoidance
          </p>
        </div>

        {/* Tagline — the hook */}
        <div
          className="text-center"
          style={{
            marginBottom: 40,
            opacity: phase >= 1 ? 1 : 0,
            transform: phase >= 1 ? 'translateY(0)' : 'translateY(12px)',
            transition: 'all 600ms ease-out',
          }}
        >
          <p style={{
            fontSize: 20,
            fontWeight: 500,
            color: '#e8e8f0',
            lineHeight: 1.5,
            letterSpacing: '-0.01em',
          }}>
            Every 5 days, objects collide in orbit.<br />
            Each collision creates thousands of new fragments.<br />
            <span style={{ color: '#3b82f6' }}>We predict the next one.</span>
          </p>
        </div>

        {/* Stats strip */}
        <div
          style={{
            display: 'flex',
            justifyContent: 'center',
            gap: 32,
            marginBottom: 40,
            opacity: phase >= 2 ? 1 : 0,
            transform: phase >= 2 ? 'translateY(0)' : 'translateY(12px)',
            transition: 'all 600ms ease-out',
          }}
        >
          <StatBlock value={nSatellites > 0 ? `${(nSatellites / 1000).toFixed(0)}K` : '--'} label="Objects Tracked" />
          <Divider />
          <StatBlock value="670" label="CDM Pairs" />
          <Divider />
          <StatBlock value="6" label="ML Models" />
          <Divider />
          <StatBlock value="98.7%" label="Recall" />
        </div>

        {/* CTA */}
        <div
          className="text-center"
          style={{
            opacity: phase >= 3 ? 1 : 0,
            transform: phase >= 3 ? 'translateY(0)' : 'translateY(12px)',
            transition: 'all 500ms ease-out',
          }}
        >
          <div style={{ display: 'flex', justifyContent: 'center', gap: 12, marginBottom: 32 }}>
            <button
              onClick={handleEnter}
              disabled={loading}
              style={{
                padding: '12px 32px',
                fontSize: 14,
                fontWeight: 600,
                fontFamily: 'inherit',
                letterSpacing: '-0.01em',
                color: '#fff',
                background: loading ? '#1a1a24' : 'linear-gradient(135deg, #3b82f6, #2563eb)',
                border: 'none',
                borderRadius: 8,
                cursor: loading ? 'wait' : 'pointer',
                transition: 'all 200ms',
                boxShadow: loading ? 'none' : '0 4px 16px rgba(59,130,246,0.25)',
              }}
              onMouseEnter={e => { if (!loading) (e.currentTarget).style.boxShadow = '0 6px 24px rgba(59,130,246,0.35)'; }}
              onMouseLeave={e => { if (!loading) (e.currentTarget).style.boxShadow = '0 4px 16px rgba(59,130,246,0.25)'; }}
            >
              {loading ? 'Loading orbital data...' : 'Enter Dashboard \u2192'}
            </button>

            {onExploreModels && (
              <button
                onClick={onExploreModels}
                style={{
                  padding: '12px 24px',
                  fontSize: 14,
                  fontWeight: 500,
                  fontFamily: 'inherit',
                  color: '#7c7c96',
                  background: 'transparent',
                  border: '1px solid #2a2a3a',
                  borderRadius: 8,
                  cursor: 'pointer',
                  transition: 'all 200ms',
                }}
                onMouseEnter={e => {
                  (e.currentTarget).style.color = '#e8e8f0';
                  (e.currentTarget).style.borderColor = '#55556a';
                }}
                onMouseLeave={e => {
                  (e.currentTarget).style.color = '#7c7c96';
                  (e.currentTarget).style.borderColor = '#2a2a3a';
                }}
              >
                Explore Models
              </button>
            )}
          </div>

          <p style={{
            fontSize: 12,
            color: '#55556a',
            letterSpacing: '0.06em',
            textTransform: 'uppercase',
            fontWeight: 500,
          }}>
            AIPI 540 &middot; Duke University &middot; Space-Track CDMs + CelesTrak TLEs
          </p>
        </div>
      </div>
    </div>
  );
}

function StatBlock({ value, label }: { value: string; label: string }) {
  return (
    <div className="text-center" style={{ minWidth: 72 }}>
      <div style={{
        fontSize: 28,
        fontWeight: 700,
        color: '#e8e8f0',
        fontFamily: "'JetBrains Mono', monospace",
        letterSpacing: '-0.02em',
        lineHeight: 1.2,
      }}>
        {value}
      </div>
      <div style={{
        fontSize: 12,
        color: '#55556a',
        textTransform: 'uppercase',
        letterSpacing: '0.06em',
        fontWeight: 500,
        marginTop: 4,
      }}>
        {label}
      </div>
    </div>
  );
}

function Divider() {
  return <div style={{ width: 1, background: '#2a2a3a', alignSelf: 'stretch' }} />;
}
