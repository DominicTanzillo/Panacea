import { useState, useCallback, useEffect, useMemo, Suspense, Component } from 'react';
import type { ReactNode, ErrorInfo } from 'react';
import { Globe } from './components/Globe';
import { Header, PanaceaLogo } from './components/Header';
import { InfoPanel } from './components/InfoPanel';
import { StatusBar } from './components/StatusBar';
import { SearchFilter } from './components/SearchFilter';
import { ConjunctionAlerts } from './components/ConjunctionAlerts';
import { AlertDetail } from './components/AlertDetail';
import { RiskDashboard } from './components/RiskDashboard';
import { CDMForecast } from './components/CDMForecast';
import { AboutPage } from './components/AboutPage';
import { LandingOverlay } from './components/LandingOverlay';
import { ModelZooPage } from './components/ModelZooPage';
import { useSatellites } from './hooks/useSatellites';
import { useApi } from './hooks/useApi';
import type { SatellitePosition, ProjectedPair } from './lib/types';
import type { ScreeningPair } from './lib/api';

export type OverlayView = 'alerts' | 'forecast' | 'dashboard' | 'models' | 'about' | null;

/* ── Error boundary for WebGL crashes ─────────────────────── */
interface ErrorBoundaryProps { children: ReactNode; fallback?: ReactNode; }
interface ErrorBoundaryState { hasError: boolean; error: Error | null; }

class SceneErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { hasError: false, error: null };
  }
  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { hasError: true, error };
  }
  componentDidCatch(error: Error, info: ErrorInfo) {
    console.error('3D Scene crashed:', error, info);
  }
  render() {
    if (this.state.hasError) {
      return (
        <div className="w-full h-full flex flex-col items-center justify-center" style={{ background: '#08080c' }}>
          <p style={{ fontSize: 16, fontWeight: 600, color: '#ef4444', marginBottom: 8 }}>WebGL Render Error</p>
          <p style={{ fontSize: 13, color: '#7c7c96', maxWidth: 400, textAlign: 'center' }}>
            {this.state.error?.message || 'The 3D scene encountered an error.'}
          </p>
          <button
            style={{ marginTop: 16, padding: '8px 20px', background: '#3b82f6', color: '#fff', fontSize: 13, fontWeight: 500, border: 'none', borderRadius: 6, cursor: 'pointer', fontFamily: 'inherit' }}
            onClick={() => this.setState({ hasError: false, error: null })}
          >
            Retry
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

function LoadingScreen() {
  return (
    <div className="w-full h-full flex flex-col items-center justify-center" style={{ background: '#08080c' }}>
      <div className="mb-4 animate-pulse">
        <PanaceaLogo size={48} />
      </div>
      <p style={{ fontSize: 13, color: '#7c7c96' }}>Loading orbital data...</p>
    </div>
  );
}

/* ── Threat HUD: floating on globe view ───────────────────── */

function formatPcCompact(pc: number): string {
  if (!pc || pc <= 0) return '--';
  if (pc >= 0.01) return `${(pc * 100).toFixed(1)}%`;
  const ratio = Math.round(1 / pc);
  if (ratio >= 1_000_000) return `1 in ${(ratio / 1_000_000).toFixed(0)}M`;
  if (ratio >= 1_000) return `1 in ${(ratio / 1_000).toFixed(0)}K`;
  return `1 in ${ratio.toLocaleString()}`;
}

function formatTCAShort(pair: ScreeningPair): string {
  if (pair.cdm_tca) {
    const h = (new Date(pair.cdm_tca).getTime() - Date.now()) / 3600000;
    if (h < -24) return `${Math.abs(h / 24).toFixed(0)}d ago`;
    if (h < 0) return `${Math.abs(h).toFixed(0)}h ago`;
    if (h < 24) return `in ${h.toFixed(1)}h`;
    return `in ${(h / 24).toFixed(1)}d`;
  }
  if (pair.tca_hours != null) {
    if (pair.tca_hours < 24) return `${pair.tca_hours.toFixed(1)}h`;
    return `${(pair.tca_hours / 24).toFixed(1)}d`;
  }
  return '';
}

function riskColor(pair: ScreeningPair): string {
  if (pair.source === 'cdm' && pair.pc != null) {
    if (pair.pc >= 1e-4) return '#ef4444';
    if (pair.pc >= 1e-7) return '#f59e0b';
    return '#22c55e';
  }
  if (pair.tca_min_distance_km != null) {
    if (pair.tca_min_distance_km < 1) return '#ef4444';
    if (pair.tca_min_distance_km < 5) return '#f59e0b';
    return '#22c55e';
  }
  if (pair.risk_score > 0.40) return '#ef4444';
  if (pair.risk_score > 0.10) return '#f59e0b';
  return '#22c55e';
}

function ThreatHUD({ pairs, onViewAlerts, onSelectPair }: {
  pairs: ScreeningPair[];
  onViewAlerts: () => void;
  onSelectPair: (pair: ScreeningPair) => void;
}) {
  const topPairs = useMemo(() => {
    return [...pairs]
      .sort((a, b) => {
        // CDM pairs with Pc first, then by Pc descending
        const pcA = a.pc ?? 0;
        const pcB = b.pc ?? 0;
        return pcB - pcA;
      })
      .slice(0, 3);
  }, [pairs]);

  if (topPairs.length === 0) return null;

  return (
    <div
      style={{
        position: 'absolute',
        bottom: 80,
        left: 24,
        zIndex: 10,
        width: 300,
        animation: 'slideUp 400ms ease-out',
      }}
    >
      {/* Title */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        marginBottom: 8,
      }}>
        <span style={{
          fontSize: 12,
          fontWeight: 600,
          color: '#55556a',
          textTransform: 'uppercase',
          letterSpacing: '0.06em',
        }}>
          Active Conjunctions
        </span>
        <button
          onClick={onViewAlerts}
          style={{
            fontSize: 12,
            fontWeight: 500,
            color: '#3b82f6',
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            fontFamily: 'inherit',
          }}
        >
          View all &rarr;
        </button>
      </div>

      {/* Cards */}
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {topPairs.map((pair) => {
          const color = riskColor(pair);
          const tca = formatTCAShort(pair);
          return (
            <button
              key={`${pair.norad_1}-${pair.norad_2}`}
              onClick={() => onSelectPair(pair)}
              style={{
                display: 'flex',
                alignItems: 'center',
                gap: 12,
                padding: '10px 14px',
                background: 'rgba(17,17,24,0.90)',
                backdropFilter: 'blur(12px)',
                border: '1px solid #2a2a3a',
                borderRadius: 8,
                cursor: 'pointer',
                textAlign: 'left',
                transition: 'all 150ms',
                fontFamily: 'inherit',
                width: '100%',
              }}
              onMouseEnter={e => { (e.currentTarget).style.borderColor = '#3b82f6'; (e.currentTarget).style.background = 'rgba(17,17,24,0.95)'; }}
              onMouseLeave={e => { (e.currentTarget).style.borderColor = '#2a2a3a'; (e.currentTarget).style.background = 'rgba(17,17,24,0.90)'; }}
            >
              {/* Risk indicator */}
              <div style={{
                width: 4,
                height: 32,
                borderRadius: 2,
                background: color,
                flexShrink: 0,
              }} />

              {/* Names */}
              <div style={{ flex: 1, minWidth: 0 }}>
                <div style={{
                  fontSize: 13,
                  fontWeight: 500,
                  color: '#e8e8f0',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}>
                  {pair.name_1}
                </div>
                <div style={{
                  fontSize: 12,
                  color: '#55556a',
                  overflow: 'hidden',
                  textOverflow: 'ellipsis',
                  whiteSpace: 'nowrap',
                }}>
                  vs {pair.name_2}
                </div>
              </div>

              {/* Metrics */}
              <div style={{ textAlign: 'right', flexShrink: 0 }}>
                {pair.pc != null && pair.pc > 0 && (
                  <div style={{
                    fontSize: 13,
                    fontWeight: 600,
                    fontFamily: "'JetBrains Mono', monospace",
                    color,
                  }}>
                    {formatPcCompact(pair.pc)}
                  </div>
                )}
                {tca && (
                  <div style={{ fontSize: 12, color: '#55556a' }}>
                    TCA {tca}
                  </div>
                )}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
}

/* ── Main App ─────────────────────────────────────────────── */

function App() {
  const {
    satellites, allTLEs, loading, totalTLEs,
    groups, toggleGroup, lastUpdate,
  } = useSatellites();

  const {
    healthy, modelComparison, experimentResults,
    screeningPairs, alertsMeta,
  } = useApi(allTLEs);

  const [selectedSatellite, setSelectedSatellite] = useState<SatellitePosition | null>(null);
  const [projectedPair, setProjectedPair] = useState<ProjectedPair | null>(null);
  const [activeOverlay, setActiveOverlay] = useState<OverlayView>(null);
  const [showLanding, setShowLanding] = useState(true);
  const [showBorders, setShowBorders] = useState(false);
  const [selectedAlertPair, setSelectedAlertPair] = useState<ScreeningPair | null>(null);

  const handleProjection = useCallback((pair: ProjectedPair | null) => {
    setProjectedPair(pair);
    if (pair) setSelectedSatellite(null);
  }, []);

  const closeOverlay = useCallback(() => {
    setActiveOverlay(null);
    setProjectedPair(null);
  }, []);

  const navigate = useCallback((view: OverlayView) => {
    // Close detail panel if open
    if (selectedAlertPair) {
      setSelectedAlertPair(null);
      setProjectedPair(null);
    }
    if (activeOverlay === view) {
      closeOverlay();
    } else {
      setActiveOverlay(view);
      setProjectedPair(null);
    }
  }, [activeOverlay, closeOverlay, selectedAlertPair]);

  // Handle pair selection from alerts list: close overlay, open detail panel
  const handleSelectPair = useCallback((pair: ScreeningPair) => {
    setActiveOverlay(null);
    setSelectedAlertPair(pair);
  }, []);

  // Handle back from detail: re-open alerts overlay
  const handleBackFromDetail = useCallback(() => {
    setSelectedAlertPair(null);
    setProjectedPair(null);
    setActiveOverlay('alerts');
  }, []);

  // Handle pair selection from ThreatHUD
  const handleHUDSelectPair = useCallback((pair: ScreeningPair) => {
    setSelectedAlertPair(pair);
    setSelectedSatellite(null);
  }, []);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      // Don't capture if typing in input
      if (e.target instanceof HTMLInputElement || e.target instanceof HTMLTextAreaElement) return;
      if (showLanding) return;

      if (e.key === 'Escape' && selectedAlertPair) {
        handleBackFromDetail();
        e.preventDefault();
      }
    };
    document.addEventListener('keydown', handler);
    return () => document.removeEventListener('keydown', handler);
  }, [showLanding, selectedAlertPair, handleBackFromDetail]);

  const showThreatHUD = !showLanding && !activeOverlay && !selectedAlertPair && !selectedSatellite;

  return (
    <div className="w-full h-full relative">
      <Header
        activeOverlay={activeOverlay}
        onNavigate={navigate}
        healthy={healthy}
        alertCount={screeningPairs.length}
        cdmAlertCount={alertsMeta.cdmCount}
        dataDate={alertsMeta.dataDate}
      />

      <SearchFilter
        satellites={satellites}
        onSelectSatellite={setSelectedSatellite}
      />

      {/* Globe layer */}
      <div className="absolute inset-0" style={{ zIndex: 0 }}>
        <SceneErrorBoundary>
          <Suspense fallback={<LoadingScreen />}>
            <Globe
              satellites={satellites}
              onSelectSatellite={setSelectedSatellite}
              selectedSatellite={selectedSatellite}
              projectedPair={projectedPair}
              showBorders={showBorders}
            />
          </Suspense>
        </SceneErrorBoundary>
      </div>

      {/* Borders toggle — hide when an overlay panel is open */}
      {!activeOverlay && <button
        onClick={() => setShowBorders(!showBorders)}
        aria-label="Toggle country borders"
        style={{
          position: 'absolute',
          bottom: 48,
          right: 24,
          zIndex: 25,
          padding: '6px 14px',
          fontSize: 12,
          fontWeight: 500,
          fontFamily: 'inherit',
          color: showBorders ? '#e8e8f0' : '#55556a',
          background: showBorders ? 'rgba(59,130,246,0.10)' : 'rgba(17,17,24,0.80)',
          border: showBorders ? '1px solid rgba(59,130,246,0.3)' : '1px solid #2a2a3a',
          borderRadius: 6,
          cursor: 'pointer',
          backdropFilter: 'blur(8px)',
          transition: 'all 150ms',
        }}
      >
        Borders
      </button>}

      {/* Threat HUD — shows on globe view when nothing else is open */}
      {showThreatHUD && (
        <ThreatHUD
          pairs={screeningPairs}
          onViewAlerts={() => navigate('alerts')}
          onSelectPair={handleHUDSelectPair}
        />
      )}

      {/* Info panel for selected satellite */}
      {!projectedPair && !selectedAlertPair && (
        <InfoPanel
          satellite={selectedSatellite}
          onClose={() => setSelectedSatellite(null)}
        />
      )}

      {/* Alert detail — right panel when pair selected */}
      {selectedAlertPair && (
        <AlertDetail
          pair={selectedAlertPair}
          tles={allTLEs}
          onBack={handleBackFromDetail}
          onProjection={handleProjection}
        />
      )}

      {/* Overlay views — only one at a time */}
      <ConjunctionAlerts
        pairs={screeningPairs}
        visible={activeOverlay === 'alerts'}
        onClose={closeOverlay}
        onSelectPair={handleSelectPair}
      />

      <CDMForecast
        visible={activeOverlay === 'forecast'}
        onClose={closeOverlay}
      />

      <RiskDashboard
        modelComparison={modelComparison}
        experimentResults={experimentResults}
        satellites={satellites}
        visible={activeOverlay === 'dashboard'}
        onClose={closeOverlay}
      />

      <ModelZooPage
        visible={activeOverlay === 'models'}
        onClose={closeOverlay}
      />

      <AboutPage
        visible={activeOverlay === 'about'}
        onClose={closeOverlay}
        onNavigate={setActiveOverlay}
      />

      {/* Status bar */}
      <StatusBar
        satellites={satellites}
        loading={loading}
        totalTLEs={totalTLEs}
        groups={groups}
        onToggleGroup={toggleGroup}
        lastUpdate={lastUpdate}
      />

      {/* Landing splash */}
      {showLanding && (
        <LandingOverlay
          loading={loading}
          nSatellites={totalTLEs}
          nPairs={screeningPairs.length}
          onEnter={() => setShowLanding(false)}
          onExploreModels={() => { setShowLanding(false); setActiveOverlay('models'); }}
        />
      )}
    </div>
  );
}

export default App;
