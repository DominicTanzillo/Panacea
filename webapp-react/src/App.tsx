import { useState, useCallback, Suspense, Component } from 'react';
import type { ReactNode, ErrorInfo } from 'react';
import { Globe } from './components/Globe';
import { Header, PanaceaLogo } from './components/Header';
import { InfoPanel } from './components/InfoPanel';
import { StatusBar } from './components/StatusBar';
import { SearchFilter } from './components/SearchFilter';
import { ConjunctionAlerts } from './components/ConjunctionAlerts';
import { RiskDashboard } from './components/RiskDashboard';
import { CDMForecast } from './components/CDMForecast';
import { AboutPage } from './components/AboutPage';
import { LandingOverlay } from './components/LandingOverlay';
import { ModelZooPage } from './components/ModelZooPage';
import { useSatellites } from './hooks/useSatellites';
import { useApi } from './hooks/useApi';
import type { SatellitePosition, ProjectedPair } from './lib/types';

export type OverlayView = 'alerts' | 'forecast' | 'dashboard' | 'models' | 'about' | null;

// Error boundary to catch Three.js / WebGL crashes
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
        <div className="w-full h-full flex flex-col items-center justify-center bg-[var(--color-bg)]">
          <p className="text-lg font-semibold text-[var(--color-risk-red)] mb-2">WebGL Render Error</p>
          <p className="text-sm text-[var(--color-text-muted)] max-w-md text-center">
            {this.state.error?.message || 'The 3D scene encountered an error.'}
          </p>
          <button
            className="mt-4 px-4 py-2 bg-[var(--color-accent)] text-white text-sm"
            style={{ borderRadius: 6 }}
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
    <div className="w-full h-full flex flex-col items-center justify-center bg-[var(--color-bg)]">
      <div className="mb-4 animate-pulse">
        <PanaceaLogo size={48} />
      </div>
      <p className="text-sm text-[var(--color-text-muted)]">
        Loading orbital data...
      </p>
    </div>
  );
}

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

  const handleProjection = useCallback((pair: ProjectedPair | null) => {
    setProjectedPair(pair);
    if (pair) setSelectedSatellite(null);
  }, []);

  const closeOverlay = useCallback(() => {
    setActiveOverlay(null);
    setProjectedPair(null);
  }, []);

  const navigate = useCallback((view: OverlayView) => {
    if (activeOverlay === view) {
      closeOverlay();
    } else {
      setActiveOverlay(view);
      setProjectedPair(null);
    }
  }, [activeOverlay, closeOverlay]);

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

      {/* Borders toggle — small button bottom-left */}
      <button
        onClick={() => setShowBorders(!showBorders)}
        className={`absolute bottom-12 left-4 z-10 px-3 py-1.5 text-xs transition-colors border
          ${showBorders
            ? 'border-[var(--color-accent)] text-[var(--color-text)] bg-[var(--color-accent)]/10'
            : 'border-[var(--color-border)] text-[var(--color-text-dim)] bg-[var(--color-surface)]/60 hover:text-[var(--color-text-muted)]'
          }`}
        style={{ borderRadius: 6, backdropFilter: 'blur(8px)' }}
      >
        Borders
      </button>

      {/* Info panel for selected satellite */}
      {!projectedPair && (
        <InfoPanel
          satellite={selectedSatellite}
          onClose={() => setSelectedSatellite(null)}
        />
      )}

      {/* Overlay views — only one at a time */}
      <ConjunctionAlerts
        pairs={screeningPairs}
        tles={allTLEs}
        visible={activeOverlay === 'alerts'}
        onClose={closeOverlay}
        onProjection={handleProjection}
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
