import { useState, Suspense, Component } from 'react';
import type { ReactNode, ErrorInfo } from 'react';
import { Globe } from './components/Globe';
import { Header, PanaceaLogo } from './components/Header';
import { InfoPanel } from './components/InfoPanel';
import { StatusBar } from './components/StatusBar';
import { SearchFilter } from './components/SearchFilter';
import { ConjunctionAlerts } from './components/ConjunctionAlerts';
import { RiskDashboard } from './components/RiskDashboard';
import { AboutPage } from './components/AboutPage';
import { useSatellites } from './hooks/useSatellites';
import { useApi } from './hooks/useApi';
import type { SatellitePosition } from './lib/types';

// Error boundary to catch Three.js / WebGL crashes
interface ErrorBoundaryProps {
  children: ReactNode;
  fallback?: ReactNode;
}
interface ErrorBoundaryState {
  hasError: boolean;
  error: Error | null;
}

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
          <p className="text-lg font-semibold text-red-400 mb-2">WebGL Render Error</p>
          <p className="text-sm text-[var(--color-text-muted)] max-w-md text-center">
            {this.state.error?.message || 'The 3D scene encountered an error.'}
          </p>
          <button
            className="mt-4 px-4 py-2 rounded-lg bg-[var(--color-accent)] text-white text-sm"
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
        Loading orbital data from CelesTrak...
      </p>
      <div className="mt-3 w-48 h-1 rounded-full bg-[var(--color-surface-2)] overflow-hidden">
        <div className="h-full bg-[var(--color-accent)] rounded-full animate-pulse" style={{ width: '60%' }} />
      </div>
    </div>
  );
}

function App() {
  const {
    satellites,
    allTLEs,
    loading,
    totalTLEs,
    groups,
    toggleGroup,
    lastUpdate,
  } = useSatellites();

  const {
    healthy,
    modelComparison,
    experimentResults,
    screeningPairs,
  } = useApi(allTLEs);

  const [selectedSatellite, setSelectedSatellite] = useState<SatellitePosition | null>(null);
  const [showAlerts, setShowAlerts] = useState(false);
  const [showDashboard, setShowDashboard] = useState(false);
  const [showBorders, setShowBorders] = useState(false);
  const [showAbout, setShowAbout] = useState(false);

  return (
    <div className="w-full h-full relative">
      <Header
        healthy={healthy}
        showBorders={showBorders}
        onToggleBorders={() => setShowBorders(!showBorders)}
        showAlerts={showAlerts}
        onToggleAlerts={() => { setShowAlerts(!showAlerts); setShowDashboard(false); }}
        showDashboard={showDashboard}
        onToggleDashboard={() => { setShowDashboard(!showDashboard); setShowAlerts(false); }}
        onShowAbout={() => setShowAbout(true)}
        alertCount={screeningPairs.length}
      />

      <SearchFilter
        satellites={satellites}
        onSelectSatellite={setSelectedSatellite}
      />

      <SceneErrorBoundary>
        <Suspense fallback={<LoadingScreen />}>
          <Globe
            satellites={satellites}
            onSelectSatellite={setSelectedSatellite}
            selectedSatellite={selectedSatellite}
            showBorders={showBorders}
          />
        </Suspense>
      </SceneErrorBoundary>

      <InfoPanel
        satellite={selectedSatellite}
        onClose={() => setSelectedSatellite(null)}
      />

      <ConjunctionAlerts
        pairs={screeningPairs}
        visible={showAlerts}
        onClose={() => setShowAlerts(false)}
      />

      <RiskDashboard
        modelComparison={modelComparison}
        experimentResults={experimentResults}
        satellites={satellites}
        visible={showDashboard}
        onClose={() => setShowDashboard(false)}
      />

      <StatusBar
        satellites={satellites}
        loading={loading}
        totalTLEs={totalTLEs}
        groups={groups}
        onToggleGroup={toggleGroup}
        lastUpdate={lastUpdate}
      />

      <AboutPage visible={showAbout} onClose={() => setShowAbout(false)} />
    </div>
  );
}

export default App;
