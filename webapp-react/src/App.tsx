import { useState, Suspense, Component } from 'react';
import type { ReactNode, ErrorInfo } from 'react';
import { Globe } from './components/Globe';
import { Header } from './components/Header';
import { InfoPanel } from './components/InfoPanel';
import { StatusBar } from './components/StatusBar';
import { useSatellites } from './hooks/useSatellites';
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
      <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-[var(--color-accent)] to-[#8b5cf6] flex items-center justify-center text-xl font-bold mb-4 animate-pulse">
        P
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
    loading,
    totalTLEs,
    groups,
    toggleGroup,
    lastUpdate,
  } = useSatellites();

  const [selectedSatellite, setSelectedSatellite] = useState<SatellitePosition | null>(null);

  return (
    <div className="w-full h-full relative">
      <Header />

      <SceneErrorBoundary>
        <Suspense fallback={<LoadingScreen />}>
          <Globe
            satellites={satellites}
            onSelectSatellite={setSelectedSatellite}
            selectedSatellite={selectedSatellite}
          />
        </Suspense>
      </SceneErrorBoundary>

      <InfoPanel
        satellite={selectedSatellite}
        onClose={() => setSelectedSatellite(null)}
      />

      <StatusBar
        satellites={satellites}
        loading={loading}
        totalTLEs={totalTLEs}
        groups={groups}
        onToggleGroup={toggleGroup}
        lastUpdate={lastUpdate}
      />
    </div>
  );
}

export default App;
