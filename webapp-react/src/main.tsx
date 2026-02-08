import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'

// StrictMode intentionally removed — it double-mounts components in dev,
// which creates two WebGL contexts and causes GPU context loss
createRoot(document.getElementById('root')!).render(<App />)
