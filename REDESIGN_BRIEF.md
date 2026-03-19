# PANACEA Webapp Complete Redesign Brief

## For the next Claude instance: Read this entire document before writing any code.

---

## What This Project Is

PANACEA is a satellite collision prediction system for an AIPI 540 (Duke University) final project. Demo Day is April 21, 2026. It has:

- A **3D globe** (Three.js / react-three-fiber) showing 25,000+ tracked orbital objects
- **8 ML models** that predict collision risk from CDM (Conjunction Data Message) sequences
- A **daily pipeline** that fetches real data from Space-Track and CelesTrak
- A **React webapp** deployed to GitHub Pages at tanzillo.me/Panacea/

The ML models are solid. The webapp design is not. It needs a ground-up visual redesign.

---

## The Core Problem

The webapp was built incrementally by adding features without design discipline. The result:

- **8+ competing accent colors** (purple, orange, teal, pink, lavender — all on screen at once)
- **Text as small as 8-9px** everywhere — unreadable
- **Panels overlap and stack** — 5 toggle buttons can all be open simultaneously
- **Inconsistent spacing** — gap values range from 1px to 16px randomly
- **Inconsistent border-radius** — mix of 4px, 6px, 8px, 12px, 16px, 9999px
- **Information overload** — every panel tries to show everything
- **The header nav items are cramped** — words too close together, feels cheap
- **Nothing breathes** — no generous whitespace anywhere

The globe with satellite blips is the ONE thing that looks good. Everything else needs redesign.

---

## Design Direction (User-Approved)

1. **Full-screen overlay views** (Apple/Stripe pattern) — Globe is home state. Each nav item opens a centered overlay. Only ONE overlay at a time.

2. **Traffic light colors only** — Green (#22c55e) for safe, Amber (#f59e0b) for monitor, Red (#ef4444) for danger. Blue (#3b82f6) for interactive UI elements. Everything else is grayscale. NO purple, NO teal, NO pink, NO orange-gold.

3. **Headline-first information** — Each conjunction pair shows: names + risk badge + one human-readable probability ("1 in 3,125"). Click to expand for details.

4. **Minimal globe by default** — fewer satellites visible, cleaner visual

5. **Landing page stays** — but should be educational and clean, explaining what orbital debris is and why it matters. Should feel like we're monitoring a real situation.

---

## Technical Stack

- React 19 + TypeScript + Vite
- Three.js via @react-three/fiber and @react-three/drei
- Recharts for 2D charts
- Tailwind CSS v4 (but inline styles are fine — consistency matters more than methodology)
- Radix UI (dialog, tabs)
- satellite.js for SGP4 orbital propagation
- Static deploy to GitHub Pages (base: './', no server)
- Data from static JSON files in public/

---

## Architecture (Current — what to preserve)

### DO NOT TOUCH (working correctly):
- `src/components/Globe.tsx` — 3D globe rendering, satellite dots, PairHighlight
- `src/components/SatelliteLayer.tsx` — satellite point rendering
- `src/components/CountryBorders.tsx` — border lines on globe
- `src/hooks/useSatellites.ts` — TLE fetching + SGP4 propagation
- `src/hooks/useApi.ts` — data fetching from static JSON
- `src/lib/types.ts` — TypeScript interfaces (except colors in OBJECT_GROUPS)
- `src/lib/orbital.ts` — SGP4 wrapper
- `src/lib/api.ts` — API client types
- All Python model code in `src/model/`
- All pipeline code in `scripts/`

### REDESIGN COMPLETELY:
- `src/index.css` — design tokens
- `src/App.tsx` — state management, layout
- `src/components/Header.tsx` — navigation bar
- `src/components/StatusBar.tsx` — bottom status
- `src/components/LandingOverlay.tsx` — initial splash screen
- `src/components/Overlay.tsx` — shared overlay wrapper (already exists, needs work)
- `src/components/CDMForecast.tsx` — Pc escalation forecast panel (~700 lines)
- `src/components/ConjunctionAlerts.tsx` — conjunction pair list
- `src/components/AlertDetail.tsx` — expanded pair detail with trajectory
- `src/components/RiskDashboard.tsx` — pipeline stats dashboard
- `src/components/ModelZooPage.tsx` — model showcase with charts
- `src/components/AboutPage.tsx` — about/info page
- `src/components/InfoPanel.tsx` — satellite info sidebar
- `src/components/SearchFilter.tsx` — satellite search

---

## Z-Index Hierarchy (CRITICAL — this broke things before)

The Three.js `<Canvas>` creates its own stacking context. You MUST use high z-index values:

```
Globe canvas wrapper:  z-index: 0    (absolute inset-0)
Status bar:            z-index: 50
Header:                z-index: 100  (must be FIXED, not absolute)
Overlays:              z-index: 200  (MUST return null when !visible)
Landing splash:        z-index: 500
```

**CRITICAL**: Overlay components must `return null` when not visible. If they render invisible DOM at z-200, they block clicks on the header at z-100. This was a bug that took multiple attempts to fix.

---

## State Architecture

Replace the old 5-boolean toggle system with:

```typescript
type OverlayView = 'alerts' | 'forecast' | 'dashboard' | 'models' | 'about' | null;
const [activeOverlay, setActiveOverlay] = useState<OverlayView>(null);
```

Only ONE overlay can be open. Clicking a nav item toggles its overlay. Clicking the same item again (or ESC, or clicking the backdrop) closes it.

Keep separate:
- `showLanding: boolean` — the initial splash (not a nav overlay)
- `showBorders: boolean` — country borders toggle
- `selectedSatellite` — satellite info panel
- `projectedPair` — conjunction visualization on globe

---

## Design System to Implement

### Colors
```
Backgrounds:    #08080c, #111118, #1a1a24, #222230
Borders:        #2a2a3a (primary), #1e1e2c (subtle)
Text:           #e8e8f0 (primary), #7c7c96 (muted), #55556a (dim)
Risk red:       #ef4444
Risk amber:     #f59e0b
Risk green:     #22c55e
UI accent:      #3b82f6
```

### Typography
- MINIMUM 12px anywhere. Nothing smaller. Ever.
- Body text in panels: 14px
- Panel headings: 16-18px
- Hero/landing: 32-48px
- Monospace for numbers: 'JetBrains Mono'

### Spacing
- Use 8px grid: 8, 16, 24, 32, 48px
- No 2px, 4px, or 6px gaps between unrelated elements
- Headers: 24-32px horizontal padding
- Cards: 16-24px padding
- Between cards: 12-16px gap
- Nav items: at LEAST 8px gap, preferably 12-16px

### Border Radius
- Small elements (buttons, badges): 6-8px
- Panels/overlays: 12px
- NO rounded-full on anything except status dots

### The Header Must Feel Spacious
- Height: 56px minimum
- Nav items: 13-14px font, 12-20px horizontal padding EACH
- Gap between nav items: 8-12px
- The words must NOT feel cramped

---

## Data Available in Static JSON

### cdm_forecast.json (60 pairs)
Each pair has: sat1_name, sat2_name, norad IDs, tca, current_pc, forecast_pc, exceedance_probability, miss_distance_km, risk_direction, pc_trend, confidence, action_recommended, n_updates, time_series[], predicted_max_pc, predicted_max_log10_pc, predicted_min_miss_km, lstm_escalation_prob, attention_weights[]

### latest_alerts.json (conjunction pairs from TLE screening)
Each pair has: name_1, name_2, norad IDs, risk_score, altitude_km, miss_estimate_km, trajectory[], trail[], pc, miss_distance_km, source ('cdm'|'tle_screening')

### pipeline_stats.json
Has: finetune_history[], daily_history[], cdm_stats, forecast_model metrics

### cv_results.json
Has: 5-fold CV results for lr, lstm, ensemble models

---

## Probability Display

NEVER show "3.2e-4" to users. Instead:
- Pc >= 1%: show as percentage ("1.2%")
- Pc >= 1e-4: show as ratio ("1 in 3,125")
- Pc < 1e-4: show as ratio ("1 in 100,000")
- Always show scientific notation as a tooltip or secondary text, not primary

---

## Demo Day Flow (What Judges Will See)

1. Landing page loads → globe spinning behind, project title, key stats
2. User clicks "Enter" → globe fills screen, header appears
3. Click "Alerts" → see conjunction pairs ranked by risk
4. Click a pair → see trajectory on globe, Pc evolution chart, AI risk assessment
5. Click "Models" → see all 6 models with charts showing their performance
6. Click "Forecast" → see Pc predictions with uncertainty
7. Click "Pipeline" → see daily automation stats

The experience should feel like walking through a polished product demo, not navigating a debugging dashboard.

---

## Common Mistakes to Avoid

1. Don't use CSS variables in Recharts — they may not resolve in SVG context. Use hex values.
2. Don't render invisible overlays — return null when !visible.
3. Don't use Tailwind's z-50 for overlays — it's not high enough for Three.js canvas.
4. Don't use text smaller than 12px anywhere.
5. Don't use more than 4 colors on screen at once (grayscale + traffic light + blue).
6. Don't put too many metrics in one view — progressive disclosure.
7. Don't forget pointer-events — the globe canvas eats clicks if you don't manage z-index.
8. Deploy locally (`npm run dev`) and check visually BEFORE pushing.

---

## Files Reference

```
webapp-react/
├── src/
│   ├── main.tsx           # Entry point (don't touch)
│   ├── App.tsx            # REDESIGN — layout + state
│   ├── index.css          # REDESIGN — design tokens
│   ├── components/
│   │   ├── Globe.tsx           # DON'T TOUCH
│   │   ├── SatelliteLayer.tsx  # DON'T TOUCH
│   │   ├── CountryBorders.tsx  # DON'T TOUCH
│   │   ├── Header.tsx          # REDESIGN
│   │   ├── StatusBar.tsx       # REDESIGN
│   │   ├── Overlay.tsx         # REDESIGN (shared overlay wrapper)
│   │   ├── LandingOverlay.tsx  # REDESIGN
│   │   ├── CDMForecast.tsx     # REDESIGN (~700 lines, biggest component)
│   │   ├── ConjunctionAlerts.tsx # REDESIGN
│   │   ├── AlertDetail.tsx     # REDESIGN
│   │   ├── RiskDashboard.tsx   # REDESIGN
│   │   ├── ModelZooPage.tsx    # REDESIGN
│   │   ├── AboutPage.tsx       # REDESIGN
│   │   ├── InfoPanel.tsx       # RESTYLE
│   │   └── SearchFilter.tsx    # RESTYLE
│   ├── hooks/
│   │   ├── useSatellites.ts   # DON'T TOUCH
│   │   └── useApi.ts          # DON'T TOUCH
│   └── lib/
│       ├── types.ts           # Update OBJECT_GROUPS colors only
│       ├── api.ts             # DON'T TOUCH
│       ├── orbital.ts         # DON'T TOUCH
│       └── formatPc.ts        # Available utility for probability formatting
├── public/
│   ├── cdm_forecast.json      # 60 pairs with ML predictions
│   ├── latest_alerts.json     # Conjunction alerts
│   ├── pipeline_stats.json    # Pipeline metrics
│   ├── cv_results.json        # Cross-validation results
│   └── latest_tles.json       # TLE fallback data
├── index.html                 # Google Fonts loaded here
├── vite.config.ts             # base: './' for GitHub Pages
└── package.json               # Dependencies
```

---

## Success Criteria

When done, a first-time user should:
1. Immediately understand this is a satellite collision prediction system
2. Navigate to any section with one click
3. Never see overlapping panels or cramped text
4. Feel like the product is polished and professional
5. Be able to understand the ML models without ML expertise
6. See the globe as the hero, with overlays appearing cleanly on top

When a judge sees it, they should think: "This team built something real and polished it."
