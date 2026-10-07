# 01 — TECH STACK & DESIGN TOKENS SPECIFICATION

## Technology Stack

```json
{
  "framework": "Next.js 15 (App Router, React 19)",
  "language": "TypeScript (strict: true)",
  "styling": "Tailwind CSS v4",
  "component_library": "shadcn/ui (Radix UI headless primitives)",
  "icons": "Lucide React",
  "data_fetching": "@tanstack/react-query v5",
  "data_tables": "@tanstack/react-table v8",
  "graph_canvas": "@xyflow/react (React Flow)",
  "charts": "recharts",
  "animations": "framer-motion",
  "validation": "zod + react-hook-form",
  "http_client": "axios"
}
```

## Design System Tokens

### Color Palette (Dark Console Base)
```css
:root {
  --background: #0B0F14;
  --surface: #111821;
  --surface-elevated: #151D27;
  --border: #202A35;
  --border-subtle: rgba(255, 255, 255, 0.08);

  --text-primary: #E8EDF3;
  --text-secondary: #8995A3;
  --text-muted: #596574;

  /* Semantic Decision Tokens */
  --decision-allow: #22C55E;   /* Green */
  --decision-review: #F59E0B;  /* Amber */
  --decision-hold: #F97316;    /* Orange */
  --decision-block: #EF4444;   /* Red */
  --intelligence-blue: #3B82F6;
  --graph-purple: #8B5CF6;
  --device-cyan: #06B6D4;
}
```

### Typography Hierarchy
- **Brand / Headers:** Geist Sans / Inter (`font-semibold tracking-tight`)
- **Metric Labels:** Small uppercase (`text-xs font-medium uppercase tracking-wider text-muted`)
- **Numbers, Scores, Hashes, Latency:** JetBrains Mono (`font-mono tabular-nums`)
