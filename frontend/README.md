# ALMASim Frontend

SolidStart frontend for ALMASim with TypeScript and Tailwind CSS.

## Setup

1. Install dependencies:
```bash
npm install
```

2. Set the backend URL (optional, only needed if not using the default):
```bash
# Dev: build-time fallback (Vite reads this and bakes it in only as a default)
export VITE_API_URL=http://localhost:8000

# Production / Docker: runtime override (read by hooks.server.ts on every
# request and injected into the served HTML — no rebuild needed).
export API_URL=http://backend:8000
```

3. Run development server:
```bash
npm run dev
```

4. Build for production:
```bash
npm run build
```

5. Start production server:
```bash
npm start
```

## Development

- Development server: http://localhost:5173
- API proxy: `/api` requests are proxied to `http://localhost:8000`

## Architecture

- **Routes** (`app/routes/`): SolidStart pages (file-based routing)
- **API Client** (`app/lib/api/`): TypeScript client for backend API
- **Components** (`app/components/`): Reusable Solid components
- **Styling**: Tailwind CSS for modern, responsive design

## Tech Stack

- **SolidStart**: Full-stack framework built on SolidJS
- **SolidJS**: Reactive UI library
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first CSS framework
- **Vite**: Fast build tool
