import { ErrorBoundary, For, Suspense, type ParentComponent } from "solid-js";
import { A, Router } from "@solidjs/router";
import { FileRoutes } from "@solidjs/start/router";
import "./app.css";

const navItems = [
  { href: "/", label: "Overview" },
  { href: "/simulations", label: "Simulations" },
  { href: "/metadata", label: "Metadata" },
  { href: "/visualizer", label: "Visualizer" },
];

const AppShell: ParentComponent = (props) => (
  <div class="flex min-h-screen bg-gray-50 text-gray-900">
    <aside class="hidden w-64 flex-col bg-slate-900 text-white shadow-lg md:flex">
      <div class="px-6 py-5 text-2xl font-semibold tracking-tight">ALMASim</div>
      <nav class="flex-1 px-4 py-4">
        <For each={navItems}>
          {(item) => (
            <A
              href={item.href}
              class="mb-2 block rounded-md px-3 py-2 text-sm font-medium text-slate-200 transition hover:bg-white/10 hover:text-white"
              activeClass="bg-white text-slate-900"
            >
              {item.label}
            </A>
          )}
        </For>
      </nav>
      <div class="px-6 py-4 text-xs text-slate-400">© {new Date().getFullYear()} ALMASim</div>
    </aside>
    <main class="flex-1 overflow-y-auto">
      <div class="mx-auto max-w-6xl px-4 py-8">{props.children}</div>
    </main>
  </div>
);

export default function App() {
  return (
    <Suspense>
      <ErrorBoundary>
        <Router root={AppShell}>
          <FileRoutes />
        </Router>
      </ErrorBoundary>
    </Suspense>
  );
}
