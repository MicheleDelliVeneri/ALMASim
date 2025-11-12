/** Shared loading components */
export const FullScreenLoader = () => (
  <div class="fixed inset-0 z-40 flex flex-col items-center justify-center bg-white/90 text-gray-700">
    <div class="h-12 w-12 animate-spin rounded-full border-4 border-blue-200 border-t-blue-600" />
    <p class="mt-4 text-base font-medium">Contacting ALMA TAP...</p>
  </div>
);

export const SkeletonCell = () => (
  <div class="h-4 w-full animate-pulse rounded bg-gray-200/80" aria-hidden="true" />
);

