<script lang="ts">
	import { toasts, type ToastTone } from '#lib/toasts.svelte.js';

	const TONE: Record<ToastTone, string> = {
		info: 'border-l-navy',
		ok: 'border-l-ok',
		warn: 'border-l-warn',
		danger: 'border-l-danger'
	};
</script>

<div
	class="pointer-events-none fixed right-4 bottom-4 z-50 flex w-[min(24rem,calc(100%-2rem))] flex-col gap-2"
	aria-live="polite"
	aria-relevant="additions"
>
	{#each toasts.items as toast (toast.id)}
		<div
			class="pointer-events-auto flex items-start gap-3 rounded-md border border-l-4 border-line bg-surface px-4 py-3 text-sm text-fg shadow-overlay {TONE[
				toast.tone
			]}"
			role={toast.tone === 'danger' ? 'alert' : 'status'}
		>
			<p class="min-w-0 flex-1 break-words">{toast.message}</p>
			<button
				type="button"
				class="-m-1 rounded p-1 text-fg-dim hover:bg-hover hover:text-fg"
				aria-label="Dismiss"
				onclick={() => toasts.dismiss(toast.id)}>✕</button
			>
		</div>
	{/each}
</div>
