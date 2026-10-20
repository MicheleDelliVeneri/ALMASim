<script lang="ts">
	import type { Snippet } from 'svelte';

	interface Props {
		title: string;
		onClose: () => void;
		/** Tailwind max-width class of the dialog. */
		width?: string;
		/** Stacking class; a modal opened over another one needs a higher layer. */
		layer?: string;
		children: Snippet;
		footer?: Snippet;
	}

	let { title, onClose, width = 'max-w-lg', layer = 'z-50', children, footer }: Props = $props();

	function onKeydown(event: KeyboardEvent) {
		if (event.key === 'Escape') onClose();
	}
</script>

<svelte:window onkeydown={onKeydown} />

<div
	class="fixed inset-0 {layer} flex items-center justify-center bg-black/50 px-4"
	role="dialog"
	aria-modal="true"
	aria-label={title}
>
	<div class="w-full {width} rounded-lg border border-line bg-surface shadow-panel">
		<header class="flex items-center justify-between border-b border-line px-5 py-3">
			<h3 class="text-sm font-semibold text-fg">{title}</h3>
			<button type="button" class="text-fg-dim hover:text-fg" aria-label="Close" onclick={onClose}>
				✕
			</button>
		</header>
		<div class="px-5 py-4">
			{@render children()}
		</div>
		{#if footer}
			<footer class="flex items-center justify-end gap-3 border-t border-line px-5 py-3">
				{@render footer()}
			</footer>
		{/if}
	</div>
</div>
