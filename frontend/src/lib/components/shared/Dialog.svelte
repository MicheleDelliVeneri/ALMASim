<!--
	A modal dialog on the native <dialog>: the browser keeps focus inside it,
	makes the page behind it inert, stacks it above everything (so a dialog
	opened from a dialog needs no z-index), and closes it on Escape. Render it
	while it should be open; it opens when mounted and calls `onClose` when the
	person dismisses it (Escape, the close button, or a click outside).
-->
<script lang="ts">
	import type { Snippet } from 'svelte';

	interface Props {
		title: string;
		onClose: () => void;
		/** Tailwind max-width class of the dialog. */
		width?: string;
		children: Snippet;
		footer?: Snippet;
	}

	let { title, onClose, width = 'max-w-lg', children, footer }: Props = $props();

	const titleId = $props.id();
	let dialog: HTMLDialogElement;

	$effect(() => {
		dialog.showModal();
		return () => dialog.close();
	});

	function onCancel(event: Event) {
		// Escape: let the owner decide (it unmounts us), rather than closing behind its back.
		event.preventDefault();
		onClose();
	}

	function onBackdrop(event: MouseEvent) {
		if (event.target === dialog) onClose();
	}
</script>

<dialog
	bind:this={dialog}
	aria-labelledby={titleId}
	class="m-auto w-[calc(100%-2rem)] {width} rounded-lg border border-line bg-surface p-0 text-fg shadow-overlay backdrop:bg-navy/40"
	oncancel={onCancel}
	onclick={onBackdrop}
>
	<header class="flex items-center justify-between border-b border-line px-5 py-3">
		<h2 id={titleId} class="text-base font-semibold text-fg">{title}</h2>
		<button
			type="button"
			class="rounded p-1 text-fg-dim hover:bg-hover hover:text-fg"
			aria-label="Close"
			onclick={onClose}
		>
			✕
		</button>
	</header>
	<div class="max-h-[70vh] overflow-y-auto px-5 py-4">
		{@render children()}
	</div>
	{#if footer}
		<footer class="flex items-center justify-end gap-3 border-t border-line px-5 py-3">
			{@render footer()}
		</footer>
	{/if}
</dialog>
