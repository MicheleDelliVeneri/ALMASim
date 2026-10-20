<script lang="ts">
	interface Props {
		/** Size and colour classes; the spinner draws in currentColor. */
		class?: string;
	}

	let { class: className = 'h-4 w-4' }: Props = $props();
</script>

<svg class="animate-spin {className}" viewBox="0 0 24 24" fill="none" aria-hidden="true">
	<circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4" class="opacity-25" />
	<path fill="currentColor" d="M4 12a8 8 0 018-8v4a4 4 0 00-4 4H4z" class="opacity-75" />
</svg>
