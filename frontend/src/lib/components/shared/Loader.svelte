<script lang="ts">
	interface Props {
		message?: string;
	}

	let { message = 'Contacting ALMA TAP...' }: Props = $props();
</script>

<div class="fixed inset-0 z-40 flex flex-col items-center justify-center bg-white/90 text-fg-muted">
	<div
		class="h-12 w-12 animate-spin rounded-full border-4 border-accent-line border-t-blue-600"
	></div>
	<p class="mt-4 text-base font-medium">{message}</p>
</div>
