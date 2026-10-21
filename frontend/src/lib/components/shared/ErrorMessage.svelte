<!-- The one way pages show an error: what failed, why, and a retry when there is one. -->
<script lang="ts">
	import { errorMessage } from '#lib/utils/errors.js';

	interface Props {
		error: unknown;
		/** What failed, e.g. "Could not load the jobs". */
		title?: string;
		onRetry?: () => void;
	}

	let { error, title, onRetry }: Props = $props();
</script>

<div
	class="flex items-start gap-3 rounded-md border border-danger/40 bg-danger/5 px-4 py-3 text-sm"
	role="alert"
>
	<div class="min-w-0 flex-1">
		{#if title}<p class="font-semibold text-danger">{title}</p>{/if}
		<p class="break-words text-fg">{errorMessage(error)}</p>
	</div>
	{#if onRetry}
		<button type="button" class="btn-secondary shrink-0" onclick={onRetry}>Retry</button>
	{/if}
</div>
