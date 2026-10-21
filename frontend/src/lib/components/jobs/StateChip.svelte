<script lang="ts">
	import type { JobState } from '#lib/api/v2/client.js';
	import { STATE_CLASSES, STATE_LABEL } from '#lib/utils/jobs.js';

	interface Props {
		state: JobState;
		/** Shown after the state, e.g. "3 h 12 m". */
		detail?: string;
	}

	let { state, detail }: Props = $props();
</script>

<span
	class="inline-flex shrink-0 items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs font-semibold whitespace-nowrap {STATE_CLASSES[
		state
	].chip}"
>
	<span class="h-1.5 w-1.5 rounded-full {STATE_CLASSES[state].dot}" aria-hidden="true"></span>
	{STATE_LABEL[state]}{#if detail}<span class="font-normal">· {detail}</span>{/if}
</span>
