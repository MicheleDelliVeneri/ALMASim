<script lang="ts">
	import { STATUS_TONE_CLASSES, statusTone } from '#lib/utils/status.js';

	interface Props {
		status: string;
	}

	let { status }: Props = $props();
</script>

<span
	class="inline-flex rounded-full px-2 py-0.5 text-xs font-semibold {STATUS_TONE_CLASSES[
		statusTone(status)
	]}"
>
	{status}
</span>
