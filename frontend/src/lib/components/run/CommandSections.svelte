<!-- A command's fields in their sections (inputs, then the help panels), each foldable. -->
<script lang="ts">
	import { sections, type Command, type Values } from '#lib/utils/commandForm.js';
	import CommandField from './CommandField.svelte';

	interface Props {
		command: Command;
		values: Values;
		/** Sections to leave out (the page sets those options itself). */
		omit?: string[];
		/** Parameters to leave out, by name (the page sets them itself). */
		hide?: string[];
	}

	let { command, values = $bindable(), omit = [], hide = [] }: Props = $props();

	const shown = $derived(
		sections(command)
			.filter((s) => !omit.includes(s.title))
			.map((s) => ({ ...s, params: s.params.filter((p) => !hide.includes(p.name)) }))
			.filter((s) => s.params.length > 0)
	);
</script>

{#each shown as section (section.title)}
	<details class="alma-panel group rounded-lg" open={!section.folded}>
		<summary
			class="flex cursor-pointer list-none items-center gap-2 px-5 py-3 text-base font-semibold text-navy"
		>
			<span class="inline-block transition-transform group-open:rotate-90" aria-hidden="true"
				>›</span
			>
			{section.title}
			<span class="text-xs font-normal text-fg-dim">{section.params.length}</span>
			{#if section.folded}
				<span class="ml-auto text-xs font-normal text-fg-dim"
					>{section.title === 'Cluster' ? 'filled in by the console' : 'rarely needed'}</span
				>
			{/if}
		</summary>
		<div class="grid gap-x-6 gap-y-4 border-t border-line px-5 py-4 md:grid-cols-2">
			{#each section.params as param (param.name)}
				<div class={param.type === 'path' || param.multiple ? 'md:col-span-2' : ''}>
					<CommandField {param} bind:value={values[param.name]} />
				</div>
			{/each}
		</div>
	</details>
{/each}
