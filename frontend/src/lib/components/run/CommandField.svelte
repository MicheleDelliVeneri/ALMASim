<!--
	One parameter of a command as a form field, by its catalogue type: a
	checkbox, a select, a number with its range, a path with the folder
	browser, a list (one value per line) or text. Its flag is shown too, so the
	form reads like the command line it builds.
-->
<script lang="ts">
	import type { FieldValue, Param } from '#lib/utils/commandForm.js';
	import PathPicker from '#lib/components/shared/PathPicker.svelte';

	interface Props {
		param: Param;
		value: FieldValue;
	}

	let { param, value = $bindable() }: Props = $props();

	const fieldId = $props.id();
	const label = $derived(
		param.name.replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase()) +
			(param.required ? ' *' : '')
	);
	const flag = $derived(param.kind === 'option' ? param.flags[0] : param.name.toUpperCase());
	// Values are kept as text, whatever the input gives (a number input gives numbers).
	const text = $derived(typeof value === 'string' ? value : '');
	const setText = (next: string | number | null | undefined) => {
		value = next == null ? '' : String(next);
	};
	// A list is edited as lines; the text keeps a line being typed.
	let linesText = $state(Array.isArray(value) ? value.join('\n') : '');
	const lines = $derived(linesText);
	const setLines = (next: string) => {
		linesText = next;
		value = next
			.split('\n')
			.map((line) => line.trim())
			.filter(Boolean);
	};
</script>

{#if param.type === 'boolean'}
	<div class="flex items-start gap-2.5">
		<input
			id={fieldId}
			type="checkbox"
			class="mt-1 h-4 w-4"
			checked={value === true}
			onchange={(e) => (value = e.currentTarget.checked)}
		/>
		<label for={fieldId} class="flex min-w-0 flex-col">
			<span class="text-sm font-medium text-fg"
				>{label}
				<code class="ml-1 inline-block text-xs font-normal whitespace-nowrap text-fg-dim"
					>{flag}</code
				></span
			>
			{#if param.help}<span class="text-xs text-fg-muted">{param.help}</span>{/if}
		</label>
	</div>
{:else if param.type === 'path' && !param.multiple}
	<PathPicker
		bind:value={() => text, setText}
		{label}
		{flag}
		kind={param.path_kind ?? 'path'}
		required={param.required}
		hint={param.help}
	/>
{:else}
	<div class="flex flex-col gap-1">
		<label for={fieldId} class="text-sm font-medium text-fg"
			>{label}
			<code class="ml-1 inline-block text-xs font-normal whitespace-nowrap text-fg-dim">{flag}</code
			></label
		>
		{#if param.multiple}
			<textarea
				id={fieldId}
				rows="2"
				class="rounded-md border px-3 py-2 font-mono text-sm"
				placeholder="One per line"
				bind:value={() => lines, setLines}></textarea>
		{:else if param.type === 'choice'}
			<select
				id={fieldId}
				class="rounded-md border px-3 py-2 text-sm"
				bind:value={() => text, setText}
			>
				{#if !param.required && param.default == null}<option value="">—</option>{/if}
				{#each param.choices ?? [] as choice (choice)}
					<option value={choice}>{choice}</option>
				{/each}
			</select>
		{:else if param.type === 'integer' || param.type === 'number'}
			<input
				id={fieldId}
				type="number"
				class="rounded-md border px-3 py-2 text-sm"
				min={param.min ?? undefined}
				max={param.max ?? undefined}
				step={param.type === 'integer' ? 1 : 'any'}
				required={param.required}
				placeholder={param.default == null ? '' : `default ${param.default}`}
				bind:value={() => text, setText}
			/>
		{:else}
			<input
				id={fieldId}
				type="text"
				class="rounded-md border px-3 py-2 text-sm"
				required={param.required}
				spellcheck="false"
				bind:value={() => text, setText}
			/>
		{/if}
		{#if param.help}<p class="m-0 text-xs text-fg-muted">{param.help}</p>{/if}
	</div>
{/if}
