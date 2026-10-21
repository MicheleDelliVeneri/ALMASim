<script lang="ts">
	import Dialog from '#lib/components/shared/Dialog.svelte';

	interface Props {
		open: boolean;
		saving: boolean;
		name: string;
		description: string;
		onCancel: () => void;
		onSave: () => void;
	}

	let {
		open,
		saving,
		name = $bindable(),
		description = $bindable(),
		onCancel,
		onSave
	}: Props = $props();
</script>

{#if open}
	<Dialog title="Save Query as Preset" width="max-w-sm" onClose={onCancel}>
		<div class="space-y-3">
			<label class="block">
				<span class="text-sm font-medium text-fg-muted"
					>Preset name <span class="text-danger">*</span></span
				>
				<input
					type="text"
					bind:value={name}
					placeholder="e.g. galaxies_band6"
					class="mt-1 w-full rounded-md border border-line p-2 text-sm"
				/>
			</label>
			<label class="block">
				<span class="text-sm font-medium text-fg-muted">Description</span>
				<input
					type="text"
					bind:value={description}
					placeholder="Optional description"
					class="mt-1 w-full rounded-md border border-line p-2 text-sm"
				/>
			</label>
		</div>
		{#snippet footer()}
			<button type="button" class="btn-ghost" onclick={onCancel}>Cancel</button>
			<button type="button" class="btn-primary" onclick={onSave} disabled={saving || !name.trim()}
				>{saving ? 'Saving…' : 'Save'}</button
			>
		{/snippet}
	</Dialog>
{/if}
