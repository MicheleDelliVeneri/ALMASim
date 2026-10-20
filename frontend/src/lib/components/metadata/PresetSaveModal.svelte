<script lang="ts">
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
	<div class="fixed inset-0 z-50 flex items-center justify-center bg-black/40">
		<div class="w-full max-w-sm rounded-lg bg-surface p-6 shadow-panel">
			<h3 class="mb-4 text-base font-semibold text-fg">Save Query as Preset</h3>
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
			<div class="mt-5 flex justify-end gap-2">
				<button
					onclick={onCancel}
					class="rounded-md px-3 py-2 text-sm font-medium text-fg-muted ring-1 ring-gray-300 hover:bg-hover"
					>Cancel</button
				>
				<button
					onclick={onSave}
					disabled={saving || !name.trim()}
					class="rounded-md bg-primary px-3 py-2 text-sm font-medium text-white hover:bg-primary-hover disabled:bg-gray-400"
					>{saving ? 'Saving…' : 'Save'}</button
				>
			</div>
		</div>
	</div>
{/if}
