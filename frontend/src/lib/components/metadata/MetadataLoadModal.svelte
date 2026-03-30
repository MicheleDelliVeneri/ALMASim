<script lang="ts">
	interface Props {
		open: boolean;
		loading: boolean;
		defaultPath: string;
		onClose: () => void;
		onSubmit: (event: Event) => void;
		formRef?: HTMLFormElement;
	}

	let { open, loading, defaultPath, onClose, onSubmit, formRef = $bindable() }: Props = $props();
</script>

{#if open}
	<div
		class="fixed inset-0 z-50 flex items-center justify-center bg-black/40 px-4"
		role="dialog"
		aria-modal="true"
	>
		<div class="w-full max-w-lg rounded-lg bg-white shadow-xl">
			<header class="flex items-center justify-between border-b px-6 py-4">
				<h2 class="text-lg font-semibold text-gray-900">Load Metadata File</h2>
				<button
					type="button"
					class="text-gray-500 hover:text-gray-800"
					aria-label="Close"
					onclick={onClose}
				>
					✕
				</button>
			</header>
			<form bind:this={formRef} onsubmit={onSubmit} class="space-y-4 px-6 py-6">
				<div class="space-y-1">
					<label class="block text-sm font-medium text-gray-700" for="file_upload">
						Select local file (JSON or CSV)
					</label>
					<input
						id="file_upload"
						name="file_upload"
						type="file"
						accept=".json,.csv,application/json,text/csv"
						class="w-full rounded-md border border-gray-300 p-2 text-sm text-gray-700 file:mr-4 file:rounded file:border-0 file:bg-gray-100 file:px-4 file:py-2 file:text-sm file:font-medium file:text-gray-700 hover:file:bg-gray-200"
					/>
					<p class="text-xs text-gray-500">
						The file contents stay on your machine; we parse supported formats directly in the
						browser.
					</p>
				</div>

				<div class="space-y-1">
					<label class="block text-sm font-medium text-gray-700" for="file_path">
						Or load from server path
					</label>
					<input
						type="text"
						id="file_path"
						name="file_path"
						placeholder="data/metadata/sample.json"
						value={defaultPath}
						class="w-full rounded-md border border-gray-300 p-2"
					/>
					<p class="text-xs text-gray-500">
						Use when the backend already has access to the metadata file.
					</p>
				</div>

				<div class="flex items-center justify-end gap-2">
					<button
						type="button"
						class="rounded-md border border-gray-300 px-4 py-2 text-sm font-medium text-gray-700 hover:bg-gray-50"
						onclick={onClose}
					>
						Cancel
					</button>
					<button
						type="submit"
						disabled={loading}
						class="rounded-md bg-gray-900 px-4 py-2 text-sm font-medium text-white hover:bg-gray-800 disabled:bg-gray-600"
					>
						{loading ? 'Loading...' : 'Load'}
					</button>
				</div>
			</form>
		</div>
	</div>
{/if}
