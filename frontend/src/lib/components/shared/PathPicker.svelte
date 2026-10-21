<!--
	A path field with a browser over the console's folders (`/api/v2/files`):
	the profile's roots and the runs root, nothing outside them. Type a path,
	or browse to one; `kind` says whether a folder, a file, or either is wanted.
-->
<script lang="ts">
	import { api, type Listing, type Root } from '#lib/api/v2/client.js';
	import { childPath } from '#lib/utils/paths.js';
	import Dialog from './Dialog.svelte';
	import ErrorMessage from './ErrorMessage.svelte';
	import Spinner from './Spinner.svelte';

	interface Props {
		value: string;
		label: string;
		/** The command-line flag the field sets, shown beside the label. */
		flag?: string;
		kind?: 'folder' | 'file' | 'path';
		/** Offer "New folder" in the browser (folders only). */
		allowCreate?: boolean;
		required?: boolean;
		placeholder?: string;
		/** Shown under the field. */
		hint?: string;
	}

	let {
		value = $bindable(''),
		label,
		flag,
		kind = 'folder',
		allowCreate = false,
		required = false,
		placeholder = '',
		hint
	}: Props = $props();

	const fieldId = $props.id();

	let open = $state(false);
	let roots = $state<Root[]>([]);
	let listing = $state<Listing | null>(null);
	let chosenFile = $state<string | null>(null);
	let loading = $state(false);
	let error = $state<unknown>(null);
	let newFolder = $state('');

	/** A listed entry's path (the server's names are single path parts). */
	const join = (folder: string, name: string) => `${folder.replace(/\/+$/, '')}/${name}`;

	async function browse(path: string) {
		loading = true;
		error = null;
		chosenFile = null;
		try {
			const { data } = await api.GET('/api/v2/files/list', { params: { query: { path } } });
			listing = data ?? null;
		} catch (err) {
			error = err;
		} finally {
			loading = false;
		}
	}

	async function openBrowser() {
		open = true;
		error = null;
		try {
			const { data } = await api.GET('/api/v2/files/roots');
			roots = data?.roots ?? [];
		} catch (err) {
			error = err;
			return;
		}
		const start =
			kind !== 'folder' && value && !value.endsWith('/')
				? value.replace(/\/[^/]*$/, '') || '/'
				: value;
		const first = roots.find((root) => root.exists)?.path;
		const target = start || first;
		if (target) await browse(target);
		// A typed path outside the roots: show the first root instead.
		if (error && first && target !== first) await browse(first);
	}

	async function makeFolder() {
		if (!listing || !newFolder.trim()) return;
		try {
			const path = childPath(listing.path, newFolder);
			if (path === null) {
				error = new Error('A folder name cannot be empty, "." or "..", or contain a slash.');
				return;
			}
			await api.POST('/api/v2/files/mkdir', { body: { path } });
			newFolder = '';
			await browse(path);
		} catch (err) {
			error = err;
		}
	}

	function choose() {
		if (kind === 'file' || (kind === 'path' && chosenFile)) {
			if (!chosenFile || !listing) return;
			value = join(listing.path, chosenFile);
		} else if (listing) {
			value = listing.path;
		}
		open = false;
	}
</script>

<div>
	<label for={fieldId} class="mb-1 block text-sm font-medium text-fg"
		>{label}{#if flag}
			<code class="ml-1 inline-block text-xs font-normal whitespace-nowrap text-fg-dim">{flag}</code
			>{/if}</label
	>
	<div class="flex gap-2">
		<input
			id={fieldId}
			type="text"
			class="min-w-0 flex-1 rounded-md border px-3 py-2 font-mono text-sm"
			bind:value
			{required}
			{placeholder}
			spellcheck="false"
			autocomplete="off"
		/>
		<button type="button" class="btn-secondary" onclick={openBrowser}>Browse…</button>
	</div>
	{#if hint}<p class="mt-1 text-xs text-fg-dim">{hint}</p>{/if}
</div>

{#if open}
	<Dialog
		title={kind === 'file'
			? `Choose a file: ${label}`
			: kind === 'folder'
				? `Choose a folder: ${label}`
				: `Choose: ${label}`}
		width="max-w-2xl"
		onClose={() => (open = false)}
	>
		{#if roots.length > 0}
			<div class="mb-3 flex flex-wrap gap-2" role="group" aria-label="Places">
				{#each roots as root (root.path)}
					<button
						type="button"
						class="alma-chip rounded-full px-3 py-1 text-xs disabled:opacity-50"
						disabled={!root.exists}
						onclick={() => browse(root.path)}>{root.runs ? 'Runs' : root.path}</button
					>
				{/each}
			</div>
		{/if}

		{#if error}
			<div class="mb-3"><ErrorMessage {error} /></div>
		{/if}

		{#if listing}
			<div class="mb-2 flex items-center gap-2">
				<button
					type="button"
					class="btn-ghost px-2 py-1"
					disabled={!listing.parent || loading}
					aria-label="Up one folder"
					onclick={() => listing?.parent && browse(listing.parent)}>↑</button
				>
				<p class="min-w-0 flex-1 truncate font-mono text-sm text-fg-muted" title={listing.path}>
					{listing.path}
				</p>
				{#if loading}<Spinner />{/if}
			</div>
			<ul class="max-h-80 overflow-y-auto rounded-md border border-line" aria-label="Contents">
				{#each listing.entries as entry (entry.name)}
					{#if entry.kind === 'folder'}
						<li>
							<button
								type="button"
								class="flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm hover:bg-hover"
								onclick={() => listing && browse(join(listing.path, entry.name))}
							>
								<span aria-hidden="true">📁</span>{entry.name}
							</button>
						</li>
					{:else if kind !== 'folder'}
						<li>
							<button
								type="button"
								class="flex w-full items-center gap-2 px-3 py-1.5 text-left text-sm hover:bg-hover aria-pressed:bg-accent-soft aria-pressed:font-semibold"
								aria-pressed={chosenFile === entry.name}
								onclick={() => (chosenFile = entry.name)}
								ondblclick={() => {
									chosenFile = entry.name;
									choose();
								}}
							>
								<span aria-hidden="true">📄</span>{entry.name}
							</button>
						</li>
					{:else}
						<li class="flex items-center gap-2 px-3 py-1.5 text-sm text-fg-dim">
							<span aria-hidden="true">📄</span>{entry.name}
						</li>
					{/if}
				{:else}
					<li class="px-3 py-4 text-center text-sm text-fg-dim">Empty folder</li>
				{/each}
			</ul>
			{#if listing.truncated}
				<p class="mt-1 text-xs text-fg-dim">Only the first entries are shown.</p>
			{/if}
			{#if allowCreate && kind === 'folder'}
				<form
					class="mt-3 flex gap-2"
					onsubmit={(event) => {
						event.preventDefault();
						makeFolder();
					}}
				>
					<label class="sr-only" for="{fieldId}-new">New folder name</label>
					<input
						id="{fieldId}-new"
						class="min-w-0 flex-1 rounded-md border px-3 py-1.5 text-sm"
						placeholder="New folder name"
						bind:value={newFolder}
					/>
					<button type="submit" class="btn-secondary" disabled={!newFolder.trim()}>Create</button>
				</form>
			{/if}
		{/if}

		{#snippet footer()}
			<button type="button" class="btn-ghost" onclick={() => (open = false)}>Cancel</button>
			<button
				type="button"
				class="btn-primary"
				disabled={!listing || (kind === 'file' && !chosenFile)}
				onclick={choose}
				>{kind === 'file'
					? 'Choose file'
					: kind === 'path' && chosenFile
						? 'Choose file'
						: 'Choose this folder'}</button
			>
		{/snippet}
	</Dialog>
{/if}
