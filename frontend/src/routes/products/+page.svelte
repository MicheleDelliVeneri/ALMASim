<!--
	Look at results: browse the console's folders, open FITS images and cubes
	(up to three side by side, their views linked), and read a task's
	quality.json. ?dir=<folder> opens a folder (job pages, the imaging tiles
	and simulations link here); ?open=<file> (repeatable) opens images.
-->
<script lang="ts">
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { page } from '$app/state';
	import { api, type Listing, type Root } from '#lib/api/v2/client.js';
	import ImageViewer from '#lib/components/products/ImageViewer.svelte';
	import QualityView from '#lib/components/products/QualityView.svelte';
	import ErrorMessage from '#lib/components/shared/ErrorMessage.svelte';
	import Spinner from '#lib/components/shared/Spinner.svelte';
	import StateChip from '#lib/components/jobs/StateChip.svelte';
	import { jobs } from '#lib/stores/jobs.svelte.js';
	import { toasts } from '#lib/toasts.svelte.js';
	import { errorMessage } from '#lib/utils/errors.js';
	import { formatBytes } from '#lib/utils/format.js';
	import { RUN_DIR } from '#lib/utils/runDir.js';

	const MAX_OPEN = 3;

	let roots = $state<Root[]>([]);
	let listing = $state<Listing | null>(null);
	let listError = $state<unknown>(null);
	let quality = $state<Record<string, unknown> | null>(null);
	let linked = $state(true);
	let sharedView = $state({ scale: 1, panX: 0, panY: 0 });
	let ownViews = $state<Record<string, { scale: number; panX: number; panY: number }>>({});

	const dir = $derived(page.url.searchParams.get('dir'));
	const opened = $derived(page.url.searchParams.getAll('open').slice(0, MAX_OPEN));

	const isFits = (name: string) => /\.fits?(\.gz)?$/i.test(name);

	/** Previews computed as jobs, opened when they succeed. */
	interface Preview {
		id: string;
		label: string;
		/** What the job writes, in its run folder. */
		result: (runDir: string) => string;
	}
	let previews = $state<Preview[]>([]);
	const shown = new Set<string>();

	async function startPreview(argv: string[], label: string, result: Preview['result']) {
		try {
			const { data } = await api.POST('/api/v2/jobs', { body: { argv, title: label } });
			if (!data) return;
			jobs.put(data);
			previews = [...previews, { id: data.id, label, result }];
			toasts.show(`${label}: started; it opens here when done.`, 'info');
		} catch (err) {
			toasts.show(`Could not start it: ${errorMessage(err)}`, 'danger');
		}
	}

	function mapOf(path: string) {
		const name = (path.split('/').pop() ?? 'product').replace(/\.(fits?|ms)$/i, '');
		const file = `${name}-map.fits`;
		void startPreview(
			['preview', 'integrate', path, `${RUN_DIR}/${file}`],
			`Map · ${name}`,
			(dir) => join(dir, file)
		);
	}

	/** A simulation's dirty and beam cubes in this folder, by run index. */
	const wienerPairs = $derived.by(() => {
		if (!listing) return [];
		const names = new Set(listing.entries.map((e) => e.name));
		return listing.entries
			.map((e) => /^dirty-cube_(\d+)\.fits$/.exec(e.name)?.[1])
			.filter((n): n is string => n !== undefined && names.has(`beam-cube_${n}.fits`))
			.map((n) => ({
				label: `simulation ${n}`,
				dirty: join(listing!.path, `dirty-cube_${n}.fits`),
				beam: join(listing!.path, `beam-cube_${n}.fits`)
			}));
	});

	function wienerOf(pair: { label: string; dirty: string; beam: string }) {
		void startPreview(
			['preview', 'wiener', pair.dirty, pair.beam, RUN_DIR],
			`Wiener · ${pair.label}`,
			(dir) => join(dir, 'wiener-restored.fits')
		);
	}

	// Open what a preview job made, once, when it succeeds.
	$effect(() => {
		for (const preview of previews) {
			const run = jobs.get(preview.id);
			if (run?.state === 'succeeded' && !shown.has(preview.id)) {
				shown.add(preview.id);
				open(preview.result(run.run_dir), true);
			}
		}
	});
	const join = (folder: string, name: string) => `${folder.replace(/\/+$/, '')}/${name}`;

	function go(params: { dir?: string | null; open?: string[] }) {
		const next = new URLSearchParams();
		const folder = params.dir === undefined ? dir : params.dir;
		if (folder) next.set('dir', folder);
		for (const file of params.open ?? opened) next.append('open', file);
		const query = next.toString();
		// eslint-disable-next-line svelte/no-navigation-without-resolve -- the path is resolved; only the query string is appended
		void goto(`${resolve('/products')}${query ? `?${query}` : ''}`, { replaceState: true });
	}

	function open(path: string, compare: boolean) {
		const files = compare ? [...opened.filter((f) => f !== path), path] : [path];
		go({ open: files.slice(-MAX_OPEN) });
	}

	$effect(() => {
		void api
			.GET('/api/v2/files/roots')
			.then(({ data }) => {
				roots = data?.roots ?? [];
				const first = roots.find((r) => r.exists);
				if (!dir && first) go({ dir: first.path });
			})
			.catch((err) => (listError = err));
	});

	$effect(() => {
		const folder = dir;
		listing = null;
		quality = null;
		listError = null;
		if (!folder) return;
		let stale = false;
		api
			.GET('/api/v2/files/list', { params: { query: { path: folder } } })
			.then(async ({ data }) => {
				if (stale || !data) return;
				listing = data;
				if (data.entries.some((e) => e.name === 'quality.json')) {
					const q = await api.GET('/api/v2/files/json', {
						params: { query: { path: join(data.path, 'quality.json') } }
					});
					if (!stale) quality = (q.data as Record<string, unknown>) ?? null;
				}
			})
			.catch((err) => !stale && (listError = err));
		return () => (stale = true);
	});
</script>

<svelte:head><title>Products · ALMASim</title></svelte:head>

<div class="flex flex-col gap-5">
	<div class="flex flex-col gap-1">
		<p class="alma-kicker">Results</p>
		<h1 class="m-0 text-3xl">Products</h1>
		<p class="m-0 text-fg-muted">
			Images and cubes from imaging and simulations. Open up to three side by side; their views move
			together.
		</p>
	</div>

	<div class="flex flex-wrap items-start gap-6">
		<nav
			aria-label="Folders"
			class="alma-panel flex w-full flex-col gap-3 rounded-lg p-4 lg:sticky lg:top-20 lg:w-80 lg:shrink-0"
		>
			<div class="flex flex-wrap gap-1.5" role="group" aria-label="Places">
				{#each roots as root (root.path)}
					<button
						type="button"
						class="alma-chip rounded-full px-3 py-1 text-xs disabled:opacity-50"
						disabled={!root.exists}
						onclick={() => go({ dir: root.path })}>{root.runs ? 'Runs' : root.path}</button
					>
				{/each}
			</div>
			{#if listing}
				<div class="flex items-center gap-2">
					<button
						type="button"
						class="btn-ghost px-2 py-1"
						aria-label="Up one folder"
						disabled={!listing.parent}
						onclick={() => listing?.parent && go({ dir: listing.parent })}>↑</button
					>
					<p
						class="m-0 min-w-0 flex-1 truncate font-mono text-xs text-fg-muted"
						title={listing.path}
					>
						{listing.path}
					</p>
				</div>
				<ul
					class="m-0 flex max-h-[60vh] list-none flex-col overflow-y-auto p-0 text-sm"
					aria-label="Contents"
				>
					{#each listing.entries as entry (entry.name)}
						<li class="flex items-center gap-2 rounded px-2 py-1 hover:bg-hover">
							{#if entry.kind === 'folder'}
								<button
									type="button"
									class="min-w-0 flex-1 truncate text-left"
									onclick={() => listing && go({ dir: join(listing.path, entry.name) })}
									><span aria-hidden="true">📁</span> {entry.name}</button
								>
								{#if /\.ms$/i.test(entry.name)}
									<button
										type="button"
										class="shrink-0 text-xs text-accent hover:underline"
										title="Grid the visibilities into a dirty image (a Slurm job)"
										onclick={() => listing && mapOf(join(listing.path, entry.name))}>map</button
									>
								{/if}
							{:else if isFits(entry.name)}
								{@const path = join(listing.path, entry.name)}
								<button
									type="button"
									class="min-w-0 flex-1 truncate text-left font-medium text-navy"
									aria-pressed={opened.includes(path)}
									onclick={() => open(path, false)}>{entry.name}</button
								>
								<button
									type="button"
									class="shrink-0 text-xs text-accent hover:underline"
									title="Open beside the others"
									onclick={() => open(path, true)}>+ compare</button
								>
								<button
									type="button"
									class="shrink-0 text-xs text-accent hover:underline"
									title="Sum the channels into one image (a Slurm job)"
									onclick={() => mapOf(path)}>map</button
								>
							{:else}
								<span class="min-w-0 flex-1 truncate text-fg-muted">{entry.name}</span>
								{#if entry.size != null}<span class="shrink-0 text-xs text-fg-dim"
										>{formatBytes(entry.size)}</span
									>{/if}
							{/if}
						</li>
					{:else}
						<li class="px-2 py-3 text-fg-muted">Empty folder</li>
					{/each}
				</ul>
			{:else if listError}
				<ErrorMessage error={listError} title="Could not open the folder" />
			{:else}
				<p class="flex items-center gap-2 text-sm text-fg-muted"><Spinner /> Loading…</p>
			{/if}
		</nav>

		<div class="flex min-w-0 flex-[999_1_36rem] flex-col gap-4">
			{#if wienerPairs.length || previews.length}
				<section class="alma-panel flex flex-col gap-3 rounded-lg p-5" aria-labelledby="pv-h">
					<h2 id="pv-h" class="m-0 text-lg font-semibold">Previews</h2>
					{#each wienerPairs as pair (pair.dirty)}
						<div class="flex flex-wrap items-center gap-3 text-sm">
							<button type="button" class="btn-secondary" onclick={() => wienerOf(pair)}
								>Wiener reconstruction of {pair.label}</button
							>
							<span class="text-xs text-fg-muted"
								>from its dirty and beam cubes: model, restored and residual, as a Slurm job</span
							>
						</div>
					{/each}
					{#each previews as preview (preview.id)}
						{@const run = jobs.get(preview.id)}
						<div class="flex flex-wrap items-center gap-3 text-sm">
							{#if run}<StateChip state={run.state} />{/if}
							<a href={resolve('/jobs/[id]', { id: preview.id })}>{preview.label}</a>
							{#if run?.state === 'succeeded'}<span class="text-xs text-fg-muted">opened below</span
								>{/if}
						</div>
					{/each}
				</section>
			{/if}

			{#if quality}
				<section class="alma-panel rounded-lg p-5" aria-labelledby="q-h">
					<h2 id="q-h" class="mt-0 mb-3 text-lg font-semibold">How well it was cleaned</h2>
					<QualityView {quality} />
				</section>
			{/if}

			{#if opened.length}
				<div class="flex items-center gap-3">
					{#if opened.length > 1}
						<label class="flex items-center gap-2 text-sm">
							<input type="checkbox" bind:checked={linked} /> Move the views together
						</label>
					{/if}
					<button
						type="button"
						class="btn-ghost ml-auto px-3 py-1.5"
						onclick={() => go({ open: [] })}>Close all</button
					>
				</div>
				<div
					class="grid gap-4"
					style="grid-template-columns: repeat({opened.length}, minmax(0, 1fr))"
				>
					{#each opened as file (file)}
						<section class="alma-panel flex min-w-0 flex-col gap-2 rounded-lg p-3">
							<div class="flex items-center gap-2">
								<h2 class="m-0 min-w-0 flex-1 truncate font-mono text-sm" title={file}>
									{file.split('/').pop()}
								</h2>
								<button
									type="button"
									class="rounded p-1 text-fg-dim hover:bg-hover hover:text-fg"
									aria-label="Close {file.split('/').pop()}"
									onclick={() => go({ open: opened.filter((f) => f !== file) })}>✕</button
								>
							</div>
							{#if linked}
								<ImageViewer
									path={file}
									bind:view={sharedView}
									height={opened.length > 2 ? 320 : 420}
								/>
							{:else}
								<ImageViewer
									path={file}
									bind:view={
										() => ownViews[file] ?? { scale: 1, panX: 0, panY: 0 },
										(v) => (ownViews = { ...ownViews, [file]: v })
									}
									height={opened.length > 2 ? 320 : 420}
								/>
							{/if}
						</section>
					{/each}
				</div>
			{:else}
				<p class="text-fg-muted">Open a FITS file from the folder list.</p>
			{/if}
		</div>
	</div>
</div>
