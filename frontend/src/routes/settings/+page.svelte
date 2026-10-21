<!--
	Settings: what your profile and the site's add up to; your own profile to
	edit (TOML, checked with the site's before it is saved, then used at
	once); the site profile to read.
-->
<script lang="ts">
	import { api } from '#lib/api/v2/client.js';
	import type { components } from '#lib/api/v2/schema.js';
	import ErrorMessage from '#lib/components/shared/ErrorMessage.svelte';
	import Spinner from '#lib/components/shared/Spinner.svelte';
	import { toasts } from '#lib/toasts.svelte.js';
	import { errorMessage } from '#lib/utils/errors.js';

	type Profiles = components['schemas']['Profiles'];

	let profiles = $state<Profiles | null>(null);
	let loadError = $state<unknown>(null);
	let text = $state('');
	let saving = $state(false);
	let saveError = $state<string | null>(null);

	const changed = $derived(profiles !== null && text !== profiles.user.text);

	async function load() {
		try {
			const { data } = await api.GET('/api/v2/profile');
			profiles = data ?? null;
			text = data?.user.text ?? '';
			loadError = null;
		} catch (err) {
			loadError = err;
		}
	}

	$effect(() => {
		void load();
	});

	async function save(event: SubmitEvent) {
		event.preventDefault();
		saving = true;
		saveError = null;
		try {
			const { data } = await api.PUT('/api/v2/profile/user', { body: { text } });
			if (data) {
				profiles = data;
				text = data.user.text;
			}
			toasts.show('Profile saved; new runs use it.', 'ok');
		} catch (err) {
			saveError = errorMessage(err);
		} finally {
			saving = false;
		}
	}

	const EXAMPLE = `# Your own settings, on top of the site's. For example:
[slurm]
partition = "zwickyB-CPU"
account = "my-project"

# Where the console may browse and write.
roots = ["/SKAO_data/cycle11", "~"]
`;
</script>

<svelte:head><title>Settings · ALMASim</title></svelte:head>

<div class="flex flex-col gap-5">
	<div class="flex flex-col gap-1">
		<p class="alma-kicker">Settings</p>
		<h1 class="m-0 text-3xl">Your profile</h1>
		<p class="m-0 text-fg-muted">
			Where runs go and what they ask Slurm for. The site profile sets the defaults; yours adds to
			them and wins where both say something.
		</p>
	</div>

	{#if loadError}
		<ErrorMessage error={loadError} title="Could not read the profiles" onRetry={load} />
	{:else if !profiles}
		<p class="flex items-center gap-2 text-sm text-fg-muted"><Spinner /> Loading…</p>
	{:else}
		<section class="alma-panel flex flex-col gap-3 rounded-lg p-5" aria-labelledby="eff-h">
			<h2 id="eff-h" class="m-0 text-xl font-semibold">In effect</h2>
			<dl class="m-0 grid grid-cols-[10rem_minmax(0,1fr)] gap-x-4 gap-y-2 text-sm">
				<dt class="text-fg-muted">Runs go to</dt>
				<dd class="m-0 font-mono text-xs break-all">{profiles.effective.runs_root}</dd>
				<dt class="text-fg-muted">Folders you can use</dt>
				<dd class="m-0 font-mono text-xs break-all">{profiles.effective.roots.join(' · ')}</dd>
				<dt class="text-fg-muted">Runs on</dt>
				<dd class="m-0">
					{profiles.effective.executor === 'server' ? "the console's own machine" : 'Slurm'}
				</dd>
				<dt class="text-fg-muted">Partition</dt>
				<dd class="m-0">{profiles.effective.partition ?? 'Slurm’s default'}</dd>
				<dt class="text-fg-muted">Account</dt>
				<dd class="m-0">{profiles.effective.account ?? 'none'}</dd>
				<dt class="text-fg-muted">Python for jobs</dt>
				<dd class="m-0 font-mono text-xs break-all">{profiles.effective.python}</dd>
				<dt class="text-fg-muted">WSClean</dt>
				<dd class="m-0 font-mono text-xs break-all">{profiles.effective.wsclean ?? 'on PATH'}</dd>
				<dt class="text-fg-muted">CASA data</dt>
				<dd class="m-0 font-mono text-xs break-all">
					{profiles.effective.casa_data ?? 'CASA’s own'}
				</dd>
				{#if profiles.effective.commands.length}
					<dt class="text-fg-muted">Commands set up</dt>
					<dd class="m-0 text-xs">{profiles.effective.commands.join(', ')}</dd>
				{/if}
			</dl>
		</section>

		<form
			class="alma-panel flex flex-col gap-3 rounded-lg p-5"
			onsubmit={save}
			aria-labelledby="user-h"
		>
			<div class="flex flex-wrap items-baseline gap-3">
				<h2 id="user-h" class="m-0 text-xl font-semibold">Yours</h2>
				<span class="font-mono text-xs text-fg-muted">{profiles.user.path}</span>
				{#if !profiles.user.exists}<span class="text-xs text-fg-dim">(not made yet)</span>{/if}
			</div>
			<label class="sr-only" for="user-profile">Your profile (TOML)</label>
			<textarea
				id="user-profile"
				class="min-h-64 rounded-md border px-3 py-2 font-mono text-sm"
				spellcheck="false"
				placeholder={EXAMPLE}
				bind:value={text}></textarea>
			{#if saveError}
				<p class="m-0 text-sm text-danger" role="alert">{saveError}</p>
			{/if}
			<div class="flex flex-wrap items-center gap-3">
				<button type="submit" class="btn-primary" disabled={!changed || saving}
					>{saving ? 'Saving…' : 'Save'}</button
				>
				{#if changed}
					<button type="button" class="btn-ghost" onclick={() => (text = profiles?.user.text ?? '')}
						>Undo changes</button
					>
				{/if}
				<p class="m-0 text-xs text-fg-muted">
					Checked together with the site profile first: a mistake is shown here and nothing is
					saved.
				</p>
			</div>
		</form>

		<section class="alma-panel flex flex-col gap-3 rounded-lg p-5" aria-labelledby="site-h">
			<div class="flex flex-wrap items-baseline gap-3">
				<h2 id="site-h" class="m-0 text-xl font-semibold">The site's</h2>
				<span class="font-mono text-xs text-fg-muted">{profiles.site.path ?? 'none'}</span>
				<span class="rounded-full bg-sunk px-2 py-0.5 text-xs text-fg-muted">read only</span>
			</div>
			{#if profiles.site.exists}
				<pre
					class="m-0 max-h-96 overflow-auto rounded-md bg-sunk p-3 font-mono text-xs leading-5">{profiles
						.site.text}</pre>
			{:else}
				<p class="m-0 text-sm text-fg-muted">
					No site profile (ALMASIM_SITE_PROFILE is not set): yours is the only one.
				</p>
			{/if}
		</section>
	{/if}
</div>
