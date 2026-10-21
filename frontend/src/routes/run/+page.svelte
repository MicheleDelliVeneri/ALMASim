<!--
	Run any command: the catalogue, then a form built from the command's
	parameters. The CLI's parser checks the form as you type (on the console),
	and the panel shows the exact command line and resources it would submit.
	`?command=<name>` opens one command; `?from=<run id>` edits a past run.
-->
<script lang="ts">
	import { goto } from '$app/navigation';
	import { resolve } from '$app/paths';
	import { page } from '$app/state';
	import { api } from '#lib/api/v2/client.js';
	import type { components } from '#lib/api/v2/schema.js';
	import CommandSections from '#lib/components/run/CommandSections.svelte';
	import ErrorMessage from '#lib/components/shared/ErrorMessage.svelte';
	import Spinner from '#lib/components/shared/Spinner.svelte';
	import { jobs } from '#lib/stores/jobs.svelte.js';
	import { toasts } from '#lib/toasts.svelte.js';
	import {
		buildArgv,
		initialValues,
		parseArgv,
		searchCommands,
		type Command,
		type Values
	} from '#lib/utils/commandForm.js';
	import { errorMessage } from '#lib/utils/errors.js';
	import { commandLine } from '#lib/utils/jobs.js';

	type Validation = components['schemas']['Validation'];
	type Cluster = components['schemas']['Cluster'];

	let commands = $state<Command[]>([]);
	let loadError = $state<unknown>(null);
	let cluster = $state<Cluster | null>(null);
	let query = $state('');
	let selected = $state<Command | null>(null);
	let values = $state<Values>({});
	let extraText = $state('');
	let title = $state('');
	let resources = $state({ partition: '', cpus: '', memory: '', time: '' });
	let validation = $state<Validation | null>(null);
	let validating = $state(false);
	let validateError = $state<unknown>(null);
	let submitting = $state(false);
	/** Bumped to rebuild the form's fields (another command, or a run loaded into it). */
	let formKey = $state(0);

	const shown = $derived(searchCommands(commands, query));
	const groups = $derived.by(() => {
		const byWord: Record<string, Command[]> = {};
		for (const c of shown) (byWord[c.name.split(' ')[0]] ??= []).push(c);
		return Object.entries(byWord);
	});
	const extra = $derived(extraText.split(/\s+/).filter(Boolean));
	const argv = $derived(selected ? buildArgv(selected, values, extra) : []);
	const overrides = $derived(
		Object.fromEntries(
			Object.entries(resources)
				.filter(([, v]) => v.trim() !== '')
				.map(([k, v]) => [k, k === 'cpus' ? Number(v) : v.trim()])
		)
	);
	const plan = $derived(validation?.plan ?? null);
	const partitionName = $derived(resources.partition || plan?.resources.partition || '');
	const partition = $derived(cluster?.partitions.find((p) => p.name === partitionName));
	const configured = $derived(partitionName ? cluster?.configured[partitionName] : undefined);

	async function loadCommands() {
		try {
			const { data } = await api.GET('/api/v2/commands');
			commands = data ?? [];
			loadError = null;
		} catch (err) {
			loadError = err;
		}
	}

	async function loadCluster() {
		try {
			const { data } = await api.GET('/api/v2/cluster');
			cluster = data ?? null;
		} catch {
			cluster = null; // the panel works without limits
		}
	}

	function open(
		command: Command,
		from?: { argv: string[]; title: string; resources?: Record<string, unknown> }
	) {
		selected = command;
		if (from) {
			const parsed = parseArgv(command, from.argv);
			values = parsed.values;
			extraText = parsed.extra.join(' ');
			title = from.title;
			const r = from.resources ?? {};
			resources = {
				partition: String(r.partition ?? ''),
				cpus: r.cpus == null ? '' : String(r.cpus),
				memory: String(r.memory ?? ''),
				time: String(r.time ?? '')
			};
		} else {
			values = initialValues(command);
			extraText = '';
			title = '';
			resources = { partition: '', cpus: '', memory: '', time: '' };
		}
		validation = null;
		formKey++;
	}

	function pick(command: Command) {
		open(command);
		// eslint-disable-next-line svelte/no-navigation-without-resolve -- the path is resolved; only the query string is appended
		void goto(`${resolve('/run')}?command=${encodeURIComponent(command.name)}`, {
			replaceState: true
		});
	}

	$effect(() => {
		void loadCommands();
		void loadCluster();
	});

	// Open what the address asks for, once the catalogue is in.
	let opened = '';
	$effect(() => {
		if (!commands.length) return;
		const name = page.url.searchParams.get('command');
		const from = page.url.searchParams.get('from');
		const key = `${name}|${from}`;
		if (key === opened) return;
		opened = key;
		if (from) void openRun(from);
		else if (name) {
			const command = commands.find((c) => c.name === name);
			if (command && command !== selected) open(command);
		}
	});

	async function openRun(runId: string) {
		try {
			const { data } = await api.GET('/api/v2/jobs/{run_id}', {
				params: { path: { run_id: runId } }
			});
			const command = data && commands.find((c) => c.name === data.command);
			if (!data || !command)
				throw new Error(`The console cannot run ${data?.command ?? 'that'} any more.`);
			open(command, { argv: data.argv, title: data.title, resources: data.resources });
			toasts.show(`Loaded ${data.short_id}: change what you need, then run it.`, 'info');
		} catch (err) {
			loadError = err;
		}
	}

	// The CLI's parser checks the form a moment after each change.
	$effect(() => {
		if (!selected) return;
		const body = { argv: [...argv], resources: { ...overrides } };
		const controller = new AbortController();
		const timer = setTimeout(async () => {
			validating = true;
			try {
				const { data } = await api.POST('/api/v2/commands/validate', {
					body,
					signal: controller.signal
				});
				validation = data ?? null;
				validateError = null;
			} catch (err) {
				if (!controller.signal.aborted) validateError = err;
			} finally {
				if (!controller.signal.aborted) validating = false;
			}
		}, 400);
		return () => {
			clearTimeout(timer);
			controller.abort();
		};
	});

	async function submit(event: SubmitEvent) {
		event.preventDefault();
		if (!selected || !validation?.ok) return;
		submitting = true;
		try {
			const { data } = await api.POST('/api/v2/jobs', {
				body: { argv, resources: overrides, params: { ...values }, title: title.trim() }
			});
			if (data) {
				jobs.put(data);
				toasts.show(`Submitted ${data.title || data.command} (${data.short_id})`, 'ok');
				await goto(resolve('/jobs/[id]', { id: data.id }));
			}
		} catch (err) {
			toasts.show(`Could not submit: ${errorMessage(err)}`, 'danger');
		} finally {
			submitting = false;
		}
	}

	const limitText = $derived.by(() => {
		const parts: string[] = [];
		const cpus = configured?.cpus ?? partition?.cpus_per_node;
		if (cpus) parts.push(`${cpus} CPUs per node`);
		const memory =
			configured?.memory ??
			(partition?.memory_mb_per_node
				? `${Math.round(partition.memory_mb_per_node / 1024)}G`
				: null);
		if (memory) parts.push(`${memory} per node`);
		const time = configured?.max_time ?? partition?.time_limit;
		parts.push(time ? `limit ${time}` : 'no time limit');
		if (partition?.load != null) parts.push(`${Math.round(partition.load * 100)}% busy`);
		return parts.join(' · ');
	});
</script>

<svelte:head
	><title>{selected ? `${selected.name} · Run` : 'Run a command'} · ALMASim</title></svelte:head
>

<div class="flex flex-col gap-5">
	<div class="flex flex-col gap-1">
		<p class="alma-kicker">Run a command</p>
		<h1 class="m-0 text-3xl">
			{selected ? `almasim ${selected.name}` : 'What do you want to run?'}
		</h1>
		{#if selected}<p class="m-0 text-fg-muted">{selected.summary}</p>{/if}
	</div>

	{#if loadError}
		<ErrorMessage error={loadError} title="Could not load the commands" onRetry={loadCommands} />
	{/if}

	<div class="flex flex-wrap items-start gap-6">
		{#if selected}
			<!-- On a narrow screen, once a command is open, the list folds into a select. -->
			<label class="flex w-full flex-col gap-1 text-sm lg:hidden">
				<span class="font-medium">Command</span>
				<select
					class="rounded-md border px-3 py-2"
					value={selected.name}
					onchange={(e) => {
						const command = commands.find((c) => c.name === e.currentTarget.value);
						if (command) pick(command);
					}}
				>
					{#each commands as command (command.name)}
						<option value={command.name}>{command.name}</option>
					{/each}
				</select>
			</label>
		{/if}
		<nav
			aria-label="Commands"
			class="alma-panel w-full flex-col gap-3 rounded-lg p-4 lg:sticky lg:top-20 lg:flex lg:w-64 lg:shrink-0 {selected
				? 'hidden'
				: 'flex'}"
		>
			<label>
				<span class="sr-only">Search commands</span>
				<input
					type="search"
					class="w-full rounded-md border px-3 py-2 text-sm"
					placeholder="Search commands…"
					bind:value={query}
				/>
			</label>
			{#if !commands.length && !loadError}
				<p class="flex items-center gap-2 text-sm text-fg-muted"><Spinner /> Loading…</p>
			{/if}
			<div class="flex max-h-[60vh] flex-col gap-3 overflow-y-auto">
				{#each groups as [word, list] (word)}
					<div class="flex flex-col gap-0.5">
						<p class="m-0 px-2 text-xs font-semibold tracking-wide text-fg-dim uppercase">{word}</p>
						{#each list as command (command.name)}
							<button
								type="button"
								class="rounded px-2 py-1.5 text-left text-sm text-fg-muted hover:bg-hover hover:text-navy aria-[current=true]:bg-sunk aria-[current=true]:font-bold aria-[current=true]:text-navy"
								aria-current={selected?.name === command.name}
								title={command.summary}
								onclick={() => pick(command)}>{command.name}</button
							>
						{/each}
					</div>
				{:else}
					{#if commands.length}<p class="m-0 text-sm text-fg-muted">No command matches.</p>{/if}
				{/each}
			</div>
		</nav>

		{#if selected}
			{#key formKey}
				<form
					class="flex min-w-0 flex-[999_1_36rem] flex-wrap items-start gap-6"
					onsubmit={submit}
					novalidate
				>
					<div class="flex min-w-0 flex-[999_1_28rem] flex-col gap-4">
						<CommandSections command={selected} bind:values />
						{#if selected.extra_args}
							<div class="alma-panel flex flex-col gap-1 rounded-lg p-5">
								<label for="extra" class="text-sm font-medium">More arguments</label>
								<input
									id="extra"
									class="rounded-md border px-3 py-2 font-mono text-sm"
									placeholder="Passed on as they are, e.g. -niter 1000"
									bind:value={extraText}
								/>
							</div>
						{/if}
					</div>

					<aside class="flex min-w-0 flex-[1_1_20rem] flex-col gap-4 lg:sticky lg:top-20">
						<section class="alma-panel flex flex-col gap-3 rounded-lg p-5" aria-labelledby="res-h">
							<h2 id="res-h" class="m-0 text-lg font-semibold">Resources</h2>
							{#if plan?.executor === 'server'}
								<p class="m-0 text-sm text-fg-muted">
									Runs on the console's own machine, under nice and ionice; it takes no Slurm
									resources.
								</p>
							{:else}
								<div class="grid grid-cols-2 gap-3">
									<label class="col-span-2 flex flex-col gap-1 text-sm">
										<span class="font-medium">Partition</span>
										<select class="rounded-md border px-3 py-2" bind:value={resources.partition}>
											<option value="">Default ({plan?.resources.partition ?? 'none'})</option>
											{#each cluster?.partitions ?? [] as p (p.name)}
												<option value={p.name} disabled={!p.up}
													>{p.name}{p.up ? '' : ' (down)'}</option
												>
											{/each}
										</select>
									</label>
									<label class="flex flex-col gap-1 text-sm">
										<span class="font-medium">CPUs</span>
										<input
											type="number"
											min="1"
											max={configured?.cpus ?? partition?.cpus_per_node ?? undefined}
											class="rounded-md border px-3 py-2"
											placeholder={String(plan?.resources.cpus ?? '')}
											bind:value={
												() => resources.cpus, (v) => (resources.cpus = v == null ? '' : String(v))
											}
										/>
									</label>
									<label class="flex flex-col gap-1 text-sm">
										<span class="font-medium">Memory</span>
										<input
											class="rounded-md border px-3 py-2"
											placeholder={plan?.resources.memory ?? '4G'}
											bind:value={resources.memory}
										/>
									</label>
									<label class="col-span-2 flex flex-col gap-1 text-sm">
										<span class="font-medium">Time limit</span>
										<input
											class="rounded-md border px-3 py-2"
											placeholder={plan?.resources.time ?? '02:00:00'}
											bind:value={resources.time}
										/>
									</label>
								</div>
								{#if partitionName}
									<p class="m-0 text-xs text-fg-muted">{partitionName}: {limitText}</p>
								{/if}
							{/if}
						</section>

						<section class="alma-panel flex flex-col gap-3 rounded-lg p-5" aria-labelledby="cmd-h">
							<div class="flex items-center gap-2">
								<h2 id="cmd-h" class="m-0 text-lg font-semibold">Command</h2>
								{#if validating}<Spinner />{/if}
							</div>
							<pre
								class="m-0 max-h-60 overflow-auto rounded-md bg-sunk p-3 font-mono text-xs leading-5 break-all whitespace-pre-wrap">{commandLine(
									plan?.argv ?? argv
								)}</pre>
							{#if validateError}
								<ErrorMessage error={validateError} title="Could not check the command" />
							{:else if validation && !validation.ok}
								<div role="alert">
									<ul class="m-0 flex list-none flex-col gap-1 p-0 text-sm text-danger">
										{#each validation.problems as problem (problem)}<li>{problem}</li>{/each}
									</ul>
								</div>
							{:else if plan}
								<p class="m-0 text-xs text-fg-muted">
									{plan.executor === 'slurm'
										? `A Slurm job: ${plan.resources.cpus} CPU, ${plan.resources.memory}, ${plan.resources.time} on ${plan.resources.partition ?? 'the default partition'}.`
										: 'Runs on the console server.'}
									{plan.argv.length > argv.length ? 'The console adds what the cluster needs.' : ''}
								</p>
							{/if}
							<label class="flex flex-col gap-1 text-sm">
								<span class="font-medium"
									>Title <span class="font-normal text-fg-dim">(optional)</span></span
								>
								<input
									class="rounded-md border px-3 py-2"
									placeholder="e.g. Band 3 galaxies, noise rerun"
									bind:value={title}
								/>
							</label>
							<button
								type="submit"
								class="btn-accent"
								disabled={!validation?.ok || submitting || validating}
							>
								{submitting ? 'Submitting…' : 'Run'}
							</button>
						</section>
					</aside>
				</form>
			{/key}
		{:else if commands.length}
			<p class="text-fg-muted">Pick a command on the left; its form appears here.</p>
		{/if}
	</div>
</div>
