<!--
	One FITS image or cube: a plane at a time (/api/v2/files/fits/plane), with
	a clip, stretch and colour map, a channel slider for cubes, and the value
	under the cursor. Drag or arrow keys pan, the wheel or +/- zoom, 0 resets,
	[ and ] step through channels. The view (zoom and pan) is bindable, so
	viewers side by side can share one.
-->
<script lang="ts">
	import { api } from '#lib/api/v2/client.js';
	import ErrorMessage from '#lib/components/shared/ErrorMessage.svelte';
	import Spinner from '#lib/components/shared/Spinner.svelte';
	import { fitScale, screenToImage, type CanvasView } from '#lib/utils/canvasGeometry.js';
	import {
		COLOR_MAPS,
		STRETCHES,
		colorAt,
		percentileRange,
		renderPlane,
		type ColorMap,
		type Plane,
		type Stretch
	} from '#lib/utils/imageRender.js';
	import { cubeInfo, type CubeInfo } from '#lib/utils/fitsHeader.js';

	interface View {
		scale: number;
		panX: number;
		panY: number;
	}

	interface Props {
		path: string;
		view?: View;
		height?: number;
	}

	let { path, view = $bindable({ scale: 1, panX: 0, panY: 0 }), height = 420 }: Props = $props();

	const CLIPS = [
		{ label: '99.5%', lo: 0.5, hi: 99.5 },
		{ label: '99%', lo: 1, hi: 99 },
		{ label: '95%', lo: 5, hi: 95 },
		{ label: 'min–max', lo: 0, hi: 100 }
	];

	let info = $state<CubeInfo | null>(null);
	let error = $state<unknown>(null);
	let channel = $state(0);
	let plane = $state.raw<Plane | null>(null);
	let step = $state(1);
	let loading = $state(false);
	let stretchKind = $state<Stretch>('asinh');
	let colorMap = $state<ColorMap>('inferno');
	let clip = $state(0);
	let width = $state(600);
	let canvas: HTMLCanvasElement | undefined = $state();
	let hover = $state<{ x: number; y: number; value: number | null } | null>(null);
	let dragging: { x: number; y: number; panX: number; panY: number } | null = null;

	const range = $derived(plane ? percentileRange(plane, CLIPS[clip].lo, CLIPS[clip].hi) : null);
	const imageSize = $derived(plane ? { w: plane[0]?.length ?? 0, h: plane.length } : null);
	const geometry = $derived<CanvasView | null>(
		imageSize
			? {
					displayWidth: width,
					displayHeight: height,
					imageWidth: imageSize.w,
					imageHeight: imageSize.h,
					...view
				}
			: null
	);

	// The header once per file.
	$effect(() => {
		const file = path;
		info = null;
		error = null;
		channel = 0;
		let stale = false;
		api
			.GET('/api/v2/files/fits', { params: { query: { path: file } } })
			.then(({ data }) => !stale && data && (info = cubeInfo(data.shape, data.cards)))
			.catch((err) => !stale && (error = err));
		return () => (stale = true);
	});

	// The plane, a moment after the channel settles.
	$effect(() => {
		const file = path;
		const wanted = channel;
		if (!info) return;
		const controller = new AbortController();
		const timer = setTimeout(async () => {
			loading = true;
			try {
				const { data } = await api.GET('/api/v2/files/fits/plane', {
					params: { query: { path: file, channel: wanted } },
					signal: controller.signal
				});
				if (data) {
					plane = data.data;
					step = data.step;
				}
				error = null;
			} catch (err) {
				if (!controller.signal.aborted) error = err;
			} finally {
				if (!controller.signal.aborted) loading = false;
			}
		}, 120);
		return () => {
			clearTimeout(timer);
			controller.abort();
		};
	});

	// Draw: the plane coloured off screen at its own size, then scaled into view.
	$effect(() => {
		if (!canvas || !plane || !range || !geometry || !imageSize) return;
		const ctx = canvas.getContext('2d');
		if (!ctx) return;
		const pixels = renderPlane(plane, {
			min: range[0],
			max: range[1],
			stretch: stretchKind,
			colorMap
		});
		const off = new OffscreenCanvas(imageSize.w, imageSize.h);
		off.getContext('2d')?.putImageData(new ImageData(pixels, imageSize.w, imageSize.h), 0, 0);
		const total = geometry.scale * fitScale(geometry);
		ctx.setTransform(1, 0, 0, 1, 0, 0);
		ctx.clearRect(0, 0, canvas.width, canvas.height);
		ctx.imageSmoothingEnabled = false;
		ctx.translate(width / 2 + geometry.panX, height / 2 + geometry.panY);
		ctx.scale(total, total);
		ctx.drawImage(off, -imageSize.w / 2, -imageSize.h / 2);
	});

	function zoom(factor: number) {
		view = { ...view, scale: Math.min(64, Math.max(0.25, view.scale * factor)) };
	}

	function pan(dx: number, dy: number) {
		view = { ...view, panX: view.panX + dx, panY: view.panY + dy };
	}

	function onKey(event: KeyboardEvent) {
		const keys: Record<string, () => void> = {
			ArrowLeft: () => pan(40, 0),
			ArrowRight: () => pan(-40, 0),
			ArrowUp: () => pan(0, 40),
			ArrowDown: () => pan(0, -40),
			'+': () => zoom(1.25),
			'=': () => zoom(1.25),
			'-': () => zoom(0.8),
			'0': () => (view = { scale: 1, panX: 0, panY: 0 }),
			'[': () => (channel = Math.max(0, channel - 1)),
			']': () => (channel = Math.min((info?.channels ?? 1) - 1, channel + 1))
		};
		const action = keys[event.key];
		if (action) {
			event.preventDefault();
			action();
		}
	}

	function onPointerMove(event: PointerEvent) {
		if (dragging) {
			view = {
				...view,
				panX: dragging.panX + event.clientX - dragging.x,
				panY: dragging.panY + event.clientY - dragging.y
			};
			return;
		}
		if (!geometry || !plane || !imageSize) return;
		const rect = (event.currentTarget as HTMLElement).getBoundingClientRect();
		const p = screenToImage(geometry, {
			x: event.clientX - rect.left,
			y: event.clientY - rect.top
		});
		const x = Math.floor(p.x);
		const yFromTop = Math.floor(p.y);
		if (x < 0 || yFromTop < 0 || x >= imageSize.w || yFromTop >= imageSize.h) {
			hover = null;
			return;
		}
		const row = imageSize.h - 1 - yFromTop; // FITS rows run bottom up
		hover = { x: x * step, y: row * step, value: plane[row][x] };
	}

	const colorbar = $derived(
		`linear-gradient(to right, ${[0, 0.25, 0.5, 0.75, 1]
			.map((t) => `rgb(${colorAt(colorMap, t).join(',')}) ${t * 100}%`)
			.join(', ')})`
	);
	const fmt = (v: number | null | undefined) =>
		v == null
			? '—'
			: Math.abs(v) >= 1e4 || (v !== 0 && Math.abs(v) < 1e-3)
				? v.toExponential(3)
				: v.toPrecision(4);
</script>

<div class="flex flex-col gap-2">
	{#if error}<ErrorMessage {error} title="Could not read it" />{/if}
	<div class="flex flex-wrap items-center gap-2 text-xs">
		<label class="flex items-center gap-1">
			<span class="text-fg-muted">Clip</span>
			<select class="rounded border px-1.5 py-1" bind:value={clip}>
				{#each CLIPS as c, i (c.label)}<option value={i}>{c.label}</option>{/each}
			</select>
		</label>
		<label class="flex items-center gap-1">
			<span class="text-fg-muted">Stretch</span>
			<select class="rounded border px-1.5 py-1" bind:value={stretchKind}>
				{#each STRETCHES as s (s)}<option value={s}>{s}</option>{/each}
			</select>
		</label>
		<label class="flex items-center gap-1">
			<span class="text-fg-muted">Colours</span>
			<select class="rounded border px-1.5 py-1" bind:value={colorMap}>
				{#each COLOR_MAPS as m (m)}<option value={m}>{m}</option>{/each}
			</select>
		</label>
		<button
			type="button"
			class="btn-ghost px-2 py-1 text-xs"
			onclick={() => (view = { scale: 1, panX: 0, panY: 0 })}>Reset view</button
		>
		{#if loading}<Spinner />{/if}
	</div>

	<div
		class="relative overflow-hidden rounded-md bg-fg"
		bind:clientWidth={width}
		style="height: {height}px"
	>
		<canvas
			bind:this={canvas}
			{width}
			{height}
			tabindex="0"
			aria-label="{path.split('/').pop()}, channel {channel +
				1}. Arrow keys pan, plus and minus zoom, 0 resets, brackets change channel."
			class="block cursor-grab touch-none active:cursor-grabbing"
			onkeydown={onKey}
			onwheel={(e) => {
				e.preventDefault();
				zoom(e.deltaY < 0 ? 1.15 : 1 / 1.15);
			}}
			onpointerdown={(e) => {
				dragging = { x: e.clientX, y: e.clientY, panX: view.panX, panY: view.panY };
				(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
			}}
			onpointerup={() => (dragging = null)}
			onpointermove={onPointerMove}
			onpointerleave={() => (hover = null)}
		></canvas>
		{#if !plane && !error}
			<p class="absolute inset-0 flex items-center justify-center gap-2 text-sm text-navy-soft">
				<Spinner /> Loading…
			</p>
		{/if}
	</div>

	<div class="flex flex-wrap items-center gap-3 text-xs text-fg-muted">
		<div class="flex items-center gap-1.5">
			<span class="font-mono">{fmt(range?.[0])}</span>
			<span class="inline-block h-2.5 w-28 rounded-sm" style="background: {colorbar}"></span>
			<span class="font-mono">{fmt(range?.[1])}</span>
			<span>{info?.unit ?? ''}</span>
		</div>
		<span class="ml-auto font-mono" aria-live="off">
			{#if hover}x {hover.x} y {hover.y}: {fmt(hover.value)}{:else}&nbsp;{/if}
		</span>
	</div>

	{#if info && info.channels > 1}
		<label class="flex items-center gap-3 text-xs">
			<span class="whitespace-nowrap text-fg-muted">Channel {channel + 1} / {info.channels}</span>
			<input
				type="range"
				class="flex-1"
				min="0"
				max={info.channels - 1}
				bind:value={channel}
				aria-label="Channel"
			/>
			<span class="w-24 text-right font-mono">{info.channelLabel(channel)}</span>
		</label>
	{/if}

	{#if info}
		<p class="m-0 text-xs text-fg-dim">
			{info.shape.join(' × ')}{step > 1 ? ` (shown 1:${step})` : ''}{info.beam
				? ` · beam ${info.beam}`
				: ''}
		</p>
	{/if}
</div>
