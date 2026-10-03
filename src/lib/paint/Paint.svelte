<script module lang="ts">
	import type { BonsaiLLM } from '$lib/runtime/bonsai-llm';
	import type { Painter, Schedule } from '$lib/runtime/painter';

	// the model, loaded once and kept while the site is open (coming back to this page does not load it again)
	let model: Promise<{ llm: BonsaiLLM; painter: Painter; megabytes: number }> | undefined;
</script>

<script lang="ts">
	// Paint: type a prompt, get a picture from the one-file model (mindview-t2i-turbo), in one, two or four steps, on
	// this machine's GPU. Used by the site's /paint and by the Hugging Face Space (spaces/paint), which differ only in
	// their links.
	import { labDevice } from '$lib/lab/shared';
	import { packedUrl } from '$lib/models';
	import { BonsaiLLM as Reader } from '$lib/runtime/bonsai-llm';
	import { PackedModel } from '$lib/runtime/packed';
	import { Painter as ThePainter } from '$lib/runtime/painter';
	import SiteHeader, { type NavLink } from '$lib/ui/SiteHeader.svelte';
	import NeedsComputer from '$lib/ui/NeedsComputer.svelte';
	import { mayLoad } from '$lib/lab/device';
	import Paintbrush from '@lucide/svelte/icons/paintbrush';
	import Dices from '@lucide/svelte/icons/dices';
	import Download from '@lucide/svelte/icons/download';
	import ArrowUpRight from '@lucide/svelte/icons/arrow-up-right';

	interface Link {
		href: string;
		label: string;
	}
	let {
		header,
		compute,
		newTab = false
	}: {
		header?: { home: string; links: NavLink[] }; // a bar of its own (the Space; on the site the layout has one)
		compute: Link; // under the description: where to see it compute
		newTab?: boolean; // links open in a new tab (in the Space, which runs in a frame)
	} = $props();
	const target = $derived(newTab ? '_blank' : undefined);
	const rel = $derived(newTab ? 'noopener' : undefined);

	const HF = 'https://huggingface.co/mohsenvand/mindview-t2i-turbo';
	let prompt = $state('a lighthouse on a cliff during a thunderstorm, oil painting');
	let seed = $state(7);
	let status = $state('Starting the graphics card');
	let progress = $state<number | null>(0);
	let ready = $state(false);
	let painting = $state(false);
	let painted = $state(false);
	let result = $state('');
	let error = $state<string | null>(null);
	let blocked = $state<{ why: string; go: () => void } | null>(null); // the model may not fit here
	// instant: 1 step with the 1-step fine-tune (when the model has it); fast: 2 steps with the few-step LoRA, the first
	// at 256 x 256 (when the model has it); best: 4 steps
	// quick: Best's 4 steps with the first three at 256 x 256 (untrained: about half of Best's time, most prompts as good)
	let mode = $state<'instant' | 'fast' | 'best' | 'quick'>('fast');
	const MODES = [
		{
			id: 'instant',
			label: 'Instant',
			about:
				'One step, with the branch trained for it. About 4 s on an M4 MacBook Air; the roughest.'
		},
		{
			id: 'fast',
			label: 'Fast',
			about: 'Two steps: a sketch at a quarter of the size, then the picture. About 6 s.'
		},
		{
			id: 'quick',
			label: 'Quick Best',
			about:
				'Best with its first three steps at a quarter of the size: about half the time. Not trained for it yet, so some prompts lose a little.'
		},
		{ id: 'best', label: 'Best', about: 'Four full steps, the most faithful. About 15 s.' }
	] as const;
	const modes = $derived(MODES.filter((m) => m.id !== 'instant' || instantAvailable));
	const about = $derived(MODES.find((m) => m.id === mode)?.about ?? '');
	let frames = $state<string[]>([]); // what it had in mind after each step, small
	let fastAvailable = $state(false);
	let instantAvailable = $state(false);
	let canvas: HTMLCanvasElement | undefined;
	let figure: HTMLElement | undefined;
	let device: GPUDevice | undefined;

	function load() {
		model ??= (async () => {
			const dev = await labDevice();
			const packed = await PackedModel.open(packedUrl());
			const size = (part: 'reader' | 'painter') =>
				packed.tensors(part).reduce((a, t) => a + packed.bytes(t), 0);
			const r = size('reader'),
				p = size('painter'),
				total = r + p;
			const mb = (b: number) => Math.round(b / 1e6);
			const say = (got: number) => {
				progress = got / total;
				status = `Downloading the model: ${mb(got)} of ${mb(total)} MB. Your browser keeps it after the first time.`;
			};
			const llm = Reader.fromGGUF(dev, await packed.readerGGUF((f) => say(f * r)));
			const painter = await ThePainter.fromPacked(dev, packed, (e) => say(r + e.fraction * p));
			// the text stream cut to what the prompt needs (the rest is padding): the same pictures, a third less work
			painter.textLength = 'auto';
			return { llm, painter, megabytes: mb(total) };
		})().catch((e) => {
			model = undefined;
			throw e;
		});
		return model;
	}

	function mount(c: HTMLCanvasElement) {
		canvas = c;
		let alive = true;
		(async () => {
			// painting keeps about 2.8 GB on the GPU: on a phone, ask first
			await mayLoad(2.8, 890, (why, go) => (blocked = { why, go }));
			blocked = null;
			device = await labDevice();
			const { megabytes } = await load();
			if (!alive) return;
			ready = true;
			fastAvailable = (await load()).painter.hasLora;
			instantAvailable = (await load()).painter.hasOneStep;
			progress = null;
			status = `The model is ready (${megabytes} MB, one file). Type something and paint it.`;
		})().catch((e) => {
			error = e instanceof Error ? e.message : String(e);
		});
		return () => {
			alive = false;
		};
	}

	/** Copy the decoder's picture into the canvas. */
	async function show(painter: Painter) {
		if (!device || !canvas) return;
		const buf = device.createBuffer({
			size: 512 * 512 * 4,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
		});
		const enc = device.createCommandEncoder();
		enc.copyTextureToBuffer(
			{ texture: painter.taef2.texture },
			{ buffer: buf, bytesPerRow: 2048 },
			[512, 512]
		);
		device.queue.submit([enc.finish()]);
		await buf.mapAsync(GPUMapMode.READ);
		const px = new Uint8ClampedArray(buf.getMappedRange().slice(0));
		buf.destroy();
		canvas.getContext('2d')!.putImageData(new ImageData(px, 512, 512), 0, 0);
		painted = true;
		frames = [...frames, canvas.toDataURL('image/jpeg', 0.75)];
	}

	async function paint(e?: SubmitEvent) {
		e?.preventDefault();
		const text = prompt.trim();
		if (!ready || painting || !text) return;
		painting = true;
		result = '';
		frames = [];
		error = null;
		// on a phone the picture is below the controls: bring it into view
		const box = figure?.getBoundingClientRect();
		if (box && box.bottom > innerHeight)
			figure?.scrollIntoView({ behavior: 'smooth', block: 'end' });
		try {
			const { llm, painter } = await load();
			const t0 = performance.now();
			const schedule: Schedule =
				instantAvailable && mode === 'instant' ? '1r' : fastAvailable && mode === 'fast' ? 2 : 4;
			painter.setSteps(schedule);
			const steps = painter.steps;
			// at 2 steps the pads matter more: keep 256 text rows (1 and 4 steps need only the prompt and a few)
			painter.textLength = steps === 2 ? 256 : 'auto';
			status = 'Reading your words';
			progress = 0;
			await painter.encode(llm, text);
			// coarse first steps at 256 x 256 (a quarter of the work each): Fast's first, Quick's first three
			const low = steps === 2 ? 1 : mode === 'quick' ? 3 : 0;
			painter.setNoise(undefined, seed, low);
			for (let s = 0; s < steps; s++) {
				status = steps === 1 ? 'Painting, in one step' : `Painting: step ${s + 1} of ${steps}`;
				progress = s / steps;
				await painter.step(s);
				// what it has in mind after this step (after the last: the picture itself); the coarse steps before the
				// last one keep their latent, so nothing is shown for them
				if (s < low - 1) continue;
				if (low && s === low - 1) await painter.decode(0, await painter.upsampleLatent(seed + 1));
				else await painter.decode(painter.sigmas[s + 1]);
				await show(painter);
			}
			progress = null;
			const secs = ((performance.now() - t0) / 1000).toFixed(1);
			status = '';
			result = `Painted in ${secs} s on this computer, from seed ${seed}.`;
		} catch (err) {
			error = err instanceof Error ? err.message : String(err);
		} finally {
			painting = false;
		}
	}

	function promptKey(e: KeyboardEvent) {
		if (e.key === 'Enter' && !e.shiftKey) {
			e.preventDefault();
			paint();
		}
	}

	function newSeed() {
		seed = Math.floor(Math.random() * 1e6);
		paint();
	}

	function save() {
		canvas?.toBlob((blob) => {
			if (!blob) return;
			const a = document.createElement('a');
			a.href = URL.createObjectURL(blob);
			const name =
				prompt
					.trim()
					.toLowerCase()
					.replace(/[^a-z0-9]+/g, '-')
					.slice(0, 60) || 'picture';
			a.download = `${name}-${seed}.png`;
			a.click();
			URL.revokeObjectURL(a.href);
		}, 'image/png');
	}
</script>

{#if header}
	<SiteHeader home={header.home} links={header.links} {newTab} />
{/if}
<main>
	<section class="controls">
		<h1>Paint</h1>
		<p class="sub">
			Type something and this tab paints it, on your graphics card, from one file of about a
			gigabyte.
		</p>
		<form onsubmit={paint}>
			<textarea
				class="prompt"
				bind:value={prompt}
				onkeydown={promptKey}
				rows="2"
				placeholder="Describe a picture"
				aria-label="Describe a picture; Enter paints it"
				spellcheck="false"></textarea>
			{#if fastAvailable}
				<div class="seg modes" role="radiogroup" aria-label="Speed">
					{#each modes as m (m.id)}
						<label>
							<input type="radio" bind:group={mode} value={m.id} disabled={painting} />{m.label}
						</label>
					{/each}
				</div>
				<p class="hint mode-about">{about}</p>
			{/if}
			<div class="row">
				<button type="submit" class="btn primary lg" disabled={!ready || painting}>
					<Paintbrush />{painting ? 'Painting' : 'Paint'}
				</button>
				<label class="field">
					Seed
					<input type="number" min="0" step="1" bind:value={seed} disabled={painting} />
				</label>
				<button
					type="button"
					class="btn icon"
					onclick={newSeed}
					disabled={!ready || painting}
					aria-label="Paint again from a new seed"
					title="Paint again from a new seed"><Dices /></button
				>
			</div>
		</form>

		<div class="status" aria-live="polite">
			{#if error}
				<p role="alert" class="error">{error}</p>
			{:else if result}
				<p>{result}</p>
			{:else}
				<p>{status}</p>
				{#if progress !== null}
					<div class="progress"><span style:transform="scaleX({progress})"></span></div>
				{/if}
			{/if}
		</div>
	</section>

	{#if blocked}
		<NeedsComputer why={blocked.why} ontry={blocked.go} home={!header} />
	{/if}
	<section class="work">
		<figure bind:this={figure} class:painted class:busy={painting}>
			<canvas {@attach mount} width="512" height="512" aria-label="The painted picture"></canvas>
			{#if !painted}
				<figcaption class="empty">The picture appears here</figcaption>
			{/if}
		</figure>
		<div class="under">
			<ol class="frames" aria-label="What it had in mind after each step">
				{#each frames as f, i (i)}
					<li><img src={f} alt="After step {i + 1}" /></li>
				{/each}
			</ol>
			{#if painted && !painting}
				<button type="button" class="btn" onclick={save}><Download />Save</button>
			{/if}
		</div>
	</section>

	<details class="about">
		<summary>How it paints</summary>
		<p>
			Ternary Bonsai 1.7B reads your words with its first 9 layers; a linear map turns them into the
			painter's conditioning; the ternary diffusion transformer of Bonsai Image 4B paints in one,
			two or four steps, each with a small side branch trained for it; TAEF2 turns the result into
			pixels. Nearly every weight is −1, 0 or +1, and all of it is one file.
		</p>
		<p class="links">
			<a href={HF} {target} {rel}>The model on Hugging Face<ArrowUpRight /></a>
			<!-- eslint-disable-next-line svelte/no-navigation-without-resolve -- the caller resolves it (the site) or gives a full URL (the Space) -->
			<a href={compute.href} {target} {rel}>{compute.label}<ArrowUpRight /></a>
		</p>
	</details>
</main>

<style>
	main {
		/* the site locks the page itself; this page scrolls inside main, which phones need */
		position: fixed;
		inset: 0;
		overflow-y: auto;
		box-sizing: border-box;
		display: grid;
		grid-template-columns: minmax(20rem, 30rem) minmax(0, 1fr);
		grid-template-rows: 1fr auto 1fr;
		grid-template-areas: '. work' 'controls work' 'about work';
		column-gap: clamp(2rem, 5vw, 5rem);
		padding: calc(var(--header-h) + 1rem) var(--gutter) 2rem;
		background: var(--void);
		color: var(--bone);
	}
	h1 {
		margin: 0;
		font: 300 clamp(2rem, 3.4vw, 2.8rem) / 1.1 var(--serif);
		letter-spacing: -0.01em;
	}
	.sub {
		margin: 0.6rem 0 1.8rem;
		max-width: 26rem;
		color: var(--bone-2);
		font: 300 var(--t-md) / 1.5 var(--serif);
	}
	.prompt {
		--prompt-size: clamp(1.35rem, 2vw, 1.75rem);
		min-height: 4.2rem;
	}
	.modes {
		margin-top: 1.4rem;
	}
	.mode-about {
		min-height: 2.9em;
		margin: 0.55rem 0 0;
		max-width: 28rem;
	}
	.row {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 0.75rem;
		margin-top: 1.3rem;
	}
	.row .btn.primary {
		min-width: 8.5rem;
	}
	.status {
		min-height: 3rem;
		margin-top: 1.4rem;
		color: var(--bone-2);
		font: 400 var(--t-sm) / 1.5 var(--sans);
	}
	.status p {
		margin: 0;
	}
	.status .error {
		color: var(--ember);
	}
	.status .progress {
		margin-top: 0.6rem;
		max-width: 26rem;
	}
	.controls {
		grid-area: controls;
	}
	.about {
		grid-area: about;
		align-self: start;
		margin-top: 0.8rem;
		max-width: 30rem;
		border-top: 1px solid var(--hair);
		padding-top: 0.8rem;
	}
	.about summary {
		width: fit-content;
		color: var(--bone-2);
		font: 500 var(--t-sm) / 1.6 var(--sans);
		cursor: pointer;
	}
	.about summary:hover {
		color: var(--bone);
	}
	.about p {
		margin: 0.7rem 0 0;
		color: var(--bone-2);
		font: 300 var(--t-base) / 1.6 var(--serif);
	}
	.links {
		display: flex;
		flex-wrap: wrap;
		gap: 0.4rem 1.3rem;
	}
	.links a {
		display: inline-flex;
		align-items: center;
		gap: 0.2rem;
		color: var(--bone);
		font: 500 var(--t-sm) / 1.6 var(--sans);
		text-decoration: none;
		border-bottom: 1px solid var(--hair-2);
	}
	.links a:hover {
		border-bottom-color: var(--ember);
	}
	.links :global(svg) {
		width: 0.9rem;
		height: 0.9rem;
		color: var(--bone-3);
	}
	.work {
		grid-area: work;
		align-self: center;
		display: flex;
		flex-direction: column;
		align-items: center;
		min-width: 0;
	}
	figure {
		position: relative;
		margin: 0;
		width: min(calc(100vh - var(--header-h) - 9rem), 100%);
		max-width: 46rem;
		min-width: 16rem;
		aspect-ratio: 1;
		border: 1px solid var(--hair);
		border-radius: var(--r-sm);
		overflow: hidden;
		background: radial-gradient(circle at 50% 45%, rgb(236 229 216 / 0.035), transparent 70%);
	}
	figure.painted {
		border-color: transparent;
	}
	canvas {
		display: block;
		width: 100%;
		height: 100%;
		opacity: 0;
		transition: opacity 0.6s ease;
	}
	figure.painted canvas {
		opacity: 1;
	}
	figure.busy canvas {
		opacity: 0.8;
	}
	.empty {
		position: absolute;
		inset: 0;
		display: grid;
		place-items: center;
		color: var(--bone-3);
		font: italic 300 var(--t-md) / 1 var(--serif);
	}
	.under {
		display: flex;
		align-items: center;
		justify-content: space-between;
		gap: 1rem;
		width: min(calc(100vh - var(--header-h) - 9rem), 100%);
		max-width: 46rem;
		min-width: 16rem;
		min-height: 3rem;
		margin-top: 0.75rem;
	}
	.frames {
		display: flex;
		gap: 0.4rem;
		margin: 0;
		padding: 0;
		list-style: none;
	}
	.frames img {
		display: block;
		width: 2.75rem;
		height: 2.75rem;
		border-radius: 4px;
		opacity: 0.85;
	}
	@media (max-width: 860px) {
		main {
			display: flex;
			flex-direction: column;
			align-items: stretch;
			gap: 1.5rem;
			padding: calc(var(--header-h) + 0.5rem) var(--gutter) 3rem;
		}
		.sub {
			margin-bottom: 1.2rem;
		}
		figure,
		.under {
			width: 100%;
			min-width: 0;
		}
	}
	@media (prefers-reduced-motion: reduce) {
		canvas {
			transition: none;
		}
	}
</style>
