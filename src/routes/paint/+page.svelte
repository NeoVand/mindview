<script module lang="ts">
	import type { BonsaiLLM } from '$lib/runtime/bonsai-llm';
	import type { Painter } from '$lib/runtime/painter';

	// the model, loaded once and kept while the site is open (coming back to this page does not load it again)
	let model: Promise<{ llm: BonsaiLLM; painter: Painter; megabytes: number }> | undefined;
</script>

<script lang="ts">
	import { resolve } from '$app/paths';
	import { labDevice } from '$lib/lab/shared';
	import { packedUrl } from '$lib/models';
	import { BonsaiLLM as Reader } from '$lib/runtime/bonsai-llm';
	import { PackedModel } from '$lib/runtime/packed';
	import { Painter as ThePainter } from '$lib/runtime/painter';

	const HF = 'https://huggingface.co/mohsenvand/mindview-t2i';
	let prompt = $state('a lighthouse on a cliff during a thunderstorm, oil painting');
	let seed = $state(7);
	let status = $state('Starting the graphics card');
	let progress = $state<number | null>(0);
	let ready = $state(false);
	let painting = $state(false);
	let painted = $state(false);
	let result = $state('');
	let error = $state<string | null>(null);
	// fast: 2 steps with the few-step LoRA (when the model has it); best: 4 steps
	let mode = $state<'fast' | 'best'>('fast');
	let fastAvailable = $state(false);
	let canvas: HTMLCanvasElement | undefined;
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
			device = await labDevice();
			const { megabytes } = await load();
			if (!alive) return;
			ready = true;
			fastAvailable = (await load()).painter.hasLora;
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
	}

	async function paint(e?: SubmitEvent) {
		e?.preventDefault();
		const text = prompt.trim();
		if (!ready || painting || !text) return;
		painting = true;
		result = '';
		error = null;
		try {
			const { llm, painter } = await load();
			const t0 = performance.now();
			const steps = fastAvailable && mode === 'fast' ? 2 : 4;
			painter.setSteps(steps);
			// at 2 steps the pads matter more: keep 256 text rows (4 steps need only the prompt and a few)
			painter.textLength = steps === 2 ? 256 : 'auto';
			status = 'Reading your words';
			await painter.encode(llm, text);
			painter.setNoise(undefined, seed);
			for (let s = 0; s < steps; s++) {
				status = `Painting: step ${s + 1} of ${steps}`;
				await painter.step(s);
				// what it has in mind after this step (after the last: the picture itself)
				await painter.decode(painter.sigmas[s + 1]);
				await show(painter);
			}
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

<svelte:head>
	<title>Paint</title>
	<meta
		name="description"
		content="A text-to-image model in one file under 1 GB, running in your browser: type something and it paints it."
	/>
</svelte:head>

<main>
	<nav aria-label="Elsewhere">
		<a href={resolve('/')}>The first piece</a>
		<a href={resolve('/lab')}>The labs</a>
	</nav>

	<section class="controls">
		<form onsubmit={paint}>
			<textarea
				bind:value={prompt}
				onkeydown={promptKey}
				rows="2"
				placeholder="Describe a picture"
				aria-label="Describe a picture; Enter paints it"
				spellcheck="false"></textarea>
			<div class="row">
				<button type="submit" class="primary" disabled={!ready || painting}
					>{painting ? 'Painting…' : 'Paint'}</button
				>
				<label>
					Seed
					<input type="number" min="0" step="1" bind:value={seed} disabled={painting} />
				</label>
				<button type="button" onclick={newSeed} disabled={!ready || painting}>New seed</button>
				{#if fastAvailable}
					<div class="mode" role="radiogroup" aria-label="Speed">
						<label class:on={mode === 'fast'}>
							<input type="radio" bind:group={mode} value="fast" disabled={painting} />Fast, 2 steps
						</label>
						<label class:on={mode === 'best'}>
							<input type="radio" bind:group={mode} value="best" disabled={painting} />Best, 4 steps
						</label>
					</div>
				{/if}
			</div>
		</form>

		<div class="status" aria-live="polite">
			{#if error}
				<p role="alert">{error}</p>
			{:else if result}
				<p>{result}</p>
			{:else}
				<p>{status}</p>
				{#if progress !== null}
					<div class="bar"><span style:transform="scaleX({progress})"></span></div>
				{/if}
			{/if}
		</div>
	</section>

	<figure class:painted class:busy={painting}>
		<canvas {@attach mount} width="512" height="512" aria-label="The painted picture"></canvas>
		{#if !painted}
			<figcaption class="empty">The picture appears here</figcaption>
		{/if}
		{#if painted && !painting}
			<button type="button" class="save" onclick={save}>Save the picture</button>
		{/if}
	</figure>

	<section class="about">
		<p>
			One file of under a gigabyte paints this, on your GPU, in this tab. Ternary Bonsai 1.7B reads
			your words with its first 9 layers; a linear map turns them into the painter's conditioning;
			the ternary diffusion transformer of Bonsai Image 4B paints in four steps; TAEF2 turns the
			result into pixels. Nearly every weight is −1, 0 or +1.
		</p>
		<p class="links">
			<a href={HF}>The model on Hugging Face</a>
			<a href={resolve('/lab')}>See it compute, in the labs</a>
		</p>
	</section>
</main>

<style>
	main {
		/* the site locks the page itself (app.css); this page scrolls inside main, which phones need */
		position: fixed;
		inset: 0;
		overflow-y: auto;
		box-sizing: border-box;
		display: grid;
		grid-template-columns: minmax(18rem, 30rem) 1fr;
		grid-template-rows: 1fr 1fr;
		column-gap: 3rem;
		padding: 4.5rem 3rem 3rem;
		background: var(--void);
		color: var(--bone);
	}
	nav {
		position: absolute;
		top: 1.1rem;
		right: 1.4rem;
		display: flex;
		gap: 1.1rem;
		font-size: 0.82rem;
	}
	nav a,
	.links a {
		color: var(--bone);
		opacity: 0.6;
		text-decoration: none;
		border-bottom: 1px solid transparent;
	}
	nav a:hover,
	nav a:focus-visible,
	.links a:hover,
	.links a:focus-visible {
		opacity: 1;
		border-bottom-color: var(--ember);
		outline: none;
	}
	textarea {
		display: block;
		width: 100%;
		box-sizing: border-box;
		field-sizing: content;
		min-height: 4.5rem;
		max-height: 40vh;
		resize: none;
		background: none;
		border: none;
		border-bottom: 1px solid rgb(232 226 214 / 0.2);
		color: var(--bone);
		font: italic 300 clamp(1.4rem, 2.2vw, 2.1rem) / 1.3 var(--serif);
		padding: 0.3rem 0 0.6rem;
		outline: none;
	}
	textarea::placeholder {
		color: var(--bone);
		opacity: 0.35;
	}
	textarea:focus-visible {
		border-bottom-color: var(--ember);
	}
	.row {
		display: flex;
		flex-wrap: wrap;
		align-items: center;
		gap: 0.8rem 1rem;
		margin-top: 1.1rem;
		font-size: 0.9rem;
	}
	label {
		display: flex;
		align-items: center;
		gap: 0.5rem;
		opacity: 0.8;
	}
	input[type='number'] {
		width: 6.5rem;
		background: none;
		border: 1px solid rgb(232 226 214 / 0.25);
		color: var(--bone);
		font: inherit;
		padding: 0.3rem 0.5rem;
		font-variant-numeric: tabular-nums;
	}
	button {
		background: none;
		border: 1px solid rgb(232 226 214 / 0.3);
		color: var(--bone);
		font: inherit;
		font-size: 0.9rem;
		padding: 0.4rem 0.9rem;
		cursor: pointer;
	}
	button.primary {
		border-color: var(--ember);
		color: var(--ember);
		padding: 0.45rem 1.4rem;
		min-width: 7.5rem;
	}
	button:disabled {
		opacity: 0.4;
		cursor: default;
	}
	button:focus-visible,
	input:focus-visible {
		outline: 1px solid var(--ember);
		outline-offset: 2px;
	}
	.mode {
		display: flex;
		border: 1px solid rgb(232 226 214 / 0.25);
	}
	.mode label {
		padding: 0.35rem 0.7rem;
		cursor: pointer;
		opacity: 0.6;
	}
	.mode label.on {
		opacity: 1;
		background: rgb(232 226 214 / 0.1);
	}
	.mode label:has(input:focus-visible) {
		outline: 1px solid var(--ember);
	}
	.mode input {
		position: absolute;
		opacity: 0;
		pointer-events: none;
	}
	.status {
		min-height: 3.2rem;
		margin-top: 1.4rem;
		font-weight: 300;
		font-size: 0.95rem;
		line-height: 1.5;
	}
	.status p {
		margin: 0;
		opacity: 0.8;
	}
	.bar {
		height: 1px;
		margin-top: 0.7rem;
		background: rgb(232 226 214 / 0.15);
	}
	.bar span {
		display: block;
		height: 100%;
		background: var(--ember);
		transform-origin: left;
		transition: transform 0.2s linear;
	}
	.controls {
		align-self: end;
	}
	.about {
		align-self: start;
	}
	.about p:first-child {
		max-width: 60ch;
		margin: 0.6rem 0 0;
		font-weight: 300;
		font-size: 0.92rem;
		line-height: 1.6;
		opacity: 0.62;
	}
	.links {
		display: flex;
		flex-wrap: wrap;
		gap: 0.6rem 1.4rem;
		margin: 1rem 0 0;
		font-size: 0.88rem;
	}
	figure {
		grid-column: 2;
		grid-row: 1 / 3;
		align-self: center;
		position: relative;
		margin: 0;
		justify-self: center;
		width: min(78vh, 100%);
		aspect-ratio: 1;
		border: 1px solid rgb(232 226 214 / 0.12);
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
		opacity: 0.75;
	}
	.empty {
		position: absolute;
		inset: 0;
		display: grid;
		place-items: center;
		font-style: italic;
		font-weight: 300;
		opacity: 0.3;
	}
	.save {
		position: absolute;
		right: 0;
		bottom: -2.8rem;
	}
	@media (max-width: 800px) {
		main {
			grid-template-columns: 1fr;
			grid-template-rows: none;
			gap: 1.5rem;
			padding: 4rem 1rem 4rem;
		}
		figure {
			grid-column: auto;
			grid-row: auto;
			width: 100%;
			margin-bottom: 2.8rem;
		}
	}
	@media (prefers-reduced-motion: reduce) {
		canvas,
		.bar span {
			transition: none;
		}
	}
</style>
