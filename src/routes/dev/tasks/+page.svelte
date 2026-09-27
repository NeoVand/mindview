<script lang="ts">
	// Fast (2 steps, the first at 256 x 256) through the painter's task path (encodeTasks, stepTasks, sketchTasks,
	// finalTasks on the scheduler, as the threads piece runs it), beside the same painting through step() and decode()
	// (as /paint runs it): the two must match. Shows the decoder's stages of each pass as they are made.
	import { labDevice } from '$lib/lab/shared';
	import { packedUrl } from '$lib/models';
	import { BonsaiLLM } from '$lib/runtime/bonsai-llm';
	import { PackedModel } from '$lib/runtime/packed';
	import { Painter } from '$lib/runtime/painter';
	import { GpuScheduler } from '$lib/runtime/scheduler';

	const prompt = 'a lighthouse on a cliff during a thunderstorm, oil painting';
	const seed = 7;
	let status = $state('Loading');
	let diffs = $state<string[]>([]);
	const rms = (a: Float32Array, b: Float32Array) => {
		let d = 0,
			n = 0;
		for (let i = 0; i < a.length; i++) {
			d += (a[i] - b[i]) ** 2;
			n += b[i] ** 2;
		}
		return Math.sqrt(d / Math.max(1e-12, n)).toExponential(2);
	};
	let shots = $state<{ label: string; url: string }[]>([]);

	async function snap(device: GPUDevice, tex: GPUTexture, label: string) {
		const w = tex.width,
			h = tex.height;
		const bpr = Math.ceil((w * 4) / 256) * 256;
		const buf = device.createBuffer({
			size: bpr * h,
			usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
		});
		const enc = device.createCommandEncoder();
		enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr }, [w, h]);
		device.queue.submit([enc.finish()]);
		await buf.mapAsync(GPUMapMode.READ);
		const src = new Uint8Array(buf.getMappedRange());
		const px = new Uint8ClampedArray(w * h * 4);
		for (let y = 0; y < h; y++) px.set(src.subarray(y * bpr, y * bpr + w * 4), y * w * 4);
		buf.destroy();
		const c = new OffscreenCanvas(w, h);
		c.getContext('2d')!.putImageData(new ImageData(px, w, h), 0, 0);
		shots = [
			...shots,
			{ label, url: URL.createObjectURL(await c.convertToBlob({ type: 'image/png' })) }
		];
	}

	async function run() {
		const device = await labDevice();
		const packed = await PackedModel.open(packedUrl());
		const llm = BonsaiLLM.fromGGUF(device, await packed.readerGGUF());
		const painter = await Painter.fromPacked(device, packed);
		// the task path
		status = 'Task path';
		const sch = new GpuScheduler(device);
		painter.setSteps(2);
		painter.textLength = 256;
		painter.setNoise(undefined, seed, true);
		const { tasks } = painter.encodeTasks(llm, prompt, sch.slice);
		sch.push(...tasks);
		const probe: Record<string, Float32Array> = {};
		const at = painter.at;
		// checkpoints: the conditioning after the text side, the latent after the first step
		sch.push({
			cost: 0,
			record: () => {},
			done: async () => {
				probe.ctx = await painter.read(at.ctxd, 256 * 3072);
				probe.lat0 = await painter.read(at.lat, 256 * 128);
			}
		});
		const got: Promise<void>[] = [];
		for (let s = 0; s < 2; s++) {
			if (s === 1)
				sch.push({
					cost: 0,
					record: () => {},
					done: async () => {
						probe.lat1 = await painter.read(at.lat, 1024 * 128);
					}
				});
			sch.push(
				...painter.stepTasks(s, { first: 3, count: 8 }, () => {}, {
					record: (enc, res) => {
						// copy each stage as it is made (the next pass reuses the textures)
						const src = res === 512 ? painter.taef2.texture : painter.taef2.stageTextures.get(res)!;
						const tex = device.createTexture({
							size: [src.width, src.height],
							format: 'rgba8unorm',
							usage: GPUTextureUsage.COPY_SRC | GPUTextureUsage.COPY_DST
						});
						enc.copyTextureToTexture({ texture: src }, { texture: tex }, [src.width, src.height]);
						got.push(
							device.queue
								.onSubmittedWorkDone()
								.then(() => snap(device, tex, `pass ${s + 1}, ${res}`))
						);
					}
				})
			);
		}
		let finished = false;
		sch.push(
			...painter.finalTasks(() => {
				finished = true;
			})
		);
		const t0 = performance.now();
		while (!finished) {
			sch.pump();
			await new Promise((r) => requestAnimationFrame(r));
		}
		await Promise.all(got);
		await snap(
			device,
			painter.taef2.texture,
			`tasks: final (${((performance.now() - t0) / 1000).toFixed(1)} s)`
		);
		// the direct path, as /paint
		status = 'Direct path';
		painter.setSteps(2);
		painter.textLength = 256;
		await painter.encode(llm, prompt);
		diffs = [
			...diffs,
			`conditioning: relative difference ${rms(probe.ctx, await painter.read(at.ctxd, 256 * 3072))}`
		];
		painter.setNoise(undefined, seed, true);
		diffs = [
			...diffs,
			`noise before step 0: ${rms(probe.lat0, await painter.read(at.lat, 256 * 128))}`
		];
		await painter.step(0);
		const clean = await painter.upsampleLatent(seed + 1);
		diffs = [
			...diffs,
			`latent after the sketch: ${rms(probe.lat1, await painter.read(at.lat, 1024 * 128))}`
		];
		await painter.decode(0, clean);
		await snap(device, painter.taef2.texture, 'direct: sketch');
		await painter.step(1);
		await painter.decode();
		await snap(device, painter.taef2.texture, 'direct: final');
		status = 'done';
	}

	function mount() {
		run().catch((e) => (status = `error: ${e instanceof Error ? e.message : e}`));
	}
</script>

<svelte:head><title>Task path</title></svelte:head>

<main {@attach mount}>
	<p class="status">{status}</p>
	{#each diffs as d (d)}<p class="diff">{d}</p>{/each}
	<div class="grid">
		{#each shots as s (s.url)}
			<figure>
				<img src={s.url} alt={s.label} />
				<figcaption>{s.label}</figcaption>
			</figure>
		{/each}
	</div>
</main>

<style>
	main {
		position: fixed;
		inset: 0;
		overflow: auto;
		padding: 1rem;
		background: var(--void);
		color: var(--bone);
	}
	.grid {
		display: flex;
		flex-wrap: wrap;
		gap: 8px;
	}
	img {
		width: 256px;
		height: 256px;
		display: block;
		image-rendering: pixelated;
	}
	figcaption {
		font-size: 0.75rem;
	}
</style>
