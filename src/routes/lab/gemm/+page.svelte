<script lang="ts">
	import { TernaryGemm, type GemmJob } from '$lib/runtime/gemm';

	let log = $state<string[]>([]);
	let busy = $state(false);
	const say = (s: string) => (log = [...log, s]);

	async function run() {
		busy = true;
		try {
			const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
			if (!adapter) throw new Error('No WebGPU adapter.');
			const device = await adapter.requestDevice({
				requiredLimits: {
					maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
					maxBufferSize: adapter.limits.maxBufferSize
				}
			});
			const shapes: [number, number, number][] = [
				[1536, 3072, 3072],
				[1536, 9216, 3072],
				[1536, 3072, 9216],
				[1536, 27648, 3072]
			];
			for (const [M, N, K] of shapes) {
				// random ternary weights (about a third zeros) and random inputs
				const codes = new Uint32Array((N * K) / 16);
				for (let i = 0; i < codes.length; i++) {
					let w = 0;
					for (let j = 0; j < 16; j++) w |= Math.floor(Math.random() * 3) << (j * 2);
					codes[i] = w >>> 0;
				}
				const scales = new Float32Array((N * K) / 128).map(() => 0.01 + Math.random() * 0.02);
				const X = new Float32Array(M * K).map(() => Math.random() * 2 - 1);
				const mk = (data: ArrayBufferView & { byteLength: number }, usage: number) => {
					const b = device.createBuffer({
						size: data.byteLength,
						usage: usage | GPUBufferUsage.COPY_DST
					});
					device.queue.writeBuffer(b, 0, data as unknown as BufferSource);
					return b;
				};
				const cBuf = mk(codes, GPUBufferUsage.STORAGE);
				const sBuf = mk(scales, GPUBufferUsage.STORAGE);
				const arena = device.createBuffer({
					size: (M * K + M * N) * 4,
					usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
				});
				device.queue.writeBuffer(arena, 0, X);
				const gemm = new TernaryGemm(device, cBuf, sBuf, arena, 64);
				const job: GemmJob = { M, N, K, codes: 0, scales: 0, x: 0, y: M * K };
				gemm.prepare([job]);
				const reps = 10;
				const time = async (n: number) => {
					const enc = device.createCommandEncoder();
					const pass = enc.beginComputePass();
					for (let r = 0; r < n; r++) gemm.dispatch(pass, 0, job);
					pass.end();
					const t0 = performance.now();
					device.queue.submit([enc.finish()]);
					await device.queue.onSubmittedWorkDone();
					return performance.now() - t0;
				};
				await time(1); // warm up
				const ms = (await time(reps)) / reps;
				// check a few outputs against the CPU
				const rd = device.createBuffer({
					size: M * N * 4,
					usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
				});
				const enc = device.createCommandEncoder();
				enc.copyBufferToBuffer(arena, M * K * 4, rd, 0, M * N * 4);
				device.queue.submit([enc.finish()]);
				await rd.mapAsync(GPUMapMode.READ);
				const Y = new Float32Array(rd.getMappedRange());
				let worst = 0;
				for (let q = 0; q < 64; q++) {
					const m = Math.floor(Math.random() * M),
						n = Math.floor(Math.random() * N);
					let ref = 0;
					for (let k = 0; k < K; k++) {
						const t = ((codes[(n * K + k) >> 4] >>> ((k % 16) * 2)) & 3) - 1;
						ref += t * scales[(n * K + k) >> 7] * X[m * K + k];
					}
					worst = Math.max(worst, Math.abs(ref - Y[m * N + n]) / (Math.abs(ref) + 1e-3));
				}
				rd.unmap();
				say(
					`${M}x${N}x${K}: ${ms.toFixed(2)} ms, ${((2 * M * N * K) / ms / 1e9).toFixed(2)} TFLOP/s, worst relative error ${worst.toExponential(1)}`
				);
				for (const b of [cBuf, sBuf, arena, rd]) b.destroy();
			}
		} catch (e) {
			say(`error: ${e instanceof Error ? e.message : String(e)}`);
		}
		busy = false;
	}
</script>

<svelte:head><title>Ternary GEMM bench</title></svelte:head>

<main>
	<h1>Ternary GEMM, many tokens at once</h1>
	<button onclick={run} disabled={busy}>Run the benchmark</button>
	<ol>
		{#each log as line, i (i)}
			<li>{line}</li>
		{/each}
	</ol>
</main>

<style>
	main {
		padding: 4vh 6vw;
		font-size: 0.95rem;
		line-height: 1.5;
	}
	h1 {
		font-weight: 300;
		font-style: italic;
		font-size: 1.6rem;
	}
	button {
		font: inherit;
		background: none;
		color: var(--bone);
		border: 1px solid rgb(232 226 214 / 0.3);
		padding: 0.4rem 0.8rem;
	}
</style>
