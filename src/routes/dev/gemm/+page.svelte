<script lang="ts">
	// The ternary GEMM on the painter's own shapes: the first kernel against the fast path, timed, and both checked
	// against the CPU on sampled outputs. The step total weights each shape by how often a step runs it.
	// ?rows=1088 also times the shapes with the text cut to 64 rows (1,024 image + 64 text).
	import { page } from '$app/state';
	import { labDevice } from '$lib/lab/shared';
	import { TernaryGemm, type GemmJob } from '$lib/runtime/gemm';

	let status = $state('Starting');
	let report = $state('');

	// [M, N, K, how many per step]
	const SHAPES: [number, number, number, number][] = [
		[512, 3072, 3072, 20],
		[1024, 3072, 3072, 20],
		[512, 18432, 3072, 5],
		[1024, 18432, 3072, 5],
		[512, 3072, 9216, 5],
		[1024, 3072, 9216, 5],
		[1536, 27648, 3072, 20],
		[1536, 3072, 12288, 20]
	];
	const extra = Number(page.url.searchParams.get('rows') ?? 0);
	const rank = Number(page.url.searchParams.get('lora') ?? 0); // also time a side branch of this rank
	// ?kinds=old,fast,fast32,f16 picks the kernels (f16: 32-input chunks)
	const kinds = (page.url.searchParams.get('kinds') ?? 'old,fast,f16').split(',');
	if (extra)
		SHAPES.push([extra - 1024, 3072, 3072, 0], [extra, 27648, 3072, 0], [extra, 3072, 12288, 0]);

	async function run() {
		const device = await labDevice();
		const lines: string[] = [];
		const total: Record<string, number> = {};
		for (const [M, N, K, per] of SHAPES) {
			status = `${M} x ${N} x ${K}`;
			const codes = new Uint32Array((N * K) / 16);
			for (let i = 0; i < codes.length; i++) {
				let w = 0;
				for (let j = 0; j < 16; j++) w |= Math.floor(Math.random() * 3) << (j * 2);
				codes[i] = w >>> 0;
			}
			const scales = new Float32Array((N * K) / 128).map(() => 0.01 + Math.random() * 0.02);
			const X = new Float32Array(M * K).map(() => Math.random() * 2 - 1);
			const T = new Float32Array(M * rank).map(() => Math.random() * 2 - 1);
			const Bt = new Float32Array(Math.max(4, rank * N)).map(() => (Math.random() * 2 - 1) * 0.01);
			const mk = (data: Float32Array | Uint32Array, usage: number) => {
				const b = device.createBuffer({
					size: data.byteLength,
					usage: usage | GPUBufferUsage.COPY_DST
				});
				device.queue.writeBuffer(b, 0, data);
				return b;
			};
			const cBuf = mk(codes, GPUBufferUsage.STORAGE);
			const sBuf = mk(scales, GPUBufferUsage.STORAGE);
			const arena = device.createBuffer({
				size: (M * K + M * N + M * rank) * 4,
				usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC
			});
			device.queue.writeBuffer(arena, 0, X);
			device.queue.writeBuffer(arena, (M * K + M * N) * 4, T);
			const lBuf = mk(Bt, GPUBufferUsage.STORAGE);
			const base: GemmJob = { M, N, K, codes: 0, scales: 0, x: 0, y: M * K };
			const out: Record<string, string> = {};
			for (const kind of [...kinds, ...(rank ? ['lora', 'lora16'] : [])]) {
				const gemm = new TernaryGemm(
					device,
					cBuf,
					sBuf,
					arena,
					4,
					kind === 'old' ? 0 : kind.endsWith('32') ? 32 : 16,
					lBuf
				);
				gemm.precision = kind.startsWith('f16') || kind === 'lora16' ? 'f16' : 'f32';
				const job: GemmJob = {
					...base,
					...(kind.startsWith('lora') ? { lora: { t: M * K + M * N, b: 0, rank } } : {})
				};
				gemm.prepare([job]);
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
				await time(2);
				const reps = Math.max(3, Math.round(4e11 / (2 * M * N * K)));
				const ms = (await time(reps)) / reps;
				total[kind] = (total[kind] ?? 0) + ms * per;
				const rd = device.createBuffer({
					size: M * N * 4,
					usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
				});
				const enc = device.createCommandEncoder();
				enc.copyBufferToBuffer(arena, M * K * 4, rd, 0, M * N * 4);
				device.queue.submit([enc.finish()]);
				await rd.mapAsync(GPUMapMode.READ);
				const Y = new Float32Array(rd.getMappedRange());
				let worst = 0,
					ss = 0; // the largest error, and the sum of squares of the references (the error is reported relative to their rms)
				for (let q = 0; q < 48; q++) {
					const m = Math.floor(Math.random() * M),
						n = Math.floor(Math.random() * N);
					let ref = 0;
					for (let k = 0; k < K; k++) {
						const t = ((codes[(n * K + k) >> 4] >>> ((k % 16) * 2)) & 3) - 1;
						ref += t * scales[(n * K + k) >> 7] * X[m * K + k];
					}
					if (kind.startsWith('lora'))
						for (let j = 0; j < rank; j++) ref += T[m * rank + j] * Bt[j * N + n];
					worst = Math.max(worst, Math.abs(ref - Y[m * N + n]));
					ss += ref * ref;
				}
				rd.unmap();
				worst /= Math.sqrt(ss / 48);
				rd.destroy();
				out[kind] =
					`${ms.toFixed(2).padStart(7)} ms ${((2 * M * N * K) / ms / 1e9).toFixed(2).padStart(5)} TFLOP/s err ${worst.toExponential(0)}`;
			}
			lines.push(
				`${`${M}x${N}x${K}`.padEnd(18)} x${String(per).padEnd(3)} ${Object.entries(out)
					.map(([k, v]) => `${k} ${v}`)
					.join(' | ')}`
			);
			for (const b of [cBuf, sBuf, arena, lBuf]) b.destroy();
		}
		lines.push(
			'',
			`ternary GEMMs of one step: ${Object.entries(total)
				.map(([k, ms]) => `${k} ${ms.toFixed(0)} ms`)
				.join(', ')}`
		);
		report = lines.join('\n');
		status = 'done';
	}

	function mount() {
		run().catch((e) => (status = `error: ${e instanceof Error ? e.message : e}`));
	}
</script>

<svelte:head><title>Ternary GEMM bench</title></svelte:head>

<main {@attach mount}>
	<p>{status}</p>
	{#if report}<pre id="result">{report}</pre>{/if}
</main>

<style>
	main {
		position: fixed;
		inset: 0;
		overflow: auto;
		padding: 1.5rem;
		background: var(--void);
		color: var(--bone);
	}
	pre {
		font-size: 0.8rem;
		line-height: 1.4;
	}
</style>
