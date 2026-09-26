<script lang="ts">
	import { asset } from '$app/paths';
	import { BonsaiLLM, MAX_TOKENS } from '$lib/runtime/bonsai-llm';
	import { chatPrompt } from '$lib/runtime/tokenizer';

	interface Ref {
		rope_inv_freq: number[];
		rope_attention_scaling: number;
		tokenize: { text: string; ids: number[] }[];
		chat: string;
		chat_ids: number[];
		hidden_shape: number[];
		last_logits_top5: [number, number][];
	}

	let log = $state<string[]>([]);
	let llm = $state<BonsaiLLM | null>(null);
	let prompt = $state('a bonsai tree made of glowing circuitry in a dark museum, volumetric light');
	let busy = $state(false);

	const say = (s: string) => (log = [...log, s]);

	function cosine(a: Float32Array, b: Float32Array) {
		let ab = 0,
			aa = 0,
			bb = 0;
		for (let i = 0; i < a.length; i++) {
			ab += a[i] * b[i];
			aa += a[i] * a[i];
			bb += b[i] * b[i];
		}
		return ab / Math.sqrt(aa * bb);
	}

	async function start() {
		try {
			const adapter = await navigator.gpu?.requestAdapter({ powerPreference: 'high-performance' });
			if (!adapter) throw new Error('No WebGPU adapter.');
			const device = await adapter.requestDevice({
				requiredLimits: {
					maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
					maxBufferSize: adapter.limits.maxBufferSize
				}
			});
			const t0 = performance.now();
			let last = 0;
			const model = await BonsaiLLM.load(
				device,
				asset('/models/ternary-bonsai-1.7b/model.gguf'),
				(f) => {
					if (f - last > 0.1) {
						say(`downloading ${(f * 100).toFixed(0)}%`);
						last = f;
					}
				}
			);
			say(
				`loaded in ${((performance.now() - t0) / 1000).toFixed(1)}s: ${JSON.stringify(model.config)}`
			);

			const ref: Ref = await (await fetch(asset('/lab/ref.json'))).json();
			// rotary table
			let ropeErr = 0;
			ref.rope_inv_freq.forEach(
				(v, i) => (ropeErr = Math.max(ropeErr, Math.abs(v - model.invFreq[i]) / v))
			);
			say(
				`rope: max relative error ${ropeErr.toExponential(2)}, attention scaling ${model.attentionScaling} vs ${ref.rope_attention_scaling}`
			);
			// tokenizer
			for (const tc of ref.tokenize) {
				const ids = model.tokenizer.encode(tc.text);
				const ok = ids.length === tc.ids.length && ids.every((v, i) => v === tc.ids[i]);
				say(
					`tokenize ${ok ? 'OK ' : 'MISMATCH'} ${JSON.stringify(tc.text)} ${ok ? '' : JSON.stringify(ids) + ' vs ' + JSON.stringify(tc.ids)}`
				);
			}
			const chatIds = model.tokenizer.encode(chatPrompt(ref.tokenize[0].text));
			say(
				`chat framing ${JSON.stringify(chatIds) === JSON.stringify(ref.chat_ids) ? 'OK' : 'MISMATCH'} (${chatIds.length} tokens)`
			);

			// forward pass vs PyTorch, layer by layer
			const r = await model.prefill(ref.chat_ids);
			say(`prefill of ${r.ids.length} tokens on the GPU: ${r.ms.toFixed(1)} ms`);
			const hid = new Float32Array(await (await fetch(asset('/lab/ref_hidden.bin'))).arrayBuffer());
			const [Lp1, n, D] = ref.hidden_shape;
			const L = model.layout,
				c = model.config;
			const lines: string[] = [];
			for (let l = 0; l < Lp1; l++) {
				const off = l < Lp1 - 1 ? L.resid + l * MAX_TOKENS * D : L.final; // HF's last hidden state is post final norm
				const ours = await model.read(off, n * D);
				const theirs = hid.subarray(l * n * D, (l + 1) * n * D);
				let worst = 1;
				for (let t = 0; t < n; t++)
					worst = Math.min(
						worst,
						cosine(ours.subarray(t * D, (t + 1) * D), theirs.subarray(t * D, (t + 1) * D))
					);
				lines.push(`${l}:${worst.toFixed(5)}`);
			}
			say(`hidden states, worst per-token cosine by layer: ${lines.join('  ')}`);
			const logits = await model.read(L.logits, c.vocab);
			const top = Array.from(logits.keys())
				.sort((a, b) => logits[b] - logits[a])
				.slice(0, 5);
			say(`next token top-5: ${top.map((i) => `${i}(${logits[i].toFixed(2)})`).join(' ')}`);
			say(
				`PyTorch top-5:    ${ref.last_logits_top5.map(([i, v]) => `${i}(${v.toFixed(2)})`).join(' ')}`
			);
			llm = model;
		} catch (e) {
			say(`error: ${e instanceof Error ? e.message : String(e)}`);
		}
	}

	async function run() {
		if (!llm || busy) return;
		busy = true;
		const ids = llm.tokenizer.encode(chatPrompt(prompt));
		const r = await llm.prefill(ids);
		const logits = await llm.read(llm.layout.logits, llm.config.vocab);
		let best = 0;
		for (let i = 1; i < logits.length; i++) if (logits[i] > logits[best]) best = i;
		say(
			`${r.ids.length} tokens in ${r.ms.toFixed(1)} ms; the model would start its answer with ${JSON.stringify(llm.tokenizer.decode([best]))}`
		);
		busy = false;
	}
</script>

<svelte:head><title>Bonsai runtime check</title></svelte:head>

<main>
	<h1>Ternary Bonsai 1.7B, running in this browser tab</h1>
	{#if !llm}
		<button onclick={start}>Load the model and check it against PyTorch</button>
	{:else}
		<form onsubmit={(e) => (e.preventDefault(), run())}>
			<input bind:value={prompt} aria-label="Prompt" />
			<button disabled={busy}>Run</button>
		</form>
	{/if}
	<ol>
		{#each log as line, i (i)}
			<li>{line}</li>
		{/each}
	</ol>
</main>

<style>
	main {
		padding: 4vh 6vw;
		overflow: auto;
		height: 100vh;
		box-sizing: border-box;
		font-size: 0.95rem;
		line-height: 1.5;
	}
	h1 {
		font-weight: 300;
		font-style: italic;
		font-size: 1.6rem;
	}
	button,
	input {
		font: inherit;
		background: none;
		color: var(--bone);
		border: 1px solid rgb(232 226 214 / 0.3);
		padding: 0.4rem 0.8rem;
	}
	input {
		width: min(40rem, 70vw);
	}
	li {
		word-break: break-word;
		opacity: 0.85;
	}
</style>
