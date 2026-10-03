<script lang="ts">
	import { asset, resolve } from '$app/paths';
	import Spline from '@lucide/svelte/icons/spline';
	import Layers2 from '@lucide/svelte/icons/layers-2';
	import Grid3x3 from '@lucide/svelte/icons/grid-3x3';
	import Box from '@lucide/svelte/icons/box';
	import Paintbrush from '@lucide/svelte/icons/paintbrush';
	import ArrowRight from '@lucide/svelte/icons/arrow-right';

	const labs = [
		{
			href: '/lab/threads',
			name: 'Threads',
			icon: Spline,
			preview: 'threads',
			what: 'Each word of your prompt is a thread through the reader’s layers; then the painter paints it, block by block.',
			shows: 'Where each word goes, what it reads, and how the picture reads the words.'
		},
		{
			href: '/lab/layer',
			name: 'One layer',
			icon: Layers2,
			preview: 'layer',
			what: 'One word through one layer of the reader, or one word or patch of the picture through one block of the painter, operation by operation, every number real and readable: the norms, the projections, the rotary clocks, attention (in the painter, 24 heads over the prompt and all 1,024 patches), the neurons, the residual adds, and the painter’s scale, shift and gates set by the time step.',
			shows:
				'Exactly what a layer or a block does, and that neither model multiplies by its weights: each ternary weight adds, subtracts or skips.'
		},
		{
			href: '/lab/machine',
			name: 'The machine',
			icon: Grid3x3,
			preview: 'machine',
			what: 'The reader’s 1.4 billion weights, the adapter and the painter’s 3.7 billion on one wall at one cell each, lit by what they multiply for the word or patch you choose, with the picture each block has in mind above it. Zoom from the whole pipeline down to a single weight.',
			shows: 'The scale, and how little of the machine one word or one patch lights up.'
		},
		{
			href: '/lab/cube',
			name: 'Compute cube',
			icon: Box,
			preview: 'cube',
			what: 'Every matrix product as a box of its multiplications, one voxel per multiply, collapsing into the sums that become the next vector: the reader’s forward pass, then the painting below it (the reader again at 512 tokens, the adapter, and the painter’s 25 blocks over 1,536 rows, four times).',
			shows:
				'Where the computation actually goes: the painting costs hundreds of times the reading.'
		}
	] as const;
</script>

<svelte:head><title>Labs</title></svelte:head>

<main>
	<header class="intro">
		<h1>Labs</h1>
		<p class="lede">
			Four ways of looking at the same computation, from the words to the picture. Each runs the
			models live in this tab: nothing is recorded, and every number on screen is one the models
			computed.
		</p>
		<p class="hint">
			The labs load the whole reader and the whole painter, about 1.6 GB, once; a computer with a
			recent graphics card shows them best.
		</p>
	</header>
	<ol>
		{#each labs as lab (lab.href)}
			<li>
				<a href={resolve(lab.href)}>
					<img
						src={asset(`/previews/${lab.preview}.webp`)}
						alt=""
						loading="lazy"
						width="1200"
						height="750"
					/>
					<span class="name"><lab.icon strokeWidth={1.5} />{lab.name}<ArrowRight class="go" /></span
					>
				</a>
				<p>{lab.what}</p>
				<p class="shows">{lab.shows}</p>
			</li>
		{/each}
	</ol>
	<p class="coda">
		<Paintbrush />
		<span
			>The same pipeline, cut to what painting needs and packed into one file of about a gigabyte:
			<a href={resolve('/paint')}>paint with it</a>.</span
		>
	</p>
</main>

<style>
	main {
		position: fixed;
		inset: 0;
		overflow-y: auto;
		box-sizing: border-box;
		padding: calc(var(--header-h) + 2.5rem) var(--gutter) 4rem;
		background: var(--void);
	}
	.intro {
		max-width: 44rem;
	}
	h1 {
		margin: 0;
		font: 300 clamp(2.4rem, 5vw, 3.8rem) / 1 var(--serif);
		letter-spacing: -0.015em;
	}
	.lede {
		margin: 1rem 0 0;
		color: var(--bone-2);
		font: 300 clamp(1.1rem, 1.8vw, var(--t-lg)) / 1.5 var(--serif);
		text-wrap: pretty;
	}
	.intro .hint {
		margin: 0.9rem 0 0;
	}
	ol {
		display: grid;
		grid-template-columns: repeat(auto-fill, minmax(min(100%, 30rem), 1fr));
		gap: 3rem clamp(1.5rem, 3vw, 3rem);
		max-width: 84rem;
		margin: 3rem 0 0;
		padding: 0;
		list-style: none;
	}
	li {
		display: flex;
		flex-direction: column;
	}
	li a {
		display: flex;
		flex-direction: column;
		gap: 1rem;
		color: var(--bone);
		text-decoration: none;
	}
	img {
		display: block;
		width: 100%;
		height: auto;
		aspect-ratio: 16 / 10;
		object-fit: cover;
		border: 1px solid var(--hair);
		border-radius: var(--r-sm);
		background: #070708;
		transition: border-color 0.2s;
	}
	li a:hover img {
		border-color: var(--hair-2);
	}
	.name {
		display: inline-flex;
		align-items: center;
		gap: 0.6rem;
		font: 400 var(--t-xl) / 1.2 var(--serif);
	}
	.name :global(svg) {
		width: 1.35rem;
		height: 1.35rem;
		color: var(--bone-2);
	}
	.name :global(.go) {
		width: 1.1rem;
		height: 1.1rem;
		color: var(--ember);
		opacity: 0;
		transform: translateX(-0.3rem);
		transition:
			opacity 0.2s,
			transform 0.2s;
	}
	li a:hover .name :global(.go),
	li a:focus-visible .name :global(.go) {
		opacity: 1;
		transform: none;
	}
	li p {
		margin: 0.6rem 0 0;
		max-width: 62ch;
		color: var(--bone-2);
		font: 300 var(--t-md) / 1.55 var(--serif);
		text-wrap: pretty;
	}
	li .shows {
		color: var(--bone);
		font: 400 var(--t-sm) / 1.5 var(--sans);
	}
	.coda {
		display: flex;
		gap: 0.6rem;
		max-width: 44rem;
		margin: 4rem 0 0;
		padding-top: 1.5rem;
		border-top: 1px solid var(--hair);
		color: var(--bone-2);
		font: 300 var(--t-md) / 1.5 var(--serif);
	}
	.coda :global(svg) {
		flex: none;
		width: 1.1rem;
		height: 1.1rem;
		margin-top: 0.2rem;
		color: var(--ember);
	}
	.coda a {
		color: var(--bone);
		text-decoration: none;
		border-bottom: 1px solid var(--ember);
	}
</style>
