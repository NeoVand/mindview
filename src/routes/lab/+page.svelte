<script lang="ts">
	import { resolve } from '$app/paths';

	const labs = [
		{
			href: '/lab/threads',
			name: 'Threads',
			what: 'Each word of your prompt is a thread through the reader’s layers; then the painter paints it, block by block.',
			shows: 'Where each word goes, what it reads, and how the picture reads the words.'
		},
		{
			href: '/lab/layer',
			name: 'One layer',
			what: 'One word through one layer of the reader, or one word or patch of the picture through one block of the painter, operation by operation, every number real and readable: the norms, the projections, the rotary clocks, attention (in the painter, 24 heads over the prompt and all 1,024 patches), the neurons, the residual adds, and the painter’s scale, shift and gates set by the time step.',
			shows:
				'Exactly what a layer or a block does, and that neither model multiplies by its weights: each ternary weight adds, subtracts or skips.'
		},
		{
			href: '/lab/machine',
			name: 'The machine',
			what: 'The reader’s 1.4 billion weights, the adapter and the painter’s 3.7 billion on one wall at one cell each, lit by what they multiply for the word or patch you choose, with the picture each block has in mind above it. Zoom from the whole pipeline down to a single weight.',
			shows: 'The scale, and how little of the machine one word or one patch lights up.'
		},
		{
			href: '/lab/cube',
			name: 'Compute cube',
			what: 'Every matrix product as a box of its multiplications, one voxel per multiply, collapsing into the sums that become the next vector: the reader’s forward pass, then the painting below it (the reader again at 512 tokens, the adapter, and the painter’s 25 blocks over 1,536 rows, four times).',
			shows:
				'Where the computation actually goes: the painting costs hundreds of times the reading.'
		}
	] as const;
</script>

<svelte:head><title>Labs</title></svelte:head>

<main>
	<h1>Labs</h1>
	<p class="lede">
		Four ways of looking at the same computation, from the words to the picture. Each runs the
		models live in this tab: nothing is recorded.
	</p>
	<ol>
		{#each labs as lab (lab.href)}
			<li>
				<a href={resolve(lab.href)}>{lab.name}</a>
				<p>{lab.what}</p>
				<p class="shows">{lab.shows}</p>
			</li>
		{/each}
	</ol>
</main>

<style>
	main {
		position: fixed;
		inset: 0;
		overflow-y: auto;
		padding: 5rem 2.2rem 4rem;
		background: var(--void);
	}
	h1 {
		margin: 0;
		font-weight: 300;
		font-style: italic;
		font-size: clamp(2rem, 4vw, 3.4rem);
		letter-spacing: -0.01em;
	}
	.lede {
		max-width: 52ch;
		margin: 0.8rem 0 3rem;
		font-weight: 300;
		line-height: 1.5;
		opacity: 0.7;
	}
	ol {
		list-style: none;
		margin: 0;
		padding: 0;
		display: grid;
		gap: 2.4rem;
		max-width: 60rem;
	}
	li {
		display: grid;
		grid-template-columns: 11rem 1fr;
		column-gap: 2rem;
		row-gap: 0.3rem;
		border-top: 1px solid rgb(232 226 214 / 0.14);
		padding-top: 1.1rem;
	}
	li a {
		grid-row: span 2;
		color: var(--bone);
		font-size: 1.35rem;
		font-weight: 300;
		text-decoration: none;
		border-bottom: 1px solid transparent;
		align-self: start;
	}
	li a:hover,
	li a:focus-visible {
		border-bottom-color: var(--ember);
		outline: none;
	}
	li p {
		margin: 0;
		max-width: 62ch;
		font-weight: 300;
		line-height: 1.5;
		opacity: 0.8;
	}
	li .shows {
		opacity: 0.5;
		font-style: italic;
	}
	@media (max-width: 40rem) {
		li {
			grid-template-columns: 1fr;
		}
		li a {
			grid-row: auto;
		}
	}
</style>
