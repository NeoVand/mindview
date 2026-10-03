<script lang="ts">
	// When the models will not fit this device: why, the recorded run instead, or trying anyway.
	import { resolve } from '$app/paths';
	import CirclePlay from '@lucide/svelte/icons/circle-play';
	import Monitor from '@lucide/svelte/icons/monitor';

	let {
		why,
		ontry,
		home = true, // offer the recorded run (a link to the landing)
		recorded // or play it here (on the landing)
	}: { why: string; ontry: () => void; home?: boolean; recorded?: () => void } = $props();
</script>

<div
	class="needs glass"
	role="alertdialog"
	aria-labelledby="needs-title"
	aria-describedby="needs-why"
>
	<Monitor />
	<h2 id="needs-title">Best on a computer</h2>
	<p id="needs-why">{why}</p>
	<div class="actions">
		{#if recorded}
			<button type="button" class="btn primary" onclick={recorded}
				><CirclePlay />Watch the recorded run</button
			>
		{:else if home}
			<a class="btn primary" href={resolve('/')}><CirclePlay />Watch a recorded run</a>
		{/if}
		<button type="button" class="btn" onclick={ontry}>Try anyway</button>
	</div>
</div>

<style>
	.needs {
		position: absolute;
		left: 50%;
		top: 50%;
		z-index: 10;
		box-sizing: border-box;
		width: min(28rem, calc(100% - 2 * var(--gutter)));
		padding: 1.4rem 1.5rem 1.5rem;
		transform: translate(-50%, -50%);
	}
	.needs > :global(svg) {
		width: 1.5rem;
		height: 1.5rem;
		color: var(--bone-2);
	}
	h2 {
		margin: 0.7rem 0 0;
		font: 400 var(--t-xl) / 1.2 var(--serif);
	}
	p {
		margin: 0.6rem 0 0;
		color: var(--bone-2);
		font: 300 var(--t-md) / 1.5 var(--serif);
	}
	.actions {
		display: flex;
		flex-wrap: wrap;
		gap: 0.6rem;
		margin-top: 1.2rem;
	}
</style>
