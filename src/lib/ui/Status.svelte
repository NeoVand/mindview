<script lang="ts">
	// A note at the bottom left while something loads (with how far it has got), or what went wrong.
	let {
		text,
		progress,
		error = false
	}: { text: string; progress?: number | null; error?: boolean } = $props();
</script>

<div class="status glass" role={error ? 'alert' : 'status'}>
	<p class:error>{text}</p>
	{#if progress != null}
		<div class="progress"><span style:transform="scaleX({progress})"></span></div>
	{/if}
</div>

<style>
	.status {
		position: absolute;
		left: var(--gutter);
		bottom: max(1.25rem, env(safe-area-inset-bottom));
		z-index: 5;
		box-sizing: border-box;
		width: min(30rem, calc(100% - 2 * var(--gutter)));
		padding: 0.85rem 1.1rem 1rem;
		color: var(--bone);
		font: 400 var(--t-base) / 1.5 var(--sans);
	}
	p {
		margin: 0;
	}
	p.error {
		color: var(--ember);
	}
	.progress {
		margin-top: 0.75rem;
	}
</style>
