<script lang="ts">
	// The bottom of a threads page: what is happening now (the narrator), and where the ride is along the whole
	// computation, with what you can do from there. Used by the landing and by /lab/threads.
	import type { Snippet } from 'svelte';
	import Pause from '@lucide/svelte/icons/pause';
	import Play from '@lucide/svelte/icons/play';
	import Rewind from '@lucide/svelte/icons/rewind';
	import Scan from '@lucide/svelte/icons/scan';
	import ImageIcon from '@lucide/svelte/icons/image';
	import Route from '@lucide/svelte/icons/route';
	import ChevronDown from '@lucide/svelte/icons/chevron-down';

	let {
		caption,
		front,
		ride,
		total,
		marks,
		mode,
		done,
		paused,
		major = () => false,
		note,
		onscrub,
		onpause,
		onfollow,
		onrideback,
		onpicture
	}: {
		caption: string;
		front: number; // how far the computation has got (0..total)
		ride: number; // where the camera is
		total: number;
		marks: number[];
		mode: string;
		done: boolean;
		paused: boolean;
		major?: (m: number) => boolean;
		note?: Snippet; // under the caption (a download, say)
		onscrub: (at: number) => void;
		onpause: () => void;
		onfollow: () => void;
		onrideback: () => void;
		onpicture: () => void;
	} = $props();

	let scrubbing = false;
	let folded = $state(false); // on a phone the narrator can fold to two lines

	function scrub(e: PointerEvent) {
		const box = (e.currentTarget as HTMLElement).getBoundingClientRect();
		onscrub(Math.max(0, Math.min(1, (e.clientX - box.left) / box.width)) * total);
	}

	function key(e: KeyboardEvent) {
		const step = e.shiftKey ? 1 : 0.25;
		if (e.key === 'ArrowLeft') onscrub(Math.max(0, ride - step));
		else if (e.key === 'ArrowRight') onscrub(Math.min(front, ride + step));
		else return;
		e.preventDefault();
	}
</script>

<section class="journey" aria-label="The computation">
	<div class="glass narrator" class:folded>
		<p class="narration" aria-live="polite">{caption}</p>
		{@render note?.()}
		<button
			type="button"
			class="fold btn quiet icon"
			aria-label={folded ? 'Show the whole caption' : 'Fold the caption'}
			aria-expanded={!folded}
			onclick={() => (folded = !folded)}><ChevronDown /></button
		>
	</div>
	<div class="row">
		<button
			type="button"
			class="btn icon"
			onclick={onpause}
			aria-label={paused ? 'Play' : 'Pause'}
			title={paused ? 'Play (space)' : 'Pause (space)'}
		>
			{#if paused}<Play />{:else}<Pause />{/if}
		</button>
		<div
			class="track"
			role="slider"
			tabindex="0"
			aria-label="Where you are along the computation"
			aria-valuemin={0}
			aria-valuemax={total}
			aria-valuenow={Math.round(ride)}
			onkeydown={key}
			onpointerdown={(e) => {
				scrubbing = true;
				(e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
				scrub(e);
			}}
			onpointermove={(e) => scrubbing && scrub(e)}
			onpointerup={() => (scrubbing = false)}
			onpointercancel={() => (scrubbing = false)}
		>
			<span class="done" style:transform="scaleX({front / total})"></span>
			{#each marks as m, i (i)}
				<span class="tick" class:major={major(m)} style:left="{(m / total) * 100}%"></span>
			{/each}
			<span class="here" style:left="{(ride / total) * 100}%"></span>
		</div>
		{#if mode !== 'live'}
			<div class="actions">
				{#if front < total - 0.01}
					<button type="button" class="btn" onclick={onfollow} aria-label="Ride along again"
						><Route /><span>Ride along</span></button
					>
				{:else}
					{#if mode !== 'rewind'}
						<button type="button" class="btn" onclick={onrideback} aria-label="Ride back"
							><Rewind /><span>Ride back</span></button
						>
					{/if}
					{#if mode !== 'overview'}
						<button type="button" class="btn" onclick={onfollow} aria-label="See it whole"
							><Scan /><span>See it whole</span></button
						>
					{/if}
					{#if mode !== 'finale' && done}
						<button type="button" class="btn" onclick={onpicture} aria-label="The picture"
							><ImageIcon /><span>The picture</span></button
						>
					{/if}
				{/if}
			</div>
		{/if}
	</div>
</section>

<style>
	.journey {
		position: absolute;
		left: var(--gutter);
		right: var(--gutter);
		bottom: max(1.1rem, env(safe-area-inset-bottom));
		display: flex;
		flex-direction: column;
		align-items: flex-start;
		gap: 0.8rem;
		pointer-events: none;
	}
	.journey > * {
		pointer-events: auto;
	}
	.narrator {
		position: relative;
		max-width: 40rem;
		padding: 0.85rem 1.1rem 0.9rem;
	}
	.narrator :global(.hint) {
		margin: 0.5rem 0 0;
	}
	.fold {
		display: none;
	}
	.row {
		display: flex;
		align-items: center;
		gap: 0.9rem;
		align-self: stretch;
	}
	.actions {
		display: flex;
		gap: 0.4rem;
	}
	.track {
		position: relative;
		flex: 1;
		height: 2.25rem;
		cursor: ew-resize;
		touch-action: none;
		border-radius: var(--r-sm);
	}
	.track::before {
		content: '';
		position: absolute;
		left: 0;
		right: 0;
		top: 50%;
		height: 2px;
		margin-top: -1px;
		border-radius: 2px;
		background: var(--hair-2);
	}
	.done {
		position: absolute;
		left: 0;
		right: 0;
		top: 50%;
		height: 2px;
		margin-top: -1px;
		border-radius: 2px;
		background: var(--bone-2);
		transform-origin: left;
	}
	.tick {
		position: absolute;
		top: calc(50% - 4px);
		width: 1px;
		height: 8px;
		background: rgb(236 229 216 / 0.3);
	}
	.tick.major {
		top: calc(50% - 7px);
		height: 14px;
		background: rgb(236 229 216 / 0.45);
	}
	.here {
		position: absolute;
		top: 50%;
		width: 12px;
		height: 12px;
		margin: -6px 0 0 -6px;
		border-radius: 50%;
		background: var(--ember);
		box-shadow: 0 0 0 4px rgb(255 176 74 / 0.18);
	}
	@media (max-width: 720px) {
		.journey {
			gap: 0.55rem;
		}
		.narrator {
			box-sizing: border-box;
			width: 100%;
			padding: 0.7rem 2.6rem 0.75rem 0.9rem;
		}
		.narrator .narration {
			font-size: 0.95rem;
			line-height: 1.5;
		}
		.narrator.folded .narration {
			display: -webkit-box;
			-webkit-line-clamp: 2;
			line-clamp: 2;
			-webkit-box-orient: vertical;
			overflow: hidden;
		}
		.fold {
			display: inline-flex;
			position: absolute;
			top: 0.3rem;
			right: 0.3rem;
			width: 2rem;
			height: 2rem;
		}
		.narrator:not(.folded) .fold :global(svg) {
			transform: rotate(180deg);
		}
		.row {
			gap: 0.5rem;
		}
		.actions .btn {
			width: 2.25rem;
			padding: 0;
		}
		.actions .btn span {
			display: none;
		}
	}
</style>
