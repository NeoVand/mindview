<script lang="ts" module>
	import type { Component } from 'svelte';

	export interface NavLink {
		href: string;
		label: string;
		icon?: Component<{ size?: number | string; strokeWidth?: number | string }>;
		current?: boolean;
	}
</script>

<script lang="ts">
	// The bar along the top of every page: the mark home, the labs (on a lab), and the rest of the site. It lies over
	// the canvas, so only its own controls catch the pointer.
	import Mark from './Mark.svelte';
	import GithubMark from './GithubMark.svelte';

	let {
		home,
		links,
		labs = [],
		code = 'https://github.com/NeoVand/mindview',
		newTab = false
	}: {
		home: string;
		links: NavLink[];
		labs?: NavLink[]; // the labs, on a lab page
		code?: string;
		newTab?: boolean; // links open in a new tab (in the Space, which runs in a frame)
	} = $props();
	const target = $derived(newTab ? '_blank' : undefined);
	const rel = $derived(newTab ? 'noopener' : undefined);
</script>

<header class="site">
	<!-- eslint-disable svelte/no-navigation-without-resolve -- the caller resolves every link (the site) or gives full URLs (the Space) -->
	<a class="brand" href={home} {target} {rel} aria-label="mindview, the first piece">
		<Mark />
		<span>mindview</span>
	</a>
	{#if labs.length}
		<nav class="site-labs" aria-label="Labs">
			<div class="seg">
				{#each labs as l (l.href)}
					<a href={l.href} aria-current={l.current ? 'page' : undefined}>
						{#if l.icon}<l.icon strokeWidth={1.75} />{/if}
						<span>{l.label}</span>
					</a>
				{/each}
			</div>
		</nav>
	{/if}
	<nav class="links" aria-label="mindview">
		{#each links as l (l.href)}
			<a
				class="btn quiet"
				href={l.href}
				{target}
				{rel}
				aria-current={l.current ? 'page' : undefined}
			>
				{#if l.icon}<l.icon strokeWidth={1.75} />{/if}
				<span>{l.label}</span>
			</a>
		{/each}
		<a
			class="btn quiet icon"
			href={code}
			target="_blank"
			rel="noopener"
			aria-label="The code, on GitHub"
		>
			<GithubMark />
		</a>
	</nav>
	<!-- eslint-enable svelte/no-navigation-without-resolve -->
</header>

<style>
	.site {
		position: fixed;
		top: 0;
		left: 0;
		right: 0;
		z-index: 20;
		display: flex;
		align-items: center;
		gap: 1rem;
		height: var(--header-h);
		padding: 0 calc(var(--gutter) - 0.4rem) 0 var(--gutter);
		background: linear-gradient(rgb(0 0 0 / 0.8), rgb(0 0 0 / 0.45) 65%, rgb(0 0 0 / 0));
		pointer-events: none;
	}
	.site > * {
		pointer-events: auto;
	}
	.brand {
		display: inline-flex;
		align-items: center;
		gap: 0.6rem;
		margin-right: auto;
		color: var(--bone);
		font: 500 1.2rem / 1 var(--serif);
		letter-spacing: 0.005em;
		text-decoration: none;
	}
	.brand:hover span {
		color: #fff;
	}
	.links {
		display: flex;
		align-items: center;
		gap: 0.15rem;
	}
	.links a[aria-current='page'] {
		color: var(--bone);
		background: var(--wash);
	}
	.site-labs {
		position: absolute;
		left: 50%;
		transform: translateX(-50%);
	}
	@media (max-width: 1080px) {
		/* the labs move to a row of their own, under the bar */
		.site-labs {
			top: calc(var(--header-h) - 0.35rem);
			left: var(--gutter);
			right: var(--gutter);
			transform: none;
			overflow-x: auto;
			scrollbar-width: none;
		}
		.site-labs::-webkit-scrollbar {
			display: none;
		}
		.site-labs .seg {
			flex-wrap: nowrap;
			background: rgb(8 8 10 / 0.7);
			-webkit-backdrop-filter: blur(12px);
			backdrop-filter: blur(12px);
		}
	}
	@media (max-width: 560px) {
		.site {
			gap: 0.4rem;
		}
		.brand {
			font-size: 1.1rem;
		}
		.links .btn:not(.icon) {
			padding: 0 0.6rem;
		}
		/* on a phone the labs keep their icons and the current one its name */
		.site-labs a:not([aria-current='page']) span {
			display: none;
		}
	}
</style>
