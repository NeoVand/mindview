<script lang="ts">
	import './layout.css';
	import favicon from '$lib/assets/favicon.svg';
	import { page } from '$app/state';
	import { resolve } from '$app/paths';
	import SiteHeader from '$lib/ui/SiteHeader.svelte';
	import Paintbrush from '@lucide/svelte/icons/paintbrush';
	import FlaskConical from '@lucide/svelte/icons/flask-conical';
	import Spline from '@lucide/svelte/icons/spline';
	import Layers2 from '@lucide/svelte/icons/layers-2';
	import Grid3x3 from '@lucide/svelte/icons/grid-3x3';
	import Box from '@lucide/svelte/icons/box';

	let { children } = $props();

	const path = $derived(page.url.pathname.replace(/\/$/, '') || '/');
	const at = (p: string) => path === resolve(p as '/');
	// the bar is on the piece, Paint and the labs (not on the development pages)
	const site = $derived(
		at('/') || at('/paint') || at('/lab') || path.startsWith(resolve('/lab') + '/')
	);
	const links = $derived([
		{ href: resolve('/paint'), label: 'Paint', icon: Paintbrush, current: at('/paint') },
		{
			href: resolve('/lab'),
			label: 'Labs',
			icon: FlaskConical,
			current: path.startsWith(resolve('/lab'))
		}
	]);
	const LABS = [
		{ path: '/lab/threads', label: 'Threads', icon: Spline },
		{ path: '/lab/layer', label: 'One layer', icon: Layers2 },
		{ path: '/lab/machine', label: 'The machine', icon: Grid3x3 },
		{ path: '/lab/cube', label: 'Compute cube', icon: Box }
	] as const;
	const labs = $derived(
		LABS.some((l) => at(l.path))
			? LABS.map((l) => ({
					href: resolve(l.path),
					label: l.label,
					icon: l.icon,
					current: at(l.path)
				}))
			: []
	);
</script>

<svelte:head><link rel="icon" href={favicon} /></svelte:head>
{#if site}
	<SiteHeader home={resolve('/')} {links} {labs} />
{/if}
{@render children()}
