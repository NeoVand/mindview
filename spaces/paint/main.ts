// The Hugging Face Space: the site's Paint page on its own, with its links pointing back to the site.
import { mount } from 'svelte';
import '../../src/routes/layout.css';
import Paint from '$lib/paint/Paint.svelte';

const SITE = 'https://neovand.github.io/mindview';

mount(Paint, {
	target: document.getElementById('app')!,
	props: {
		elsewhere: [
			{ href: `${SITE}/`, label: 'The piece' },
			{ href: `${SITE}/lab`, label: 'The labs' },
			{ href: 'https://github.com/NeoVand/mindview', label: 'The code' }
		],
		compute: { href: `${SITE}/`, label: 'Watch it compute, on the mindview site' },
		newTab: true
	}
});
