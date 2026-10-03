// The Hugging Face Space: the site's Paint page on its own, with its links pointing back to the site.
import { mount } from 'svelte';
import '../../src/routes/layout.css';
import Paint from '$lib/paint/Paint.svelte';
import FlaskConical from '@lucide/svelte/icons/flask-conical';
import Spline from '@lucide/svelte/icons/spline';

const SITE = 'https://neovand.github.io/mindview';

mount(Paint, {
	target: document.getElementById('app')!,
	props: {
		header: {
			home: `${SITE}/`,
			links: [
				{ href: `${SITE}/`, label: 'The piece', icon: Spline },
				{ href: `${SITE}/lab`, label: 'Labs', icon: FlaskConical }
			]
		},
		compute: { href: `${SITE}/`, label: 'Watch it compute, on the mindview site' },
		newTab: true
	}
});
