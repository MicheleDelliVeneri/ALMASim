import { redirect } from '@sveltejs/kit';

import type { PageLoad } from './$types';

/** The page was renamed to Combination. */
export const load: PageLoad = ({ url }) => {
	redirect(308, `/combination${url.search}`);
};
