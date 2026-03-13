// See https://kit.svelte.dev/docs/types#app
// for information about these interfaces
declare global {
	namespace App {
		// interface Error {}
		// interface Locals {}
		// interface PageData {}
		// interface PageState {}
		// interface Platform {}
	}

	interface ImportMetaEnv {
		readonly VITE_API_URL?: string;
		readonly VITE_LOG_LEVEL?: string;
		readonly DEV: boolean;
	}

	interface ImportMeta {
		readonly env: ImportMetaEnv;
	}
}

export {};
