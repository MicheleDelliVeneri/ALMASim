import { type SelectionOptions } from "./utils.js";
/** Options for showing an input where the user enters a value. */
export interface PromptOptions {
    /** Message to display to the user. */
    message: string;
    /**
     * Default value.
     */
    default?: string;
    /**
     * Whether typed characters should be hidden by
     * a mask, optionally allowing a choice of mask
     * character (`*` by default) and whether or not
     * to keep the final character visible as the user
     * types (`false` by default).
     * @default `false`
     */
    mask?: PromptInputMask | boolean;
    /**
     * Whether to not clear the prompt text on selection.
     * @default `false`
     */
    noClear?: boolean;
}
/** Configuration of the prompt input mask */
export interface PromptInputMask {
    /** The character used to mask input (`*` by default) */
    char?: string;
    /** Whether or not to keep the last character "unmasked" (`false` by default) */
    lastVisible?: boolean;
}
export declare function prompt(optsOrMessage: PromptOptions | string, options?: Omit<PromptOptions, "message">): Promise<string>;
export declare function maybePrompt(optsOrMessage: PromptOptions | string, options?: Omit<PromptOptions, "message">): Promise<string | undefined>;
export declare function innerPrompt(opts: PromptOptions): Pick<SelectionOptions<string | undefined>, "render" | "onKey">;
//# sourceMappingURL=prompt.d.ts.map