import { type SelectionOptions } from "./utils.js";
/** Options for showing confirming a yes or no question. */
export interface ConfirmOptions {
    /** Message to display to the user. */
    message: string;
    /**
     * Default value.
     * @default `undefined`
     */
    default?: boolean | undefined;
    /**
     * Whether to not clear the prompt text on selection.
     * @default `false`
     */
    noClear?: boolean;
}
export declare function confirm(optsOrMessage: ConfirmOptions | string, options?: Omit<ConfirmOptions, "message">): Promise<boolean>;
export declare function maybeConfirm(optsOrMessage: ConfirmOptions | string, options?: Omit<ConfirmOptions, "message">): Promise<boolean | undefined>;
export declare function innerConfirm(opts: ConfirmOptions): Pick<SelectionOptions<boolean | undefined>, "render" | "onKey">;
//# sourceMappingURL=confirm.d.ts.map