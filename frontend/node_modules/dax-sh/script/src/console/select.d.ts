import { type SelectionOptions } from "./utils.js";
/** Options for showing a selection that only has one result. */
export interface SelectOptions {
    /** Prompt text to show the user. */
    message: string;
    /** Initial selected option index. Defaults to 0. */
    initialIndex?: number;
    /** Options to show the user. */
    options: string[];
    /**
     * Whether to not clear the selection text on selection.
     * @default `false`
     */
    noClear?: boolean;
}
export declare function select(opts: SelectOptions): Promise<number>;
export declare function maybeSelect(opts: SelectOptions): Promise<number | undefined>;
export declare function innerSelect(opts: SelectOptions): Pick<SelectionOptions<number | undefined>, "render" | "onKey">;
//# sourceMappingURL=select.d.ts.map