import { type SelectionOptions } from "./utils.js";
/** Single options within a multi-select option. */
export interface MultiSelectOption {
    /** Text to display for this option. */
    text: string;
    /** Whether it is selected by default. */
    selected?: boolean;
}
/** Options for showing a selection that has multiple possible values. */
export interface MultiSelectOptions {
    /** Prompt text to show the user. */
    message: string;
    /** Options to show the user. */
    options: (string | MultiSelectOption)[];
    /**
     * Whether to not clear the prompt text on selection.
     * @default `false`
     */
    noClear?: boolean;
}
export declare function multiSelect(opts: MultiSelectOptions): Promise<number[]>;
export declare function maybeMultiSelect(opts: MultiSelectOptions): Promise<number[] | undefined>;
export declare function innerMultiSelect(opts: MultiSelectOptions): Pick<SelectionOptions<number[] | undefined>, "render" | "onKey">;
//# sourceMappingURL=multiSelect.d.ts.map