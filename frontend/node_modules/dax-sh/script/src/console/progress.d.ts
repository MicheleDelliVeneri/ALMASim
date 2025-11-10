import { type ConsoleSize, type TextItem } from "../../deps/jsr.io/@david/console-static-text/0.3.0/mod.js";
/** Options for showing progress. */
export interface ProgressOptions {
    /** Prefix message/word that will be displayed in green. */
    prefix?: string;
    /** Message to show after the prefix in white. */
    message?: string;
    /**
     * Optional length if known.
     *
     * If this is undefined then the progress will be indeterminate.
     */
    length?: number;
    /** Do not clear the progress bar when finishing it. */
    noClear?: boolean;
}
/** A progress bar instance created via `$.progress(...)`. */
export declare class ProgressBar {
    #private;
    /** @internal */
    constructor(onLog: (...data: any[]) => void, opts: ProgressOptions);
    /** Sets the prefix message/word, which will be displayed in green. */
    prefix(prefix: string | undefined): this;
    /** Sets the message the progress bar will display after the prefix in white. */
    message(message: string | undefined): this;
    /** Sets how to format the length values. */
    kind(kind: "raw" | "bytes"): this;
    /** Sets the current position of the progress bar. */
    position(position: number): this;
    /** Increments the position of the progress bar. */
    increment(inc?: number): this;
    /** Sets the total length of the progress bar. */
    length(size: number | undefined): this;
    /** Whether the progress bar should output a summary when finished. */
    noClear(value?: boolean): this;
    /** Forces a render to the console. */
    forceRender(): void;
    /** Finish showing the progress bar. */
    finish(): void;
    /** Does the provided action and will call `.finish()` when this is the last `.with(...)` action that runs. */
    with<TResult>(action: () => TResult): TResult;
    with<TResult>(action: () => Promise<TResult>): Promise<TResult>;
}
interface RenderState {
    message: string | undefined;
    prefix: string | undefined;
    length: number | undefined;
    currentPos: number;
    tickCount: number;
    hasCompleted: boolean;
    kind: "raw" | "bytes";
}
export declare function renderProgressBar(state: RenderState, size: ConsoleSize | undefined): TextItem[];
export declare function isShowingProgressBars(): any;
export declare function humanDownloadSize(byteCount: number, totalBytes?: number): string;
export {};
//# sourceMappingURL=progress.d.ts.map