/** Text item to display. */
export type TextItem = string | DeferredItem | DetailedTextItem;
/** Function called on each render. */
export type DeferredItem = (size: ConsoleSize | undefined) => TextItem | TextItem[];
/** Item that also supports hanging indentation. */
export interface DetailedTextItem {
    text: string | DeferredItem;
    hangingIndent?: number;
}
/** Console size. */
export interface ConsoleSize {
    /** Number of horizontal columns. */
    columns: number;
    /** Number of vertical rows. */
    rows: number;
}
declare const scopesSymbol: unique symbol;
declare const getItemsSymbol: unique symbol;
declare const renderOnceSymbol: unique symbol;
declare const onItemsChangedEventsSymbol: unique symbol;
export declare class StaticTextScope implements Disposable {
    #private;
    constructor(container: StaticTextContainer);
    [Symbol.dispose](): void;
    private [getItemsSymbol];
    /** Sets the text to render for this scope. */
    setText(text: string): void;
    /** Text with a render function. */
    setText(deferredText: DeferredItem): void;
    /** Sets the items for this scope. */
    setText(items: TextItem[]): void;
    /** Logs the provided text above the static text. */
    logAbove(text: string, size?: ConsoleSize): void;
    logAbove(items: TextItem[], size?: ConsoleSize): void;
    /** Forces a refresh of the container. */
    refresh(size?: ConsoleSize): void;
}
export declare class StaticTextContainer {
    #private;
    private readonly [scopesSymbol];
    private readonly [onItemsChangedEventsSymbol];
    constructor(onWriteText: (text: string) => void, getConsoleSize: () => ConsoleSize | undefined);
    /** Creates a scope which can be used to set the text for. */
    createScope(): StaticTextScope;
    /** Gets the containers current console size. */
    getConsoleSize(): ConsoleSize | undefined;
    /** Logs the provided text above the static text. */
    logAbove(text: string, size?: ConsoleSize): void;
    logAbove(items: TextItem[], size?: ConsoleSize): void;
    logAbove(textOrItems: TextItem[] | string, size?: ConsoleSize): void;
    /** Clears the displayed text for the provided action. */
    withTempClear(action: () => void, size?: ConsoleSize): void;
    /** Clears the text and flushes it to the console. */
    clear(size?: ConsoleSize): void;
    /** Refreshes the static text (writes it to the console). */
    refresh(size?: ConsoleSize): void;
    /**
     * Renders the clear text.
     *
     * Note: this is a low level method. Prefer calling `.clear()` instead.
     */
    renderClearText(size?: ConsoleSize): string | undefined;
    /**
     * Renders the next text that should be displayed.
     *
     * Note: This is a low level method. Prefer calling `.refresh()` instead.
     */
    renderRefreshText(size?: ConsoleSize): string | undefined;
    private [renderOnceSymbol];
}
export interface RenderIntervalScope extends Disposable {
}
/** Renders a container at an interval. */
export declare class RenderInterval implements Disposable {
    #private;
    /**
     * Constructs a new `RenderInterval` from the provided `StaticTextContainer`.
     * @param container Container to render every `intervalMs`.
     */
    constructor(container: StaticTextContainer);
    [Symbol.dispose](): void;
    /** Gets how often this interval will refresh the output.
     * @default `60`
     */
    get intervalMs(): number;
    /** Sets how often this should refresh the output. */
    set intervalMs(value: number);
    /**
     * Starts the render task returning a disposable for stopping it.
     *
     * Note that it's perfectly fine to just start this and never dispose it.
     * The underlying interval won't run if there's no items in the container.
     */
    start(): RenderIntervalScope;
}
/**
 * Global `StaticTextContainer` that can be shared amongst many libraries.
 * This writes the static text to stderr and gets the real console size.
 */
export declare const staticText: StaticTextContainer;
export declare const renderInterval: RenderInterval;
/** Renders the text items to a string using no knowledge of a `StaticTextContainer`. */
export declare function renderTextItems(items: TextItem[], size?: ConsoleSize): string;
/** Helper to get the console size and return undefined if it's not available. */
export declare function maybeConsoleSize(): ConsoleSize | undefined;
/** Convenience function for stripping ANSI codes.
 * Exposed because it's used in the rust crate. */
export declare function stripAnsiCodes(text: string): string;
export {};
//# sourceMappingURL=mod.d.ts.map