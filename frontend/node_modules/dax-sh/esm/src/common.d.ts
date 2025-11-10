import * as dntShim from "../_dnt.shims.js";
import type { Reader } from "./pipes.js";
interface Symbols {
    /** Use this symbol to enable the provided object to be written to in
     * an output redirect within a template literal expression.
     *
     * @example
     * ```ts
     * class MyClass {
     *   [$.symbols.writable](): WritableStream<Uint8Array> {
     *     // return a WritableStream here
     *   }
     * }
     * const myObj = new MyClass();
     * await $`echo 1 > ${myObj}`;
     * ```
     */
    readonly writable: symbol;
    /** Use this symbol to enable the provided object to be read from in
     * an input redirect within a template literal expression.
     *
     * @example
     * ```ts
     * class MyClass {
     *   [$.symbols.readable](): ReadableStream<Uint8Array> {
     *     // return a ReadableStream here
     *   }
     * }
     * const myObj = new MyClass();
     * await $`gzip < ${myObj}`;
     * ```
     */
    readonly readable: symbol;
}
export declare const symbols: Symbols;
/** A timeout error. */
export declare class TimeoutError extends Error {
    constructor(message: string);
    get name(): string;
}
/**
 * Delay used for certain actions.
 *
 * @remarks Providing just a number will use milliseconds.
 */
export type Delay = number | `${number}ms` | `${number}s` | `${number}m` | `${number}m${number}s` | `${number}h` | `${number}h${number}m` | `${number}h${number}m${number}s`;
/** An iterator that returns a new delay each time. */
export interface DelayIterator {
    next(): number;
}
export declare function formatMillis(ms: number): string;
export declare function delayToIterator(delay: Delay | DelayIterator): DelayIterator;
export declare function delayToMs(delay: Delay): number;
export declare function filterEmptyRecordValues<TValue>(record: Record<string, TValue | undefined>): Record<string, TValue>;
export declare function resolvePath(cwd: string, arg: string): string;
export declare class Box<T> {
    value: T;
    constructor(value: T);
}
export declare class TreeBox<T> {
    #private;
    constructor(value: T | TreeBox<T>);
    getValue(): T;
    setValue(value: T): void;
    createChild(): TreeBox<T>;
}
/** A special kind of tree box that handles logging with static text. */
export declare class LoggerTreeBox extends TreeBox<(...args: any[]) => void> {
    getValue(): (...args: any[]) => void;
}
/** lstat that doesn't throw when the path is not found. */
export declare function safeLstat(path: string): Promise<dntShim.Deno.FileInfo | undefined>;
export declare function getFileNameFromUrl(url: string | URL): string | undefined;
/**
 * Gets an executable shebang from the provided file path.
 * @returns
 * - An object outlining information about the shebang.
 * - `undefined` if the file exists, but doesn't have a shebang.
 * - `false` if the file does NOT exist.
 */
export declare function getExecutableShebangFromPath(path: string): Promise<false | ShebangInfo | undefined>;
export interface ShebangInfo {
    stringSplit: boolean;
    command: string;
}
export declare function getExecutableShebang(reader: Reader): Promise<ShebangInfo | undefined>;
export declare function abortSignalToPromise(signal: AbortSignal): {
    [Symbol.dispose](): void;
    promise: Promise<void>;
};
export declare function errorToString(err: unknown): string;
export {};
//# sourceMappingURL=common.d.ts.map