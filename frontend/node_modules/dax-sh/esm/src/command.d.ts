import * as dntShim from "../_dnt.shims.js";
import { Path } from "../deps/jsr.io/@david/path/0.2.0/mod.js";
import { Buffer } from "../deps/jsr.io/@std/io/0.225.2/buffer.js";
import type { CommandHandler } from "./command_handler.js";
import { Box, LoggerTreeBox } from "./common.js";
import type { Delay } from "./common.js";
import { PipedBuffer, type Reader, type ShellPipeReaderKind, type ShellPipeWriterKind } from "./pipes.js";
import { StreamFds } from "./shell.js";
type BufferStdio = "inherit" | "null" | "streamed" | Buffer;
type StreamKind = "stdout" | "stderr" | "combined";
declare class Deferred<T> {
    #private;
    constructor(create: () => T | Promise<T>);
    create(): T | Promise<T>;
}
interface ShellPipeWriterKindWithOptions {
    kind: ShellPipeWriterKind;
    options?: dntShim.StreamPipeOptions;
}
interface CommandBuilderStateCommand {
    text: string;
    fds: StreamFds | undefined;
}
interface CommandBuilderState {
    command: Readonly<CommandBuilderStateCommand> | undefined;
    stdin: "inherit" | "null" | Box<Reader | dntShim.ReadableStream<Uint8Array> | "consumed"> | Deferred<dntShim.ReadableStream<Uint8Array> | Reader>;
    combinedStdoutStderr: boolean;
    stdout: ShellPipeWriterKindWithOptions;
    stderr: ShellPipeWriterKindWithOptions;
    noThrow: boolean | number[];
    env: Record<string, string | undefined>;
    commands: Record<string, CommandHandler>;
    cwd: string | undefined;
    clearEnv: boolean;
    exportEnv: boolean;
    printCommand: boolean;
    printCommandLogger: LoggerTreeBox;
    timeout: number | undefined;
    signal: KillSignal | undefined;
}
/** @internal */
export declare const getRegisteredCommandNamesSymbol: unique symbol;
/** @internal */
export declare const setCommandTextStateSymbol: unique symbol;
/**
 * Underlying builder API for executing commands.
 *
 * This is what `$` uses to execute commands. Using this provides
 * a way to provide a raw text command or an array of arguments.
 *
 * Command builders are immutable where each method call creates
 * a new command builder.
 *
 * ```ts
 * const builder = new CommandBuilder()
 *  .cwd("./src")
 *  .command("echo $MY_VAR");
 *
 * // outputs 5
 * console.log(await builder.env("MY_VAR", "5").text());
 * // outputs 6
 * console.log(await builder.env("MY_VAR", "6").text());
 * ```
 */
export declare class CommandBuilder implements PromiseLike<CommandResult> {
    #private;
    then<TResult1 = CommandResult, TResult2 = never>(onfulfilled?: ((value: CommandResult) => TResult1 | PromiseLike<TResult1>) | null | undefined, onrejected?: ((reason: any) => TResult2 | PromiseLike<TResult2>) | null | undefined): PromiseLike<TResult1 | TResult2>;
    /**
     * Explicit way to spawn a command.
     *
     * This is an alias for awaiting the command builder or calling `.then(...)`
     */
    spawn(): CommandChild;
    /**
     * Register a command.
     */
    registerCommand(command: string, handleFn: CommandHandler): CommandBuilder;
    /**
     * Register multilple commands.
     */
    registerCommands(commands: Record<string, CommandHandler>): CommandBuilder;
    /**
     * Unregister a command.
     */
    unregisterCommand(command: string): CommandBuilder;
    /** Sets the raw command to execute. */
    command(command: string | string[]): CommandBuilder;
    /** The command should not throw for the provided non-zero exit codes. */
    noThrow(exclusionExitCode: number, ...additional: number[]): CommandBuilder;
    /** The command should not throw when it fails or times out. */
    noThrow(value?: boolean): CommandBuilder;
    /** Sets the command signal that will be passed to all commands
     * created with this command builder.
     */
    signal(killSignal: KillSignal): CommandBuilder;
    /**
     * Whether to capture a combined buffer of both stdout and stderr.
     *
     * This will set both stdout and stderr to "piped" if not already "piped"
     * or "inheritPiped".
     */
    captureCombined(value?: boolean): CommandBuilder;
    /**
     * Sets the stdin to use for the command.
     *
     * @remarks If multiple launches of a command occurs, then stdin will only be
     * read from the first consumed reader or readable stream and error otherwise.
     * For this reason, if you are setting stdin to something other than "inherit" or
     * "null", then it's recommended to set this each time you spawn a command.
     */
    stdin(reader: ShellPipeReaderKind): CommandBuilder;
    /**
     * Sets the stdin string to use for a command.
     *
     * @remarks See the remarks on stdin. The same applies here.
     */
    stdinText(text: string): CommandBuilder;
    /** Set the stdout kind. */
    stdout(kind: ShellPipeWriterKind): CommandBuilder;
    stdout(kind: dntShim.WritableStream<Uint8Array>, options?: dntShim.StreamPipeOptions): CommandBuilder;
    /** Set the stderr kind. */
    stderr(kind: ShellPipeWriterKind): CommandBuilder;
    stderr(kind: dntShim.WritableStream<Uint8Array>, options?: dntShim.StreamPipeOptions): CommandBuilder;
    /** Pipes the current command to the provided command returning the
     * provided command builder. When chaining, it's important to call this
     * after you are done configuring the current command or else you will
     * start modifying the provided command instead.
     *
     * @example
     * ```ts
     * const lineCount = await $`echo 1 && echo 2`
     *  .pipe($`wc -l`)
     *  .text();
     * ```
     */
    pipe(builder: CommandBuilder): CommandBuilder;
    /** Sets multiple environment variables to use at the same time via an object literal. */
    env(items: Record<string, string | undefined>): CommandBuilder;
    /** Sets a single environment variable to use. */
    env(name: string, value: string | undefined): CommandBuilder;
    /** Sets the current working directory to use when executing this command. */
    cwd(dirPath: string | URL | Path): CommandBuilder;
    /**
     * Exports the environment of the command to the executing process.
     *
     * So for example, changing the directory in a command or exporting
     * an environment variable will actually change the environment
     * of the executing process.
     *
     * ```ts
     * await $`cd src && export SOME_VALUE=5`;
     * console.log(Deno.env.get("SOME_VALUE")); // 5
     * console.log(Deno.cwd()); // will be in the src directory
     * ```
     */
    exportEnv(value?: boolean): CommandBuilder;
    /**
     * Clear environmental variables from parent process.
     *
     * Doesn't guarantee that only `env` variables are present, as the OS may
     * set environmental variables for processes.
     */
    clearEnv(value?: boolean): CommandBuilder;
    /**
     * Prints the command text before executing the command.
     *
     * For example:
     *
     * ```ts
     * const text = "example";
     * await $`echo ${text}`.printCommand();
     * ```
     *
     * Outputs:
     *
     * ```
     * > echo example
     * example
     * ```
     */
    printCommand(value?: boolean): CommandBuilder;
    /**
     * Mutates the command builder to change the logger used
     * for `printCommand()`.
     */
    setPrintCommandLogger(logger: (...args: any[]) => void): void;
    /**
     * Ensures stdout and stderr are piped if they have the default behaviour or are inherited.
     *
     * ```ts
     * // ensure both stdout and stderr is not logged to the console
     * await $`echo 1`.quiet();
     * // ensure stdout is not logged to the console
     * await $`echo 1`.quiet("stdout");
     * // ensure stderr is not logged to the console
     * await $`echo 1`.quiet("stderr");
     * ```
     */
    quiet(kind?: StreamKind | "both"): CommandBuilder;
    /**
     * Specifies a timeout for the command. The command will exit with
     * exit code `124` (timeout) if it times out.
     *
     * Note that when using `.noThrow()` this won't cause an error to
     * be thrown when timing out.
     */
    timeout(delay: Delay | undefined): CommandBuilder;
    /**
     * Sets stdout as quiet, spawns the command, and gets stdout as a Uint8Array.
     *
     * Shorthand for:
     *
     * ```ts
     * const data = (await $`command`.quiet("stdout")).stdoutBytes;
     * ```
     */
    bytes(kind?: StreamKind): Promise<Uint8Array>;
    /**
     * Sets the provided stream (stdout by default) as quiet, spawns the command, and gets the stream as a string without the last newline.
     * Can be used to get stdout, stderr, or both.
     *
     * Shorthand for:
     *
     * ```ts
     * const data = (await $`command`.quiet("stdout")).stdout.replace(/\r?\n$/, "");
     * ```
     */
    text(kind?: StreamKind): Promise<string>;
    /** Gets the text as an array of lines. */
    lines(kind?: StreamKind): Promise<string[]>;
    /**
     * Sets stream (stdout by default) as quiet, spawns the command, and gets stream as JSON.
     *
     * Shorthand for:
     *
     * ```ts
     * const data = (await $`command`.quiet("stdout")).stdoutJson;
     * ```
     */
    json<TResult = any>(kind?: Exclude<StreamKind, "combined">): Promise<TResult>;
    /** @internal */
    [getRegisteredCommandNamesSymbol](): string[];
    /** @internal */
    [setCommandTextStateSymbol](textState: CommandBuilderStateCommand): CommandBuilder;
}
export declare class CommandChild extends Promise<CommandResult> {
    #private;
    /** @internal */
    constructor(executor: (resolve: (value: CommandResult) => void, reject: (reason?: any) => void) => void, options?: {
        pipedStdoutBuffer: PipedBuffer | undefined;
        pipedStderrBuffer: PipedBuffer | undefined;
        killSignalController: KillSignalController | undefined;
    });
    /** Send a signal to the executing command's child process. Note that SIGTERM,
     * SIGKILL, SIGABRT, SIGQUIT, SIGINT, or SIGSTOP will cause the entire command
     * to be considered "aborted" and if part of a command runs after this has occurred
     * it will return a 124 exit code. Other signals will just be forwarded to the command.
     *
     * Defaults to "SIGTERM".
     */
    kill(signal?: dntShim.Deno.Signal): void;
    stdout(): dntShim.ReadableStream<Uint8Array>;
    stderr(): dntShim.ReadableStream<Uint8Array>;
}
export declare function parseAndSpawnCommand(state: CommandBuilderState): CommandChild;
/** Result of running a command. */
export declare class CommandResult {
    #private;
    /** The exit code. */
    readonly code: number;
    /** @internal */
    constructor(code: number, stdout: BufferStdio, stderr: BufferStdio, combined: Buffer | undefined);
    /** Raw decoded stdout text. */
    get stdout(): string;
    /**
     * Stdout text as JSON.
     *
     * @remarks Will throw if it can't be parsed as JSON.
     */
    get stdoutJson(): any;
    /** Raw stdout bytes. */
    get stdoutBytes(): Uint8Array;
    /** Raw decoded stdout text. */
    get stderr(): string;
    /**
     * Stderr text as JSON.
     *
     * @remarks Will throw if it can't be parsed as JSON.
     */
    get stderrJson(): any;
    /** Raw stderr bytes. */
    get stderrBytes(): Uint8Array;
    /** Raw combined stdout and stderr text. */
    get combined(): string;
    /** Raw combined stdout and stderr bytes. */
    get combinedBytes(): Uint8Array;
}
export declare function escapeArg(arg: string): string;
export declare class RawArg<T> {
    #private;
    constructor(value: T);
    get value(): T;
}
export declare function rawArg<T>(arg: T): RawArg<T>;
interface KillSignalState {
    abortedCode: number | undefined;
    listeners: ((signal: dntShim.Deno.Signal) => void)[];
}
/** Similar to an AbortController, but for sending signals to commands. */
export declare class KillSignalController {
    #private;
    constructor();
    get signal(): KillSignal;
    /** Send a signal to the downstream child process. Note that SIGTERM,
     * SIGKILL, SIGABRT, SIGQUIT, SIGINT, or SIGSTOP will cause all the commands
     * to be considered "aborted" and will return a 124 exit code, while other
     * signals will just be forwarded to the commands.
     */
    kill(signal?: dntShim.Deno.Signal): void;
}
/** Listener for when a KillSignal is killed. */
export type KillSignalListener = (signal: dntShim.Deno.Signal) => void;
/** Similar to `AbortSignal`, but for `Deno.Signal`.
 *
 * A `KillSignal` is considered aborted if its controller
 * receives SIGTERM, SIGKILL, SIGABRT, SIGQUIT, SIGINT, or SIGSTOP.
 *
 * These can be created via a `KillSignalController`.
 */
export declare class KillSignal {
    #private;
    /** @internal */
    constructor(symbol: Symbol, state: KillSignalState);
    /** Returns if the command signal has ever received a SIGTERM,
     * SIGKILL, SIGABRT, SIGQUIT, SIGINT, or SIGSTOP
     */
    get aborted(): boolean;
    /** Gets the exit code to use if aborted. */
    get abortedExitCode(): number | undefined;
    /**
     * Causes the provided kill signal to be triggered when this
     * signal receives a signal.
     */
    linkChild(killSignal: KillSignal): {
        unsubscribe(): void;
    };
    addListener(listener: KillSignalListener): void;
    removeListener(listener: KillSignalListener): void;
}
export declare function getSignalAbortCode(signal: dntShim.Deno.Signal): number | undefined;
export declare function template(strings: TemplateStringsArray, exprs: TemplateExpr[]): CommandBuilderStateCommand;
export declare function templateRaw(strings: TemplateStringsArray, exprs: TemplateExpr[]): CommandBuilderStateCommand;
type NonRedirectTemplateExpr = string | number | boolean | Path | Uint8Array | CommandResult | RawArg<NonRedirectTemplateExpr> | {
    toString(): string;
    catch?: never;
};
export type TemplateExpr = NonRedirectTemplateExpr | NonRedirectTemplateExpr[] | dntShim.ReadableStream<Uint8Array> | {
    [readable: symbol]: () => dntShim.ReadableStream<Uint8Array>;
} | (() => dntShim.ReadableStream<Uint8Array>) | {
    [writable: symbol]: () => dntShim.WritableStream<Uint8Array>;
} | dntShim.WritableStream<Uint8Array> | (() => dntShim.WritableStream<Uint8Array>);
export {};
//# sourceMappingURL=command.d.ts.map