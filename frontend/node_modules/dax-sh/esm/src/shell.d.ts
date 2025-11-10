import { RealEnvironment as DenoWhichRealEnvironment } from "../deps/jsr.io/@david/which/0.4.1/mod.js";
import type { KillSignal } from "./command.js";
import type { CommandContext, CommandHandler, CommandPipeReader } from "./command_handler.js";
import { type PipeWriter, type Reader, ShellPipeWriter } from "./pipes.js";
import { type EnvChange, type ExecuteResult } from "./result.js";
export interface SequentialList {
    items: SequentialListItem[];
}
export interface SequentialListItem {
    isAsync: boolean;
    sequence: Sequence;
}
export type Sequence = ShellVar | Pipeline | BooleanList;
export interface ShellVar extends EnvVar {
    kind: "shellVar";
}
export interface EnvVar {
    name: string;
    value: Word;
}
export interface Pipeline {
    kind: "pipeline";
    negated: boolean;
    inner: PipelineInner;
}
export type PipelineInner = Command | PipeSequence;
export interface Command {
    kind: "command";
    inner: CommandInner;
    redirect: Redirect | undefined;
}
export type CommandInner = SimpleCommand | Subshell;
export interface SimpleCommand {
    kind: "simple";
    envVars: EnvVar[];
    args: Word[];
}
export type Word = WordPart[];
export type WordPart = Text | Variable | StringPartCommand | Quoted | Tilde;
export interface Text {
    kind: "text";
    value: string;
}
export interface Variable {
    kind: "variable";
    value: string;
}
export interface StringPartCommand {
    kind: "command";
    value: SequentialList;
}
export interface Quoted {
    kind: "quoted";
    value: WordPart[];
}
export interface Tilde {
    kind: "tilde";
}
export interface Subshell extends SequentialList {
    kind: "subshell";
}
export interface PipeSequence {
    kind: "pipeSequence";
    current: Command;
    op: PipeSequenceOp;
    next: PipelineInner;
}
export type PipeSequenceOp = "stdout" | "stdoutstderr";
export type RedirectFd = RedirectFdFd | RedirectFdStdoutStderr;
export interface RedirectFdFd {
    kind: "fd";
    fd: number;
}
export interface RedirectFdStdoutStderr {
    kind: "stdoutStderr";
}
export type RedirectOp = RedirectOpInput | RedirectOpOutput;
export interface RedirectOpInput {
    kind: "input";
    value: "redirect";
}
export interface RedirectOpOutput {
    kind: "output";
    value: "overwrite" | "append";
}
export interface Redirect {
    maybeFd: RedirectFd | undefined;
    op: RedirectOp;
    ioFile: IoFile;
}
export type IoFile = IoFileWord | IoFileFd;
export interface IoFileWord {
    kind: "word";
    value: Word;
}
export interface IoFileFd {
    kind: "fd";
    value: number;
}
export type BooleanListOperator = "and" | "or";
export interface BooleanList {
    kind: "booleanList";
    current: Sequence;
    op: BooleanListOperator;
    next: Sequence;
}
interface Env {
    setCwd(cwd: string): void;
    getCwd(): string;
    setEnvVar(key: string, value: string | undefined): void;
    getEnvVar(key: string): string | undefined;
    getEnvVars(): Record<string, string>;
    clone(): Env;
}
export declare class StreamFds {
    #private;
    insertReader(fd: number, stream: () => Reader): void;
    insertWriter(fd: number, stream: () => PipeWriter): void;
    getReader(fd: number): Reader | undefined;
    getWriter(fd: number): PipeWriter | undefined;
}
interface ContextOptions {
    stdin: CommandPipeReader;
    stdout: ShellPipeWriter;
    stderr: ShellPipeWriter;
    env: Env;
    shellVars: Record<string, string>;
    static: StaticContextState;
}
/** State that never changes across the entire execution of the shell. */
interface StaticContextState {
    signal: KillSignal;
    commands: Record<string, CommandHandler>;
    fds: StreamFds | undefined;
}
export declare class Context {
    #private;
    stdin: CommandPipeReader;
    stdout: ShellPipeWriter;
    stderr: ShellPipeWriter;
    constructor(opts: ContextOptions);
    get signal(): KillSignal;
    applyChanges(changes: EnvChange[] | undefined): void;
    setEnvVar(key: string, value: string | undefined): void;
    setShellVar(key: string, value: string | undefined): void;
    getEnvVars(): Record<string, string>;
    getCwd(): string;
    getVar(key: string): string | undefined;
    getCommand(command: string): CommandHandler;
    getFdReader(fd: number): Reader | undefined;
    getFdWriter(fd: number): PipeWriter | undefined;
    asCommandContext(args: string[]): CommandContext;
    error(text: string): Promise<ExecuteResult> | ExecuteResult;
    error(code: number, text: string): Promise<ExecuteResult> | ExecuteResult;
    error(codeOrText: number | string, maybeText: string | undefined): Promise<ExecuteResult> | ExecuteResult;
    withInner(opts: Partial<Pick<ContextOptions, "stdout" | "stderr" | "stdin">>): Context;
    clone(): Context;
}
export declare function parseCommand(command: string): SequentialList;
export interface SpawnOpts {
    stdin: CommandPipeReader;
    stdout: ShellPipeWriter;
    stderr: ShellPipeWriter;
    env: Record<string, string>;
    commands: Record<string, CommandHandler>;
    cwd: string;
    exportEnv: boolean;
    clearedEnv: boolean;
    signal: KillSignal;
    fds: StreamFds | undefined;
}
export declare function spawn(list: SequentialList, opts: SpawnOpts): Promise<number>;
declare class WhichEnv extends DenoWhichRealEnvironment {
    requestPermission(folderPath: string): void;
}
export declare const denoWhichRealEnv: WhichEnv;
export declare function whichFromContext(commandName: string, context: {
    getVar(key: string): string | undefined;
}): Promise<string | undefined>;
export {};
//# sourceMappingURL=shell.d.ts.map