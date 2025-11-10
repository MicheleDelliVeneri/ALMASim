import * as dntShim from "../../_dnt.shims.js";
export interface SpawnCommandOptions {
    args: string[];
    cwd: string;
    env: Record<string, string>;
    clearEnv: boolean;
    stdin: "inherit" | "null" | "piped";
    stdout: "inherit" | "null" | "piped";
    stderr: "inherit" | "null" | "piped";
}
export interface SpawnedChildProcess {
    stdin(): dntShim.WritableStream;
    stdout(): dntShim.ReadableStream;
    stderr(): dntShim.ReadableStream;
    kill(signo?: dntShim.Deno.Signal): void;
    waitExitCode(): Promise<number>;
}
export type SpawnCommand = (path: string, options: SpawnCommandOptions) => SpawnedChildProcess;
//# sourceMappingURL=process.common.d.ts.map