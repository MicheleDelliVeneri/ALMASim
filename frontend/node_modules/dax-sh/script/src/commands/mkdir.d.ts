import type { CommandContext } from "../command_handler.js";
import type { ExecuteResult } from "../result.js";
export declare function mkdirCommand(context: CommandContext): Promise<ExecuteResult>;
interface MkdirFlags {
    parents: boolean;
    paths: string[];
}
export declare function parseArgs(args: string[]): MkdirFlags;
export {};
//# sourceMappingURL=mkdir.d.ts.map