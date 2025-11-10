import type { CommandContext } from "../command_handler.js";
export declare function touchCommand(context: CommandContext): Promise<import("../result.js").ExecuteResult>;
interface TouchFlags {
    paths: string[];
}
export declare function parseArgs(args: string[]): TouchFlags;
export {};
//# sourceMappingURL=touch.d.ts.map