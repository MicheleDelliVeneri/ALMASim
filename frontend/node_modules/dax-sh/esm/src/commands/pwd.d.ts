import type { CommandContext } from "../command_handler.js";
import type { ExecuteResult } from "../result.js";
export declare function pwdCommand(context: CommandContext): ExecuteResult | Promise<ExecuteResult>;
interface PwdFlags {
    logical: boolean;
}
export declare function parseArgs(args: string[]): PwdFlags;
export {};
//# sourceMappingURL=pwd.d.ts.map