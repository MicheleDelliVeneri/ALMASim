import type { CommandContext } from "../command_handler.js";
import type { ExecuteResult } from "../result.js";
export declare function whichCommand(context: CommandContext): Promise<ExecuteResult>;
export declare function parseArgs(args: string[]): {
    commandName: string | undefined;
};
//# sourceMappingURL=which.d.ts.map