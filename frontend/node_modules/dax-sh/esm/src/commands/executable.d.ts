import type { CommandHandler } from "../command_handler.js";
/**
 * Creates a new command that runs the executable at the specified path.
 * @param resolvedPath A fully resolved path.
 * @returns Command handler that can be registered in a `CommandBuilder`.
 */
export declare function createExecutableCommand(resolvedPath: string): CommandHandler;
//# sourceMappingURL=executable.d.ts.map