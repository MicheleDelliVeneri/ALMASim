import type { CommandContext } from "../command_handler.js";
import type { ExecuteResult } from "../result.js";
export declare function cpCommand(context: CommandContext): Promise<ExecuteResult>;
interface PathWithSpecified {
    path: string;
    specified: string;
}
interface CopyFlags {
    recursive: boolean;
    operations: {
        from: PathWithSpecified;
        to: PathWithSpecified;
    }[];
}
export declare function parseCpArgs(cwd: string, args: string[]): Promise<CopyFlags>;
export declare function mvCommand(context: CommandContext): Promise<ExecuteResult>;
interface MoveFlags {
    operations: {
        from: PathWithSpecified;
        to: PathWithSpecified;
    }[];
}
export declare function parseMvArgs(cwd: string, args: string[]): Promise<MoveFlags>;
export {};
//# sourceMappingURL=cp_mv.d.ts.map