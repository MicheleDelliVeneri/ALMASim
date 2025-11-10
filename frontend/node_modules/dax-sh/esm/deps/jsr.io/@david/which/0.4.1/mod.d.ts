import * as dntShim from "../../../../../_dnt.shims.js";
export interface Environment {
    /** Gets an environment variable. */
    env(key: string): string | undefined;
    /** Resolves the `Deno.FileInfo` for the specified
     * path following symlinks.
     */
    stat(filePath: string): Promise<Pick<dntShim.Deno.FileInfo, "isFile">>;
    /** Synchronously resolves the `Deno.FileInfo` for
     * the specified path following symlinks.
     */
    statSync(filePath: string): Pick<dntShim.Deno.FileInfo, "isFile">;
    /** Gets the current operating system. */
    os: typeof dntShim.Deno.build.os;
    /** Optional method for requesting broader permissions for a folder
     * instead of asking for each file when the operating system requires
     * probing multiple files for an executable path.
     *
     * This is not the default, but is useful on Windows for example.
     */
    requestPermission?(folderPath: string): void;
}
/** Default implementation that interacts with the file system and process env vars. */
export declare class RealEnvironment implements Environment {
    env(key: string): string | undefined;
    stat(path: string): Promise<Pick<dntShim.Deno.FileInfo, "isFile">>;
    statSync(path: string): Pick<dntShim.Deno.FileInfo, "isFile">;
    get os(): typeof dntShim.Deno.build.os;
}
/** Finds the path to the specified command asynchronously. */
export declare function which(command: string, environment?: Omit<Environment, "statSync">): Promise<string | undefined>;
/** Finds the path to the specified command synchronously. */
export declare function whichSync(command: string, environment?: Omit<Environment, "stat">): string | undefined;
//# sourceMappingURL=mod.d.ts.map