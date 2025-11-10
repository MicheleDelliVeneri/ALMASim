import * as dntShim from "../../_dnt.shims.js";
import { type TextItem } from "../../deps/jsr.io/@david/console-static-text/0.3.0/mod.js";
export declare enum Keys {
    Up = 0,
    Down = 1,
    Left = 2,
    Right = 3,
    Enter = 4,
    Space = 5,
    Backspace = 6
}
export declare function readKeys(): AsyncGenerator<string | Keys, void, unknown>;
export declare function innerReadKeys(reader: Pick<typeof dntShim.Deno.stdin, "read">): AsyncGenerator<string | Keys, void, unknown>;
export declare function hideCursor(): void;
export declare function showCursor(): void;
export declare let isOutputTty: any;
export declare function setNotTtyForTesting(): void;
export declare function resultOrExit<T>(result: T | undefined): T;
export interface SelectionOptions<TReturn> {
    message: string;
    render: () => TextItem[];
    noClear: boolean | undefined;
    onKey: (key: string | Keys) => TReturn | undefined;
}
export declare function createSelection<TReturn>(options: SelectionOptions<TReturn>): Promise<TReturn | undefined>;
//# sourceMappingURL=utils.d.ts.map