import { type ConsoleSize, type TextItem } from "../../deps/jsr.io/@david/console-static-text/0.3.0/mod.js";
export declare enum LoggerRefreshItemKind {
    ProgressBars = 0,
    Selection = 1
}
declare function setItems(kind: LoggerRefreshItemKind, items: TextItem[] | undefined, size?: ConsoleSize): void;
declare const logger: {
    setItems: typeof setItems;
    logOnce(items: TextItem[], size?: ConsoleSize): void;
    withTempClear(action: () => void): void;
};
export { logger };
//# sourceMappingURL=logger.d.ts.map