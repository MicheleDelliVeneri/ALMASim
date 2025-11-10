export interface ArgKind {
    kind: "ShortFlag" | "LongFlag" | "Arg";
    arg: string;
}
export declare function parseArgKinds(flags: string[]): ArgKind[];
export declare function bailUnsupported(arg: ArgKind): never;
//# sourceMappingURL=args.d.ts.map