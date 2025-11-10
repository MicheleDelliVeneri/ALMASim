import { Deno } from "@deno/shim-deno";
export { Deno } from "@deno/shim-deno";
import { ReadableStream, WritableStream, TextDecoderStream, TransformStream } from "node:stream/web";
export { ReadableStream, WritableStream, TextDecoderStream, TransformStream, type StreamPipeOptions, type ReadableStreamDefaultReader, type WritableStreamDefaultWriter, type StreamPipeOptions as PipeOptions, type QueuingStrategy } from "node:stream/web";
export { type BodyInit, type RequestCache, type RequestMode, type RequestRedirect, type ReferrerPolicy } from "undici-types";
export declare const dntGlobalThis: Omit<typeof globalThis, "ReadableStream" | "TextDecoderStream" | "TransformStream" | "WritableStream" | "Deno"> & {
    Deno: typeof Deno;
    ReadableStream: {
        new (underlyingSource: import("stream/web").UnderlyingByteSource, strategy?: import("stream/web").QueuingStrategy<Uint8Array>): ReadableStream<Uint8Array>;
        new <R = any>(underlyingSource?: import("stream/web").UnderlyingSource<R>, strategy?: import("stream/web").QueuingStrategy<R>): ReadableStream<R>;
        prototype: ReadableStream;
        from<T>(iterable: Iterable<T> | AsyncIterable<T>): ReadableStream<T>;
    };
    WritableStream: {
        new <W = any>(underlyingSink?: import("stream/web").UnderlyingSink<W>, strategy?: import("stream/web").QueuingStrategy<W>): WritableStream<W>;
        prototype: WritableStream;
    };
    TextDecoderStream: {
        new (encoding?: string, options?: import("stream/web").TextDecoderOptions): TextDecoderStream;
        prototype: TextDecoderStream;
    };
    TransformStream: {
        new <I = any, O = any>(transformer?: import("stream/web").Transformer<I, O>, writableStrategy?: import("stream/web").QueuingStrategy<I>, readableStrategy?: import("stream/web").QueuingStrategy<O>): TransformStream<I, O>;
        prototype: TransformStream;
    };
};
//# sourceMappingURL=_dnt.shims.d.ts.map