import * as dntShim from "../../../../../_dnt.shims.js";
import type { Reader } from "./types.js";
/**
 * Create a {@linkcode Reader} from a {@linkcode ReadableStreamDefaultReader}.
 *
 * @example Usage
 * ```ts ignore
 * import { copy } from "@std/io/copy";
 * import { readerFromStreamReader } from "@std/io/reader-from-stream-reader";
 *
 * const res = await fetch("https://deno.land");
 *
 * const reader = readerFromStreamReader(res.body!.getReader());
 * await copy(reader, Deno.stdout);
 * ```
 *
 * @param streamReader The stream reader to read from
 * @returns The reader
 */
export declare function readerFromStreamReader(streamReader: dntShim.ReadableStreamDefaultReader<Uint8Array>): Reader;
//# sourceMappingURL=reader_from_stream_reader.d.ts.map