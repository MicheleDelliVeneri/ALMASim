import type { Reader, ReaderSync, Writer, WriterSync } from "./types.js";
/**
 * A variable-sized buffer of bytes with `read()` and `write()` methods.
 *
 * Buffer is almost always used with some I/O like files and sockets. It allows
 * one to buffer up a download from a socket. Buffer grows and shrinks as
 * necessary.
 *
 * Buffer is NOT the same thing as Node's Buffer. Node's Buffer was created in
 * 2009 before JavaScript had the concept of ArrayBuffers. It's simply a
 * non-standard ArrayBuffer.
 *
 * ArrayBuffer is a fixed memory allocation. Buffer is implemented on top of
 * ArrayBuffer.
 *
 * Based on {@link https://golang.org/pkg/bytes/#Buffer | Go Buffer}.
 *
 * @example Usage
 * ```ts
 * import { Buffer } from "@std/io/buffer";
 * import { assertEquals } from "@std/assert/equals";
 *
 * const buf = new Buffer();
 * await buf.write(new TextEncoder().encode("Hello, "));
 * await buf.write(new TextEncoder().encode("world!"));
 *
 * const data = new Uint8Array(13);
 * await buf.read(data);
 *
 * assertEquals(new TextDecoder().decode(data), "Hello, world!");
 * ```
 */
export declare class Buffer implements Writer, WriterSync, Reader, ReaderSync {
    #private;
    /**
     * Constructs a new instance with the specified {@linkcode ArrayBuffer} as its
     * initial contents.
     *
     * @param ab The ArrayBuffer to use as the initial contents of the buffer.
     */
    constructor(ab?: ArrayBufferLike | ArrayLike<number>);
    /**
     * Returns a slice holding the unread portion of the buffer.
     *
     * The slice is valid for use only until the next buffer modification (that
     * is, only until the next call to a method like `read()`, `write()`,
     * `reset()`, or `truncate()`). If `options.copy` is false the slice aliases the buffer content at
     * least until the next buffer modification, so immediate changes to the
     * slice will affect the result of future reads.
     *
     * @example Usage
     * ```ts
     * import { Buffer } from "@std/io/buffer";
     * import { assertEquals } from "@std/assert/equals";
     *
     * const buf = new Buffer();
     * await buf.write(new TextEncoder().encode("Hello, world!"));
     *
     * const slice = buf.bytes();
     * assertEquals(new TextDecoder().decode(slice), "Hello, world!");
     * ```
     *
     * @param options The options for the slice.
     * @returns A slice holding the unread portion of the buffer.
     */
    bytes(options?: {
        copy: boolean;
    }): Uint8Array;
    /**
     * Returns whether the unread portion of the buffer is empty.
     *
     * @example Usage
     * ```ts
     * import { Buffer } from "@std/io/buffer";
     * import { assertEquals } from "@std/assert/equals";
     *
     * const buf = new Buffer();
     * assertEquals(buf.empty(), true);
     * await buf.write(new TextEncoder().encode("Hello, world!"));
     * assertEquals(buf.empty(), false);
     * ```
     *
     * @returns `true` if the unread portion of the buffer is empty, `false`
     *          otherwise.
     */
    empty(): boolean;
    /**
     * A read only number of bytes of the unread portion of the buffer.
     *
     * @example Usage
     * ```ts
     * import { Buffer } from "@std/io/buffer";
     * import { assertEquals } from "@std/assert/equals";
     *
     * const buf = new Buffer();
     * await buf.write(new TextEncoder().encode("Hello, world!"));
     *
     * assertEquals(buf.length, 13);
     * ```
     *
     * @returns The number of bytes of the unread portion of the buffer.
     */
    get length(): number;
    /**
     * The read only capacity of the buffer's underlying byte slice, that is,
     * the total space allocated for the buffer's data.
     *
     * @example Usage
     * ```ts
     * import { Buffer } from "@std/io/buffer";
     * import { assertEquals } from "@std/assert/equals";
     *
     * const buf = new Buffer();
     * assertEquals(buf.capacity, 0);
     * await buf.write(new TextEncoder().encode("Hello, world!"));
     * assertEquals(buf.capacity, 13);
     * ```
     *
     * @returns The capacity of the buffer.
     */
    get capacity(): number;
    /**
     * Discards all but the first `n` unread bytes from the buffer but
     * continues to use the same allocated storage. It throws if `n` is
     * negative or greater than the length of the buffer.
     *
     * @example Usage
     * ```ts
     * import { Buffer } from "@std/io/buffer";
     * import { assertEquals } from "@std/assert/equals";
     *
     * const buf = new Buffer();
     * await buf.write(new TextEncoder().encode("Hello, world!"));
     * buf.truncate(6);
     * assertEquals(buf.length, 6);
     * ```
     *
     * @param n The number of bytes to keep.
     */
    truncate(n: number): void;
    /**
     * Resets the contents
     *
     * @example Usage
     * ```ts
     * import { Buffer } from "@std/io/buffer";
     * import { assertEquals } from "@std/assert/equals";
     *
     * const buf = new Buffer();
     * await buf.write(new TextEncoder().encode("Hello, world!"));
     * buf.reset();
     * assertEquals(buf.length, 0);
     * ```
     */
    reset(): void;
    /**
     * Reads the next `p.length` bytes from the buffer or until the buffer is
     * drained. Returns the number of bytes read. If the buffer has no data to
     * return, the return is EOF (`null`).
     *
     * @example Usage
     * ```ts
     * import { Buffer } from "@std/io/buffer";
     * import { assertEquals } from "@std/assert/equals";
     *
     * const buf = new Buffer();
     * await buf.write(new TextEncoder().encode("Hello, world!"));
     *
     * const data = new Uint8Array(5);
     * const res = await buf.read(data);
     *
     * assertEquals(res, 5);
     * assertEquals(new TextDecoder().decode(data), "Hello");
     * ```
     *
     * @param p The buffer to read data into.
     * @returns The number of bytes read.
     */
    readSync(p: Uint8Array): number | null;
    /**
     * Reads the next `p.length` bytes from the buffer or until the buffer is
     * drained. Resolves to the number of bytes read. If the buffer has no
     * data to return, resolves to EOF (`null`).
     *
     * NOTE: This methods reads bytes synchronously; it's provided for
     * compatibility with `Reader` interfaces.
     *
     * @example Usage
     * ```ts
     * import { Buffer } from "@std/io/buffer";
     * import { assertEquals } from "@std/assert/equals";
     *
     * const buf = new Buffer();
     * await buf.write(new TextEncoder().encode("Hello, world!"));
     *
     * const data = new Uint8Array(5);
     * const res = await buf.read(data);
     *
     * assertEquals(res, 5);
     * assertEquals(new TextDecoder().decode(data), "Hello");
     * ```
     *
     * @param p The buffer to read data into.
     * @returns The number of bytes read.
     */
    read(p: Uint8Array): Promise<number | null>;
    /**
     * Writes the given data to the buffer.
     *
     * @example Usage
     * ```ts
     * import { Buffer } from "@std/io/buffer";
     * import { assertEquals } from "@std/assert/equals";
     *
     * const buf = new Buffer();
     * const data = new TextEncoder().encode("Hello, world!");
     * buf.writeSync(data);
     *
     * const slice = buf.bytes();
     * assertEquals(new TextDecoder().decode(slice), "Hello, world!");
     * ```
     *
     * @param p The data to write to the buffer.
     * @returns The number of bytes written.
     */
    writeSync(p: Uint8Array): number;
    /**
     * Writes the given data to the buffer. Resolves to the number of bytes
     * written.
     *
     * > [!NOTE]
     * > This methods writes bytes synchronously; it's provided for compatibility
     * > with the {@linkcode Writer} interface.
     *
     * @example Usage
     * ```ts
     * import { Buffer } from "@std/io/buffer";
     * import { assertEquals } from "@std/assert/equals";
     *
     * const buf = new Buffer();
     * const data = new TextEncoder().encode("Hello, world!");
     * await buf.write(data);
     *
     * const slice = buf.bytes();
     * assertEquals(new TextDecoder().decode(slice), "Hello, world!");
     * ```
     *
     * @param p The data to write to the buffer.
     * @returns The number of bytes written.
     */
    write(p: Uint8Array): Promise<number>;
    /** Grows the buffer's capacity, if necessary, to guarantee space for
     * another `n` bytes. After `.grow(n)`, at least `n` bytes can be written to
     * the buffer without another allocation. If `n` is negative, `.grow()` will
     * throw. If the buffer can't grow it will throw an error.
     *
     * Based on Go Lang's
     * {@link https://golang.org/pkg/bytes/#Buffer.Grow | Buffer.Grow}.
     *
     * @example Usage
     * ```ts
     * import { Buffer } from "@std/io/buffer";
     * import { assertEquals } from "@std/assert/equals";
     *
     * const buf = new Buffer();
     * buf.grow(10);
     * assertEquals(buf.capacity, 10);
     * ```
     *
     * @param n The number of bytes to grow the buffer by.
     */
    grow(n: number): void;
    /**
     * Reads data from `r` until EOF (`null`) and appends it to the buffer,
     * growing the buffer as needed. It resolves to the number of bytes read.
     * If the buffer becomes too large, `.readFrom()` will reject with an error.
     *
     * Based on Go Lang's
     * {@link https://golang.org/pkg/bytes/#Buffer.ReadFrom | Buffer.ReadFrom}.
     *
     * @example Usage
     * ```ts
     * import { Buffer } from "@std/io/buffer";
     * import { assertEquals } from "@std/assert/equals";
     *
     * const buf = new Buffer();
     * const r = new Buffer(new TextEncoder().encode("Hello, world!"));
     * const n = await buf.readFrom(r);
     *
     * assertEquals(n, 13);
     * ```
     *
     * @param r The reader to read from.
     * @returns The number of bytes read.
     */
    readFrom(r: Reader): Promise<number>;
    /** Reads data from `r` until EOF (`null`) and appends it to the buffer,
     * growing the buffer as needed. It returns the number of bytes read. If the
     * buffer becomes too large, `.readFromSync()` will throw an error.
     *
     * Based on Go Lang's
     * {@link https://golang.org/pkg/bytes/#Buffer.ReadFrom | Buffer.ReadFrom}.
     *
     * @example Usage
     * ```ts
     * import { Buffer } from "@std/io/buffer";
     * import { assertEquals } from "@std/assert/equals";
     *
     * const buf = new Buffer();
     * const r = new Buffer(new TextEncoder().encode("Hello, world!"));
     * const n = buf.readFromSync(r);
     *
     * assertEquals(n, 13);
     * ```
     *
     * @param r The reader to read from.
     * @returns The number of bytes read.
     */
    readFromSync(r: ReaderSync): number;
}
//# sourceMappingURL=buffer.d.ts.map