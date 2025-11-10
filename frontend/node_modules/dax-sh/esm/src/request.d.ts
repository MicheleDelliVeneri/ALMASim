import * as dntShim from "../_dnt.shims.js";
import { Path } from "../deps/jsr.io/@david/path/0.2.0/mod.js";
import { type Delay } from "./common.js";
import type { ProgressBar } from "./console/mod.js";
interface RequestBuilderState {
    noThrow: boolean | number[];
    url: string | URL | undefined;
    body: dntShim.BodyInit | undefined;
    cache: dntShim.RequestCache | undefined;
    headers: Record<string, string | undefined>;
    integrity: string | undefined;
    keepalive: boolean | undefined;
    method: string | undefined;
    mode: dntShim.RequestMode | undefined;
    progressBarFactory: ((message: string) => ProgressBar) | undefined;
    redirect: dntShim.RequestRedirect | undefined;
    referrer: string | undefined;
    referrerPolicy: dntShim.ReferrerPolicy | undefined;
    progressOptions: {
        noClear: boolean;
    } | undefined;
    timeout: number | undefined;
}
/** @internal */
export declare const withProgressBarFactorySymbol: unique symbol;
/**
 * Builder API for downloading files.
 */
export declare class RequestBuilder implements PromiseLike<RequestResponse> {
    #private;
    then<TResult1 = RequestResponse, TResult2 = never>(onfulfilled?: ((value: RequestResponse) => TResult1 | PromiseLike<TResult1>) | null | undefined, onrejected?: ((reason: any) => TResult2 | PromiseLike<TResult2>) | null | undefined): PromiseLike<TResult1 | TResult2>;
    /** Fetches and gets the response. */
    fetch(): Promise<RequestResponse>;
    /** Specifies the URL to send the request to. */
    url(value: string | URL | undefined): RequestBuilder;
    /** Sets multiple headers at the same time via an object literal. */
    header(items: Record<string, string | undefined>): RequestBuilder;
    /** Sets a header to send with the request. */
    header(name: string, value: string | undefined): RequestBuilder;
    /**
     * Do not throw if a non-2xx status code is received.
     *
     * By default the request builder will throw when
     * receiving a non-2xx status code. Specify this
     * to have it not throw.
     */
    noThrow(value?: boolean): RequestBuilder;
    /**
     * Do not throw if a non-2xx status code is received
     * except for these excluded provided status codes.
     *
     * This overload may be especially useful when wanting to ignore
     * 404 status codes and have it return undefined instead. For example:
     *
     * ```ts
     * const data = await $.request(`https://crates.io/api/v1/crates/${crateName}`)
     *   .noThrow(404)
     *   .json<CratesIoMetadata | undefined>();
     * ```
     *
     * Note, use multiple arguments to ignore multiple status codes (ex. `.noThrow(400, 404)`) as
     * multiple calls to `.noThrow()` will overwrite the previous.
     */
    noThrow(exclusionStatusCode: number, ...additional: number[]): RequestBuilder;
    body(value: dntShim.BodyInit | undefined): RequestBuilder;
    cache(value: dntShim.RequestCache | undefined): RequestBuilder;
    integrity(value: string | undefined): RequestBuilder;
    keepalive(value: boolean): RequestBuilder;
    method(value: string): RequestBuilder;
    mode(value: dntShim.RequestMode): RequestBuilder;
    /** @internal */
    [withProgressBarFactorySymbol](factory: (message: string) => ProgressBar): RequestBuilder;
    redirect(value: dntShim.RequestRedirect): RequestBuilder;
    referrer(value: string | undefined): RequestBuilder;
    referrerPolicy(value: dntShim.ReferrerPolicy | undefined): RequestBuilder;
    /** Shows a progress bar while downloading with the provided options. */
    showProgress(opts: {
        noClear?: boolean;
    }): RequestBuilder;
    /** Shows a progress bar while downloading. */
    showProgress(show?: boolean): RequestBuilder;
    /** Timeout the request after the specified delay throwing a `TimeoutError`. */
    timeout(delay: Delay | undefined): RequestBuilder;
    /** Fetches and gets the response as an array buffer. */
    arrayBuffer(): Promise<ArrayBuffer>;
    /** Fetches and gets the response as a blob. */
    blob(): Promise<Blob>;
    /** Fetches and gets the response as form data. */
    formData(): Promise<FormData>;
    /** Fetches and gets the response as JSON additionally setting
     * a JSON accept header if not set. */
    json<TResult = any>(): Promise<TResult>;
    /** Fetches and gets the response as text. */
    text(): Promise<string>;
    /** Pipes the response body to the provided writable stream. */
    pipeTo(dest: dntShim.WritableStream<Uint8Array>, options?: dntShim.StreamPipeOptions): Promise<void>;
    /**
     * Pipes the response body to a file.
     *
     * @remarks The path will be derived from the request's url
     * and downloaded to the current working directory.
     *
     * @returns The path reference of the downloaded file.
     */
    pipeToPath(options?: dntShim.Deno.WriteFileOptions): Promise<Path>;
    /**
     * Pipes the response body to a file.
     *
     * @remarks If no path is provided then it will be derived from the
     * request's url and downloaded to the current working directory.
     *
     * @returns The path reference of the downloaded file.
     */
    pipeToPath(path?: string | URL | Path | undefined, options?: dntShim.Deno.WriteFileOptions): Promise<Path>;
    /** Pipes the response body through the provided transform. */
    pipeThrough<T>(transform: {
        writable: dntShim.WritableStream<Uint8Array>;
        readable: dntShim.ReadableStream<T>;
    }): Promise<dntShim.ReadableStream<T>>;
}
interface RequestAbortController {
    controller: AbortController;
    /** Clears the timeout that may be set if there's a delay */
    clearTimeout(): void;
}
/** Response of making a request where the body can be read. */
export declare class RequestResponse {
    #private;
    /** @internal */
    constructor(opts: {
        response: Response;
        originalUrl: string;
        progressBar: ProgressBar | undefined;
        abortController: RequestAbortController;
    });
    /** Raw response. */
    get response(): Response;
    /** Response headers. */
    get headers(): Headers;
    /** If the response had a 2xx code. */
    get ok(): boolean;
    /** If the response is the result of a redirect. */
    get redirected(): boolean;
    /** The underlying `AbortSignal` used to abort the request body
     * when a timeout is reached or when the `.abort()` method is called. */
    get signal(): AbortSignal;
    /** Status code of the response. */
    get status(): number;
    /** Status text of the response. */
    get statusText(): string;
    /** URL of the response. */
    get url(): string;
    /** Aborts  */
    abort(reason?: unknown): void;
    /**
     * Throws if the response doesn't have a 2xx code.
     *
     * This might be useful if the request was built with `.noThrow()`, but
     * otherwise this is called automatically for any non-2xx response codes.
     */
    throwIfNotOk(): void;
    /**
     * Respose body as an array buffer.
     *
     * Note: Returns `undefined` when `.noThrow(404)` and status code is 404.
     */
    arrayBuffer(): Promise<ArrayBuffer>;
    /**
     * Response body as a blog.
     *
     * Note: Returns `undefined` when `.noThrow(404)` and status code is 404.
     */
    blob(): Promise<Blob>;
    /**
     * Response body as a form data.
     *
     * Note: Returns `undefined` when `.noThrow(404)` and status code is 404.
     */
    formData(): Promise<FormData>;
    /**
     * Respose body as JSON.
     *
     * Note: Returns `undefined` when `.noThrow(404)` and status code is 404.
     */
    json<TResult = any>(): Promise<TResult>;
    /**
     * Respose body as text.
     *
     * Note: Returns `undefined` when `.noThrow(404)` and status code is 404.
     */
    text(): Promise<string>;
    /** Pipes the response body to the provided writable stream. */
    pipeTo(dest: dntShim.WritableStream<Uint8Array>, options?: dntShim.StreamPipeOptions): Promise<void>;
    /**
     * Pipes the response body to a file.
     *
     * @remarks The path will be derived from the request's url
     * and downloaded to the current working directory.
     *
     * @remarks  If the path is a directory, then the file name will be derived
     * from the request's url and the file will be downloaded to the provided directory
     *
     * @returns The path reference of the downloaded file
     */
    pipeToPath(options?: dntShim.Deno.WriteFileOptions): Promise<Path>;
    /**
     * Pipes the response body to a file.
     *
     * @remarks If no path is provided then it will be derived from the
     * request's url and downloaded to the current working directory.
     *
     * @remarks  If the path is a directory, then the file name will be derived
     * from the request's url and the file will be downloaded to the provided directory
     *
     * @returns The path reference of the downloaded file
     */
    pipeToPath(path?: string | URL | Path | undefined, options?: dntShim.Deno.WriteFileOptions): Promise<Path>;
    /** Pipes the response body through the provided transform. */
    pipeThrough<T>(transform: {
        writable: dntShim.WritableStream<Uint8Array>;
        readable: dntShim.ReadableStream<T>;
    }): dntShim.ReadableStream<T>;
    get readable(): dntShim.ReadableStream<Uint8Array>;
}
export declare function makeRequest(state: RequestBuilderState): Promise<RequestResponse>;
export {};
//# sourceMappingURL=request.d.ts.map