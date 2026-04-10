/* tslint:disable */
/* eslint-disable */

/**
 * Shared state for the WASM module.
 */
export class MyceliumWeb {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Create a latent vector from raw data.
     */
    create_latent(data: Float32Array, layer: number): any;
    /**
     * Decode a latent vector from JSON.
     */
    decode_latent(json: string): any;
    /**
     * Initialize WebGPU compute.
     */
    init_gpu(): Promise<boolean>;
    /**
     * Run latent vector interpolation (lerp) on GPU.
     */
    latent_lerp(data_a: Float32Array, data_b: Float32Array, t: number): Promise<Float32Array>;
    /**
     * Run matrix multiplication on GPU.
     */
    matmul(a: Float32Array, b: Float32Array, m: number, k: number, n: number): Promise<Float32Array>;
    /**
     * Create a new Mycelium web node.
     */
    constructor();
    /**
     * Generate a random latent vector (for testing).
     */
    random_latent(dim: number, _layer: number): Float32Array;
    /**
     * RMS normalization on GPU.
     */
    rms_norm(input: Float32Array, eps: number): Promise<Float32Array>;
    /**
     * Apply SiLU activation on GPU.
     */
    silu_activation(input: Float32Array): Promise<Float32Array>;
    /**
     * Get the status of the WebGPU node.
     */
    status(): any;
    /**
     * Get node capabilities (for browser).
     */
    readonly capabilities: any;
    /**
     * Get model configuration.
     */
    readonly model_config: any;
}

/**
 * Initialize the WASM module with console logging.
 */
export function start(): void;

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_myceliumweb_free: (a: number, b: number) => void;
    readonly myceliumweb_capabilities: (a: number) => number;
    readonly myceliumweb_create_latent: (a: number, b: number, c: number, d: number, e: number) => void;
    readonly myceliumweb_decode_latent: (a: number, b: number, c: number, d: number) => void;
    readonly myceliumweb_init_gpu: (a: number) => number;
    readonly myceliumweb_latent_lerp: (a: number, b: number, c: number, d: number, e: number, f: number) => number;
    readonly myceliumweb_matmul: (a: number, b: number, c: number, d: number, e: number, f: number, g: number, h: number) => number;
    readonly myceliumweb_model_config: (a: number) => number;
    readonly myceliumweb_new: () => number;
    readonly myceliumweb_random_latent: (a: number, b: number, c: number, d: number) => void;
    readonly myceliumweb_rms_norm: (a: number, b: number, c: number, d: number) => number;
    readonly myceliumweb_silu_activation: (a: number, b: number, c: number) => number;
    readonly myceliumweb_status: (a: number) => number;
    readonly start: () => void;
    readonly __wasm_bindgen_func_elem_2287: (a: number, b: number, c: number, d: number) => void;
    readonly __wasm_bindgen_func_elem_2291: (a: number, b: number, c: number, d: number) => void;
    readonly __wasm_bindgen_func_elem_680: (a: number, b: number, c: number) => void;
    readonly __wbindgen_export: (a: number, b: number) => number;
    readonly __wbindgen_export2: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_export3: (a: number) => void;
    readonly __wbindgen_export4: (a: number, b: number, c: number) => void;
    readonly __wbindgen_export5: (a: number, b: number) => void;
    readonly __wbindgen_add_to_stack_pointer: (a: number) => number;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;
