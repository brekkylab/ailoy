// TypeScript bindings for emscripten-generated code.  Automatically generated at compile time.
declare namespace RuntimeExports {
    /**
     * @param {string|null=} returnType
     * @param {Array=} argTypes
     * @param {Array=} args
     * @param {Object=} opts
     */
    function ccall(ident: any, returnType?: (string | null) | undefined, argTypes?: any[] | undefined, args?: any[] | undefined, opts?: any | undefined): any;
    /**
     * @param {string=} returnType
     * @param {Array=} argTypes
     * @param {Object=} opts
     */
    function cwrap(ident: any, returnType?: string | undefined, argTypes?: any[] | undefined, opts?: any | undefined): (...args: any[]) => any;
    function stringToNewUTF8(str: any): any;
}
interface WasmModule {
}

type EmbindString = ArrayBuffer|Uint8Array|Uint8ClampedArray|Int8Array|string;
export interface ClassHandle {
  isAliasOf(other: ClassHandle): boolean;
  delete(): void;
  deleteLater(): this;
  isDeleted(): boolean;
  // @ts-ignore - If targeting lower than ESNext, this symbol might not exist.
  [Symbol.dispose](): void;
  clone(): this;
}
export interface VectorString extends ClassHandle {
  push_back(_0: EmbindString): void;
  resize(_0: number, _1: EmbindString): void;
  size(): number;
  get(_0: number): EmbindString | undefined;
  set(_0: number, _1: EmbindString): boolean;
}

export interface NDArray extends ClassHandle {
  toString(): string;
  valueOf(): string;
  getShape(): any;
  getDtype(): string;
  getData(): any;
}

export interface BrokerClient extends ClassHandle {
  send_type1(_0: EmbindString, _1: EmbindString): boolean;
  send_type2(_0: EmbindString, _1: EmbindString, _2: EmbindString, _3: any): boolean;
  send_type3(_0: EmbindString, _1: EmbindString, _2: boolean, _3: number, _4: any): boolean;
  listen(): any;
}

interface EmbindModule {
  VectorString: {
    new(): VectorString;
  };
  start_threads(): void;
  stop_threads(): void;
  generate_uuid(): string;
  NDArray: {
    new(_0: any): NDArray;
  };
  BrokerClient: {
    new(_0: EmbindString): BrokerClient;
  };
}

export type MainModule = WasmModule & typeof RuntimeExports & EmbindModule;
export default function MainModuleFactory (options?: unknown): Promise<MainModule>;
