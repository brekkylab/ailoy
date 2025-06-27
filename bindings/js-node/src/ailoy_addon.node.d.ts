export type NDArrayDtype =
  | "uint8"
  | "uint16"
  | "uint32"
  | "uint64"
  | "int8"
  | "int16"
  | "int32"
  | "int64"
  | "float32"
  | "float64";

export type NDArrayData =
  | Uint8Array
  | Uint16Array
  | Uint32Array
  | BigUint64Array
  | Int8Array
  | Int16Array
  | Int32Array
  | BigInt64Array
  | Float32Array
  | Float64Array;

export declare class NDArray {
  constructor(arg: {
    shape: Array<number>;
    dtype: NDArrayDtype;
    data: NDArrayData;
  });
  getShape(): Array<number>;
  getDtype(): NDArrayDtype;
  getData(): NDArrayData;
}

export type PacketType =
  | "connect"
  | "disconnect"
  | "subscribe"
  | "unsubscribe"
  | "execute"
  | "respond"
  | "respond_execute";

export type InstructionType =
  | "call_function"
  | "call_method"
  | "define_compoennt"
  | "delete_component";

export interface Packet {
  packet_type: PacketType;
  instruction_type: InstructionType | undefined;
  headers: Array<string | boolean | number>;
  body: Record<string, any>;
}

export declare class BrokerClient {
  constructor(url: string);

  send_type1(txid: string, ptype: "connect" | "disconnect"): boolean;

  send_type2(
    txid: string,
    ptype: "subscribe" | "unsubscribe",
    itype: "call_function",
    fname: string
  ): boolean;

  send_type2(
    txid: string,
    ptype: "subscribe" | "unsubscribe",
    itype: "define_component",
    ctname: string
  ): boolean;

  send_type2(
    txid: string,
    ptype: "subscribe" | "unsubscribe",
    itype: "delete_component",
    cname: string
  ): boolean;

  send_type2(
    txid: string,
    ptype: "subscribe" | "unsubscribe",
    itype: "call_method",
    cname: string,
    fname: string
  ): boolean;

  send_type2(
    txid: string,
    ptype: "execute",
    itype: "call_function",
    fname: string,
    input: any
  ): boolean;

  send_type2(
    txid: string,
    ptype: "execute",
    itype: "define_component",
    ctname: string,
    cname: string,
    input: any
  ): boolean;

  send_type2(
    txid: string,
    ptype: "execute",
    itype: "delete_component",
    cname: string
  ): boolean;

  send_type2(
    txid: string,
    ptype: "execute",
    itype: "call_method",
    cname: string,
    fname: string,
    input: any
  ): boolean;

  send_type3(
    txid: string,
    ptype: "respond_execute",
    status: true,
    sequence: number,
    finish: boolean,
    out: any
  ): boolean;

  send_type3(
    txid: string,
    ptype: "respond_execute",
    status: false,
    sequence: number,
    reason: string
  ): boolean;

  listen(): Promise<Packet | null>;
}

export declare function startThreads(url: string): void;

export declare function stopThreads(url: string): void;

export declare function generateUUID(): string;
