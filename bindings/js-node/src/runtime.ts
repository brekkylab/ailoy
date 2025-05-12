import type * as ailoy from "./ailoy_addon.node";

const addon = require("./ailoy_addon.node");
const startThreads: typeof ailoy.startThreads = addon.startThreads;
const stopThreads: typeof ailoy.stopThreads = addon.stopThreads;
export const generateUUID: typeof ailoy.generateUUID = addon.generateUUID;
const BrokerClient: typeof ailoy.BrokerClient = addon.BrokerClient;
type Packet = ailoy.Packet;

export type HTTPRequestType = {
  url: string;
  method: "GET" | "POST" | "PUT" | "DELETE";
  headers: { [key: string]: string };
};

export class Runtime {
  private url: string;

  private client: ailoy.BrokerClient;

  private resolvers: Map<
    string,
    { resolve: () => void; reject: (reason: any) => void }
  >;

  private responses: Map<string, Packet>;

  private execResolvers: Map<
    string,
    {
      index: number;
      setFinish: () => void;
      resolve: (value: IteratorResult<any>) => void;
      reject: (reason: any) => void;
    }
  >;

  private execResponses: Map<string, Map<number, Packet>>;

  /**
   * Only one listen function should be running at any given time.
   * To prevent multiple logic paths from calling listen redundantly,
   * the Promise of the currently running listen function is cached.
   */
  private listener: Promise<void> | null;

  constructor(url: string = "inproc://") {
    this.url = url;
    this.resolvers = new Map();
    this.responses = new Map();
    this.execResolvers = new Map();
    this.execResponses = new Map();
    this.listener = null;
    startThreads(this.url);
    this.client = new BrokerClient(url);
  }

  start(): Promise<void> {
    const txid = generateUUID();
    if (!this.client.send_type1(txid, "connect"))
      throw Error("Connection failed");
    return new Promise<void>(async (resolve, reject) => {
      this.registerResolver(txid, resolve, reject);
      while (this.resolvers.has(txid)) await this.listen();
    });
  }

  stop(): Promise<void> {
    const txid = generateUUID();
    if (!this.client.send_type1(txid, "disconnect"))
      throw Error("Disconnection failed");
    return new Promise<void>(async (resolve, reject) => {
      this.registerResolver(txid, resolve, reject);
      while (this.resolvers.has(txid)) await this.listen();
      stopThreads(this.url);
    });
  }

  async call(funcName: string, inputs: any = null): Promise<any> {
    let rv: any[] = [];
    for await (const out of this.callIter(funcName, inputs)) {
      rv.push(out);
    }
    switch (rv.length) {
      case 0:
        return null;
      case 1:
        return rv[0];
      default:
        return rv;
    }
  }

  callIter(funcName: string, inputs: any = null): AsyncIterableIterator<any> {
    const txid = generateUUID();
    const sendResult = this.client.send_type2(
      txid,
      "execute",
      "call_function",
      funcName,
      inputs
    );
    if (!sendResult) throw Error("Call failed");
    return this.handleListenIter(txid);
  }

  define(
    componentType: string,
    componentName: string,
    inputs: any = {}
  ): Promise<boolean> {
    const txid = generateUUID();
    const sendResult = this.client.send_type2(
      txid,
      "execute",
      "define_component",
      componentType,
      componentName,
      inputs
    );
    if (!sendResult) throw Error("Define failed");
    return this.handleListen(txid);
  }

  delete(componentName: string): Promise<boolean> {
    const txid = generateUUID();
    const sendResult = this.client.send_type2(
      txid,
      "execute",
      "delete_component",
      componentName
    );
    if (!sendResult) throw Error("Delete failed");
    return this.handleListen(txid);
  }

  async callMethod(
    componentName: string,
    methodName: string,
    inputs: any = null
  ): Promise<any> {
    let rv: any[] = [];
    for await (const out of this.callIterMethod(
      componentName,
      methodName,
      inputs
    )) {
      rv.push(out);
    }
    switch (rv.length) {
      case 0:
        return null;
      case 1:
        return rv[0];
      default:
        return rv;
    }
  }

  callIterMethod(
    componentName: string,
    methodName: string,
    inputs: any = null
  ): AsyncIterableIterator<any> {
    const txid = generateUUID();
    const sendResult = this.client.send_type2(
      txid,
      "execute",
      "call_method",
      componentName,
      methodName,
      inputs
    );
    if (!sendResult) throw Error("CallMethod failed");
    return this.handleListenIter(txid);
  }

  private registerResolver(
    txid: string,
    resolve: () => void,
    reject: (resaon: any) => void
  ) {
    const response = this.responses.get(txid);
    if (response) {
      this.responses.delete(txid);
      if (response.body.status) resolve();
      else reject(response.body.reason as string);
    } else {
      this.resolvers.set(txid, { resolve, reject });
    }
  }

  private registerResponse(packet: Packet) {
    const txid = packet.headers[0] as string;
    const status = packet.body.status as boolean;
    const resolver = this.resolvers.get(txid);
    if (resolver) {
      const { resolve, reject } = resolver;
      this.resolvers.delete(txid);
      if (status) resolve();
      else reject(packet.body.reason as string);
    } else {
      this.responses.set(txid, packet);
    }
  }

  private registerExecResolver(
    txid: string,
    index: number,
    setFinish: () => void,
    resolve: (value: IteratorResult<any, void>) => void,
    reject: (resaon: any) => void
  ) {
    if (
      this.execResponses.has(txid) &&
      this.execResponses.get(txid)!.has(index)
    ) {
      const response = this.execResponses.get(txid)!.get(index)!;
      this.execResponses.get(txid)!.delete(index);
      if (this.execResponses.get(txid)!.size == 0)
        this.execResponses.delete(txid);
      if (response.body.status) {
        const finished = response.headers[2] as boolean;
        if (finished) setFinish();
        resolve({ value: response.body.out, done: false });
      } else reject(response.body.reason);
    } else {
      if (this.execResolvers.has(txid))
        throw Error(
          `iterResolver for ${txid} already registerd at index ${index}`
        );
      this.execResolvers.set(txid, { index, setFinish, resolve, reject });
    }
  }

  private registerExecResponse(packet: Packet) {
    const txid = packet.headers[0] as string;
    const index = packet.headers[1] as number;
    const finished = packet.headers[2] as boolean;
    const status = packet.body.status as boolean;
    if (
      this.execResolvers.has(txid) &&
      this.execResolvers.get(txid)!.index === index
    ) {
      const { setFinish, resolve, reject } = this.execResolvers.get(txid)!;
      this.execResolvers.delete(txid);
      if (finished) setFinish();
      if (status) resolve({ value: packet.body.out, done: false });
      else reject(packet.body.reason as string);
    } else {
      if (!this.execResponses.has(txid))
        this.execResponses.set(txid, new Map());
      this.execResponses.get(txid)!.set(index, packet);
    }
  }

  private listen(): Promise<void> {
    if (!this.listener) {
      this.listener = new Promise(async (resolve, reject) => {
        const resp = await this.client.listen();
        this.listener = null;
        if (!resp) {
          resolve();
        } else if (resp.packet_type == "respond") {
          this.registerResponse(resp);
          resolve();
        } else if (resp.packet_type == "respond_execute") {
          this.registerExecResponse(resp);
          resolve();
        } else {
          reject(`Undefined packet type ${resp.packet_type}`);
        }
      });
    }
    return this.listener!;
  }

  /**
   * Performs a single listen operation.
   * For functions like `define_component` and `delete_component`.
   * @param txid
   * @returns
   */
  private handleListen(txid: string): Promise<boolean> {
    const registerExecResolver = this.registerExecResolver.bind(this);
    const execResolvers = this.execResolvers;
    const listen = this.listen.bind(this);
    return new Promise<boolean>(async (resolve, reject) => {
      let iterator = {
        [Symbol.asyncIterator]() {
          return this;
        },
        next() {
          return new Promise<IteratorResult<any>>(
            async (innerResolve, innerReject) => {
              registerExecResolver(
                txid,
                0,
                () => {},
                innerResolve,
                innerReject
              );
              while (execResolvers.has(txid)) await listen();
            }
          );
        },
      };
      iterator
        .next()
        .then(() => resolve(true))
        .catch((reason) => reject(reason));
    });
  }

  /**
   * * performs iterative listen operations.
   * For functions like `call_function` and `call_method`.
   * @param txid
   * @returns
   */
  private handleListenIter(txid: string): AsyncIterableIterator<any> {
    const registerExecResolver = this.registerExecResolver.bind(this);
    const execResolvers = this.execResolvers;
    const listen = this.listen.bind(this);
    let idx = 0;
    let finished = false;
    const setFinish = () => {
      finished = true;
    };
    return {
      [Symbol.asyncIterator]() {
        return this;
      },
      next() {
        return new Promise<IteratorResult<any>>(async (resolve, reject) => {
          if (finished) {
            resolve({ value: undefined, done: true });
          } else {
            registerExecResolver(txid, idx, setFinish, resolve, reject);
            idx += 1;
            while (execResolvers.has(txid)) await listen();
          }
        });
      },
    };
  }
}

export async function startRuntime(url?: string): Promise<Runtime> {
  const rt = new Runtime(url);
  await rt.start();
  return rt;
}
