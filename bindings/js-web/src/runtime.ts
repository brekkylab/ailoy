import AiloyModule, { MainModule, BrokerClient } from "./ailoy_js_web.js";

type PacketType =
  | "connect"
  | "disconnect"
  | "subscribe"
  | "unsubscribe"
  | "execute"
  | "respond"
  | "respond_execute";

type InstructionType =
  | "call_function"
  | "call_method"
  | "define_compoennt"
  | "delete_component";

interface Packet {
  packet_type: PacketType;
  instruction_type: InstructionType | undefined;
  headers: Array<string | boolean | number>;
  body: Record<string, any>;
}

async function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

class Runtime {
  private module: MainModule | undefined;
  private brokerClient: BrokerClient | undefined;

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

  constructor() {
    this.resolvers = new Map();
    this.responses = new Map();
    this.execResolvers = new Map();
    this.execResponses = new Map();
    this.listener = null;
  }

  async start() {
    if (this.isAlive()) return;

    // initialize Emscripten module
    this.module = await AiloyModule();

    // start broker and vm threads
    await this.module.start_threads();

    // initialize broker client
    this.brokerClient = new this.module.BrokerClient("inproc://");

    // connect to broker
    const txid = await this.#_sendType1("connect");
    return new Promise<void>(async (resolve, reject) => {
      this.#_registerResolver(txid, resolve, reject);
      while (this.resolvers.has(txid)) await this.#_listen();
    });
  }

  async stop() {
    if (!this.isAlive()) return;

    // disconnect from broker
    const txid = await this.#_sendType1("disconnect");
    return new Promise<void>(async (resolve, reject) => {
      this.#_registerResolver(txid, resolve, reject);
      while (this.resolvers.has(txid)) await this.#_listen();
      this.module!.stop_threads();

      this.brokerClient = undefined;
      this.module = undefined;
    });
  }

  isAlive() {
    return this.module !== undefined && this.brokerClient !== undefined;
  }

  generateUUID() {
    return this.module!.generate_uuid();
  }

  async #_sendType1(ptype: "connect" | "disconnect") {
    const txid = this.generateUUID();
    let retryCount = 0;
    while (retryCount < 3) {
      if (this.brokerClient!.send_type1(txid, ptype)) {
        return txid;
      }
      retryCount += 1;
      await sleep(100);
    }
    throw Error(`Failed to send package "${ptype}"`);
  }

  async call(funcName: string, inputs: any = {}): Promise<any> {
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

  callIter(funcName: string, inputs: any = {}): AsyncIterableIterator<any> {
    const txid = this.generateUUID();
    const sendResult = this.brokerClient!.send_type2(
      txid,
      "execute",
      "call_function",
      [funcName, inputs]
    );
    if (!sendResult) {
      throw Error("Call failed");
    }
    return this.#_handleListenIter(txid);
  }

  define(
    componentType: string,
    componentName: string,
    inputs: any = {}
  ): Promise<boolean> {
    const txid = this.generateUUID();
    const sendResult = this.brokerClient!.send_type2(
      txid,
      "execute",
      "define_component",
      [componentType, componentName, inputs]
    );
    if (!sendResult) throw Error("Define component failed");
    return this.#_handleListen(txid);
  }

  delete(componentName: string): Promise<boolean> {
    const txid = this.generateUUID();
    const sendResult = this.brokerClient!.send_type2(
      txid,
      "execute",
      "delete_component",
      [componentName]
    );
    if (!sendResult) throw Error("Delete component failed");
    return this.#_handleListen(txid);
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
    const txid = this.generateUUID();
    const sendResult = this.brokerClient!.send_type2(
      txid,
      "execute",
      "call_method",
      [componentName, methodName, inputs]
    );
    if (!sendResult) throw Error("Call method failed");
    return this.#_handleListenIter(txid);
  }

  #_registerResolver(
    txid: string,
    resolve: () => void,
    reject: (reason: any) => void
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

  #_registerResponse(packet: Packet) {
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

  #_registerExecResolver(
    txid: string,
    index: number,
    setFinish: () => void,
    resolve: (value: IteratorResult<any, void>) => void,
    reject: (reason: any) => void
  ) {
    if (
      this.execResponses.has(txid) &&
      this.execResponses.get(txid)!.has(index)
    ) {
      const response = this.execResponses.get(txid)!.get(index)!;
      this.execResponses.get(txid)!.delete(index);
      if (this.execResponses.get(txid)!.size === 0)
        this.execResponses.delete(txid);
      if (response.body.status) {
        const finished = response.headers[2] as boolean;
        if (finished) setFinish();
        resolve({ value: response.body.out, done: false });
      } else reject(response.body.reason);
    } else {
      if (this.execResolvers.has(txid))
        throw Error(
          `iterResolver for ${txid} already registered at index ${index}`
        );
      this.execResolvers.set(txid, { index, setFinish, resolve, reject });
    }
  }

  #_registerExecResponse(packet: Packet) {
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

  #_listen(): Promise<void> {
    if (!this.listener) {
      this.listener = new Promise(async (resolve, reject) => {
        const resp = await this.brokerClient!.listen();
        this.listener = null;
        if (!resp) {
          resolve();
        } else if (resp.packet_type === "respond") {
          this.#_registerResponse(resp);
          resolve();
        } else if (resp.packet_type === "respond_execute") {
          this.#_registerExecResponse(resp);
          resolve();
        } else {
          reject(`Undefined packet type ${resp.packet_type}`);
        }
      });
    }
    return this.listener!;
  }

  #_handleListen(txid: string): Promise<boolean> {
    const registerExecResolver = this.#_registerExecResolver.bind(this);
    const execResolvers = this.execResolvers;
    const listen = this.#_listen.bind(this);
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

  #_handleListenIter(txid: string): AsyncIterableIterator<any> {
    const registerExecResolver = this.#_registerExecResolver.bind(this);
    const execResolvers = this.execResolvers;
    const listen = this.#_listen.bind(this);
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

export { Runtime };
