import time
from asyncio import Event, to_thread
from collections import defaultdict
from typing import Any, AsyncGenerator, Generator, Literal, Optional, TypedDict

from .ailoy_py import BrokerClient, generate_uuid, start_threads, stop_threads

__all__ = ["Runtime", "AsyncRuntime"]


class Packet(TypedDict):
    packet_type: Literal["respond", "respond_execute"]
    instruction_type: Optional[Literal["call_function", "define_component", "delete_component", "call_method"]]
    headers: list[bool | int | str]
    body: dict[str, Any]


class RuntimeBase:
    __client_count: dict[str, int] = {}

    def __init__(self, url: str = "inproc://"):
        self.url: str = url
        self._responses: dict[str, Packet] = {}
        self._exec_responses: defaultdict[str, dict[int, Packet]] = defaultdict(dict)
        self._listen_lock: Optional[Event] = None

        if RuntimeBase.__client_count.get(self.url, 0) == 0:
            start_threads(self.url)
            RuntimeBase.__client_count[self.url] = 0

        self._client: BrokerClient = BrokerClient(self.url)
        txid = self._send_type1("connect")
        self._sync_listen()
        if not self._responses[txid]["body"]["status"]:
            raise RuntimeError("Connection failed")
        del self._responses[txid]
        RuntimeBase.__client_count[self.url] += 1

    def __del__(self):
        self.stop()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.stop()

    def stop(self):
        if self.is_alive():
            txid = self._send_type1("disconnect")
            while txid not in self._responses:
                self._sync_listen()
            if not self._responses[txid]["body"]["status"]:
                raise RuntimeError("Disconnection failed")
            self._client = None
            RuntimeBase.__client_count[self.url] -= 1
        if RuntimeBase.__client_count.get(self.url, 0) <= 0:
            stop_threads(self.url)
            RuntimeBase.__client_count.pop(self.url, 0)

    def is_alive(self):
        return self._client is not None

    def _send_type1(self, ptype: Literal["connect", "disconnect"]) -> str:
        txid = generate_uuid()
        retry_count = 0
        # Since the broker thread might start slightly later than the runtime client,
        # we retry sending the packat a few times to ensure delivery.
        while retry_count < 3:
            if self._client.send_type1(txid, ptype):
                return txid
            time.sleep(0.001)
            retry_count += 1
        raise RuntimeError(f'Failed to send packet "{ptype}"')

    def _send_type2(
        self,
        ptype: Literal["subscribe", "unsubscribe", "execute"],
        itype: Literal["call_function", "define_component", "delete_component", "call_method"],
        *args,
    ):
        txid = generate_uuid()
        if self._client.send_type2(txid, ptype, itype, *args):
            return txid
        raise RuntimeError("Failed to send packet")

    def _send_type3(
        self,
        ptype: Literal["respond", "respond_execute"],
        status: bool,
        *args,
    ):
        txid = generate_uuid()
        if self._client.send_type3(txid, ptype, status, *args):
            return txid
        raise RuntimeError("Failed to send packet")

    def _sync_listen(self) -> None:
        packet = self._client.listen()
        if packet is not None:
            txid = packet["headers"][0]
            if packet["packet_type"] == "respond_execute":
                idx = packet["headers"][1]
                self._exec_responses[txid][idx] = packet
            else:
                self._responses[txid] = packet

    async def _listen(self) -> None:
        # If listen lock exists -> wait
        if self._listen_lock:
            await self._listen_lock.wait()
        else:
            # Create a new event
            self._listen_lock = Event()
            # Listen packet
            packet = await to_thread(self._client.listen)
            if packet is not None:
                txid = packet["headers"][0]
                if packet["packet_type"] == "respond_execute":
                    idx = packet["headers"][1]
                    self._exec_responses[txid][idx] = packet
                else:
                    self._responses[txid] = packet
            # Emit event
            self._listen_lock.set()
            self._listen_lock = None


class Runtime(RuntimeBase):
    def __init__(self, url: str = "inproc://"):
        super().__init__(url)

    def call(self, func_name: str, input: Any) -> Any:
        rv = [v for v in self.call_iter(func_name, input)]
        if len(rv) == 0:
            return None
        elif len(rv) == 1:
            return rv[0]
        else:
            return rv

    def call_iter(self, func_name: str, input: Any) -> Generator[Any, None, None]:
        txid = self._send_type2("execute", "call_function", func_name, input)

        def generator():
            idx = 0
            finished = False
            while not finished:
                while idx not in self._exec_responses[txid]:
                    self._sync_listen()
                packet = self._exec_responses[txid].pop(idx)
                if not packet["body"]["status"]:
                    raise RuntimeError(packet["body"]["reason"])
                if packet["headers"][2]:
                    finished = True
                yield packet["body"]["out"]
                idx += 1
            del self._exec_responses[txid]

        return generator()

    def define(self, comp_type: str, comp_name: str, input: Any) -> None:
        txid = self._send_type2("execute", "define_component", comp_type, comp_name, input)
        while 0 not in self._exec_responses[txid]:
            self._sync_listen()
        packet = self._exec_responses[txid][0]
        if not packet["body"]["status"]:
            raise RuntimeError(packet["body"]["reason"])
        del self._exec_responses[txid]

    def delete(self, comp_name: str) -> None:
        txid = self._send_type2("execute", "delete_component", comp_name)
        while 0 not in self._exec_responses[txid]:
            self._sync_listen()
        packet = self._exec_responses[txid][0]
        if not packet["body"]["status"]:
            raise RuntimeError(packet["body"]["reason"])
        del self._exec_responses[txid]

    def call_method(self, comp_name: str, func_name: str, input: Any) -> Any:
        rv = [v for v in self.call_iter_method(comp_name, func_name, input)]
        if len(rv) == 0:
            return None
        elif len(rv) == 1:
            return rv[0]
        else:
            return rv

    def call_iter_method(self, comp_name: str, func_name: str, input: Any) -> Generator[Any, None, None]:
        txid = self._send_type2("execute", "call_method", comp_name, func_name, input)

        def generator():
            idx = 0
            finished = False
            while not finished:
                while idx not in self._exec_responses[txid]:
                    self._sync_listen()
                packet = self._exec_responses[txid].pop(idx)
                if not packet["body"]["status"]:
                    raise RuntimeError(packet["body"]["reason"])
                if packet["headers"][2]:
                    finished = True
                yield packet["body"]["out"]
                idx += 1
            del self._exec_responses[txid]

        return generator()


class AsyncRuntime(RuntimeBase):
    def __init__(self, url: str = "inproc://"):
        super().__init__(url)

    async def call(self, func_name: str, input: Any) -> Any:
        rv = [v async for v in self.call_iter(func_name, input)]
        if len(rv) == 0:
            return None
        elif len(rv) == 1:
            return rv[0]
        else:
            return rv

    def call_iter(self, func_name: str, input: Any) -> AsyncGenerator[Any, None]:
        txid = self._send_type2("execute", "call_function", func_name, input)

        async def generator():
            idx = 0
            finished = False
            while not finished:
                while idx not in self._exec_responses[txid]:
                    await self._listen()
                packet = self._exec_responses[txid].pop(idx)
                if not packet["body"]["status"]:
                    raise RuntimeError(packet["body"]["reason"])
                if packet["headers"][2]:
                    finished = True
                yield packet["body"]["out"]
                idx += 1
            del self._exec_responses[txid]

        return generator()

    async def define(self, comp_type: str, comp_name: str, input: Any) -> None:
        txid = self._send_type2("execute", "define_component", comp_type, comp_name, input)
        while 0 not in self._exec_responses[txid]:
            await self._listen()
        packet = self._exec_responses[txid][0]
        if not packet["body"]["status"]:
            raise RuntimeError(packet["body"]["reason"])
        del self._exec_responses[txid]

    async def delete(self, comp_name: str) -> None:
        txid = self._send_type2("execute", "delete_component", comp_name)
        while 0 not in self._exec_responses[txid]:
            await self._listen()
        packet = self._exec_responses[txid][0]
        if not packet["body"]["status"]:
            raise RuntimeError(packet["body"]["reason"])
        del self._exec_responses[txid]

    async def call_method(self, comp_name: str, func_name: str, input: Any) -> Any:
        rv = [v async for v in self.call_iter_method(comp_name, func_name, input)]
        if len(rv) == 0:
            return None
        elif len(rv) == 1:
            return rv[0]
        else:
            return rv

    def call_iter_method(self, comp_name: str, func_name: str, input: Any) -> AsyncGenerator[Any, None]:
        txid = self._send_type2("execute", "call_method", comp_name, func_name, input)

        async def generator():
            idx = 0
            finished = False
            while not finished:
                while idx not in self._exec_responses[txid]:
                    await self._listen()
                packet = self._exec_responses[txid].pop(idx)
                if not packet["body"]["status"]:
                    raise RuntimeError(packet["body"]["reason"])
                if packet["headers"][2]:
                    finished = True
                yield packet["body"]["out"]
                idx += 1
            del self._exec_responses[txid]

        return generator()
