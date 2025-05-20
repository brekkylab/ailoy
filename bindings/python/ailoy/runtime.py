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
    def __init__(self, address: str = "inproc://"):
        self.address: str = address
        self._responses: dict[str, Packet] = {}
        self._exec_responses: defaultdict[str, dict[int, Packet]] = defaultdict(dict)
        self._listen_lock: Optional[Event] = None

        start_threads(self.address)
        self._client: BrokerClient = BrokerClient(address)
        txid = self._send_type1("connect")
        if not txid:
            raise RuntimeError("Connection failed")
        self._sync_listen()
        if not self._responses[txid]["body"]["status"]:
            raise RuntimeError("Connection failed")
        del self._responses[txid]

    def __del__(self):
        self.stop()

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        self.stop()

    def stop(self):
        if self._client:
            txid = self._send_type1("disconnect")
            if not txid:
                raise RuntimeError("Disconnection failed")
            while txid not in self._responses:
                self._sync_listen()
            if not self._responses[txid]["body"]["status"]:
                raise RuntimeError("Disconnection failed")
            self._client = None
            stop_threads(self.address)

    def _send_type1(self, ptype: Literal["connect", "disconnect"]) -> Optional[str]:
        txid = generate_uuid()
        if self._client.send_type1(txid, ptype):
            return txid
        raise RuntimeError("Failed to send packet")

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
        if self._client.send_type2(txid, ptype, status, *args):
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
    def __init__(self, address: str = "inproc://"):
        super().__init__(address)

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
    def __init__(self, address: str = "inproc://"):
        super().__init__(address)

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
