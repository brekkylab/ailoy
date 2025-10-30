from typing import Any, Literal, Optional, TypedDict

class Packet(TypedDict):
    packet_type: Literal["respond", "respond_execute"]
    instruction_type: Optional[Literal["call_function", "define_component", "delete_component", "call_method"]]
    headers: list[bool | int | str]
    body: dict[str, Any]

def start_threads(address: str) -> None: ...
def stop_threads(address: str) -> None: ...
def generate_uuid() -> str: ...

class BrokerClient:
    def send_type1(self, txid: str, ptype: Literal["connect", "disconnect"]) -> bool: ...
    def send_type2(
        self,
        txid: str,
        ptype: Literal["subscribe", "unsubscribe", "execute"],
        itype: Literal["call_function", "define_component", "delete_component", "call_method"],
        *args,
    ) -> bool: ...
    def send_type3(
        self,
        txid: str,
        ptype: Literal["respond", "respond_execute"],
        status: bool,
        *args,
    ) -> bool: ...
    def listen() -> Optional[Packet]: ...
