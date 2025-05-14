import pytest

from ailoy import Runtime

pytestmark = [pytest.mark.runtime]


@pytest.fixture(scope="module")
def sync_runtime():
    print("creating sync runtime")
    rt = Runtime("inproc://sync")
    yield rt
    rt.close()


def test_echo(sync_runtime: Runtime):
    sync_runtime.call("echo", "hello world") == "hello world"


def test_spell(sync_runtime: Runtime):
    for i, out in enumerate(sync_runtime.call_iter("spell", "abcdefghijk")):
        assert out == "abcdefghijk"[i]
