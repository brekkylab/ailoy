from enum import Enum
from typing import Annotated

import typer
from pydantic import BaseModel
from rich.console import Console
from rich.table import Table

from ailoy.runtime import Runtime

app = typer.Typer(no_args_is_help=True)


def _create_runtime(ctx: typer.Context) -> Runtime:
    runtime = Runtime()
    ctx.call_on_close(runtime.stop)
    return runtime


def _humanize_bytes(num: float, suffix="B"):
    for unit in ("", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}Yi{suffix}"


class ListModelsResponse(BaseModel):
    class Item(BaseModel):
        type: str
        model_id: str
        attributes: dict[str, str]
        model_path: str
        total_bytes: int

    results: list[Item]


@app.command("list")
def list_local_models(ctx: typer.Context):
    rt = _create_runtime(ctx)
    resp = rt.call("list_local_models", {})

    table = Table(title="List models in local cache", title_justify="center")
    table.add_column("Type", justify="center")
    table.add_column("Model ID")
    table.add_column("Attributes")
    table.add_column("Path", overflow="fold")
    table.add_column("Size", justify="center")

    resp = ListModelsResponse.model_validate(resp)
    models = sorted(resp.results, key=lambda x: x.model_id)

    for model in models:
        attributes = [f"{key}: {val}" for key, val in model.attributes.items()]
        attributes_str = "\n".join(attributes)

        model_size = _humanize_bytes(model.total_bytes)

        table.add_row(model.type, model.model_id, attributes_str, model.model_path, model_size)

    console = Console()
    console.print(table)


class Quantization(str, Enum):
    q4f16_1 = "q4f16_1"


class Device(str, Enum):
    metal = "metal"
    vulkan = "vulkan"


class DownloadModelResponse(BaseModel):
    model_path: str


@app.command("download")
def download_models(
    ctx: typer.Context,
    model_id: Annotated[str, typer.Argument(help="model id")],
    quantization: Annotated[Quantization, typer.Option("-q", "--quantization", help="quantization method")],
    device: Annotated[Device, typer.Option("-d", "--device", help="target device")],
):
    rt = _create_runtime(ctx)
    resp = DownloadModelResponse.model_validate(
        rt.call(
            "download_model",
            {
                "model_id": model_id,
                "quantization": quantization,
                "device": device,
            },
        )
    )

    print(f"Successfully downloaded {model_id} in {resp.model_path}")


@app.command("remove")
def remove_model(
    ctx: typer.Context,
    model_id: Annotated[str, typer.Argument(help="model id")],
):
    rt = _create_runtime(ctx)
    resp = rt.call("remove_model", {"model_id": model_id})

    if resp["skipped"]:
        print(f"Skipped removing {model_id}")
    else:
        print(f"Successfully removed {model_id} in {resp['model_path']}")
