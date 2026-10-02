# Building VM

The computer an agent works on is a Linux VM run by [virtx](https://github.com/brekkylab/virtx). Ailoy ships virtx with it, so there is nothing else to install:

- **Python:** `ailoy.virtx`
- **Node.js:** exported from `ailoy-node`
- **Rust:** the `virtx` crate (`ailoy::console::ConsoleClient` is the same type as `virtx::console::ConsoleClient`)

In Python and Node, use the classes from `ailoy`, not from the standalone `virtx` package: a `ConsoleClient` built by `virtx` cannot be handed to an `AgentBuilder`.

Two clients do the work:

- `ImageClient` builds the VM's disk image from a `Recipe` and manages the built images.
- `ConsoleClient` boots a VM from an image and runs commands in it. This is what the agent gets.

## Images

A `Recipe` is a base image plus steps, much like a Dockerfile. `ImageClient` builds it and stores it under a name, so a console can boot from it later without building it again.

::: code-group

```python [Python]
import asyncio

from ailoy.virtx import ImageClient, ImageSource, Recipe, ensure_virtx


async def main() -> None:
    # Fetches virtx's server into its cache the first time; a no-op after.
    await ensure_virtx()

    async with await ImageClient.try_new() as images:
        recipe = Recipe("python:3.12-slim-trixie").step("pip install matplotlib")
        built = await images.build(recipe, "plot:latest")
        print(built.reference, built.digest)

        for image in await images.list():
            print(image.digest, image.refs)

        # Remove it by its reference (or ImageSource.digest(...) for its digest).
        await images.remove(ImageSource.reference("plot:latest"))


asyncio.run(main())
```

```js [Node.js]
const { ImageClient, ImageSource, Recipe, ensureVirtx } = require('ailoy-node')

// Fetches virtx's server into its cache the first time; a no-op after.
await ensureVirtx()

const images = await ImageClient.tryNew()
try {
  const recipe = new Recipe('python:3.12-slim-trixie').step('pip install matplotlib')
  const built = await images.build(recipe, 'plot:latest')
  console.log(built.reference, built.digest)

  for (const image of await images.list()) {
    console.log(image.digest, image.refs)
  }

  // Remove it by its reference (or ImageSource.digest(...) for its digest).
  await images.remove(ImageSource.reference('plot:latest'))
} finally {
  await images.close()
}
```

```rust [Rust]
use virtx::image::{ImageClient, ImageSource, Recipe};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Fetches virtx's server into its cache the first time; a no-op after.
    virtx::ensure_virtx().await?;

    let mut images = ImageClient::try_new().await?;
    let recipe = Recipe::new("python:3.12-slim-trixie").step("pip install matplotlib");
    let built = images.build(recipe, Some("plot:latest")).await?;
    println!("{} {}", built.reference, built.digest);

    for image in images.list().await? {
        println!("{} {:?}", image.digest, image.refs);
    }

    // Remove it by its reference (or ImageSource::digest(..) for its digest).
    images.remove(ImageSource::reference("plot:latest")).await?;
    Ok(())
}
```

:::

You don't have to build ahead of time: passing a `Recipe` straight to a console builds it on first use.

## Consoles

`ConsoleClient.builder()` sets up the VM: the image to boot, host directories to mount, network access, CPUs, memory and GPU. `build()` boots it, and `exec` runs a command in it.

::: code-group

```python [Python]
import asyncio

from ailoy.virtx import ConsoleClient, ImageSource


async def main() -> None:
    console = await (
        ConsoleClient.builder()
        .image(ImageSource.reference("plot:latest"))
        .mount("./artifacts", "/artifacts")
        .network(True)
        .build()
    )

    async with console:
        result = await console.exec(["python", "-c", "import matplotlib; print(matplotlib.__version__)"])

    print(result.stdout.decode(), end="")


asyncio.run(main())
```

```js [Node.js]
const { ConsoleClient, ImageSource } = require('ailoy-node')

const console_ = await ConsoleClient.builder()
  .image(ImageSource.reference('plot:latest'))
  .mount('./artifacts', '/artifacts')
  .network(true)
  .build()

try {
  const result = await console_.exec(['python', '-c', 'import matplotlib; print(matplotlib.__version__)'])
  process.stdout.write(result.stdout)
} finally {
  await console_.close()
}
```

```rust [Rust]
use virtx::{console::ConsoleClient, image::ImageSource};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let mut console = ConsoleClient::builder()
        .image(ImageSource::reference("plot:latest"))
        .mount(std::path::absolute("./artifacts")?, "/artifacts")
        .network(true)
        .build()
        .await?;

    let result = console
        .exec(["python", "-c", "import matplotlib; print(matplotlib.__version__)"], None)
        .await?;
    print!("{}", String::from_utf8_lossy(&result.stdout));
    Ok(())
}
```

:::

::: tip
An image built from a `Recipe` passed straight to `.image(...)` is removed when the console closes.
:::

Besides `exec`, a console can `read` and `write` files in the VM.

For more information, see the [virtx README](https://github.com/brekkylab/virtx).
