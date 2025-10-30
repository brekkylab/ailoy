import Table from "cli-table3";
import { Command, Option } from "commander";

import { startRuntime } from "../runtime";

function humanizeBytes(bytes: number): string {
  for (const unit of ["B", "KiB", "MiB", "GiB", "TiB", "PiB", "EiB", "ZiB"]) {
    if (bytes < 1024) {
      return `${bytes.toFixed(1)}${unit}`;
    }
    bytes /= 1024.0;
  }
  return `${bytes.toFixed(1)}YiB`;
}

interface ListModelsResponse {
  results: Array<{
    type: string;
    model_id: string;
    attributes: Record<string, string>;
    model_path: string;
    total_bytes: number;
  }>;
}

interface DownloadModelResponse {
  model_path: string;
}

interface RemoveModelResponse {
  skipped: boolean;
  model_path: string;
}

const command = new Command("model")
  .description("Manage Ailoy models")
  .addCommand(
    new Command("list").action(async () => {
      const runtime = await startRuntime();
      try {
        const resp = (await runtime.call(
          "list_local_models",
          {}
        )) as ListModelsResponse;

        const table = new Table({
          head: ["Type", "Model ID", "Attributes", "Path", "Size"],
          wordWrap: true,
        });
        for (const m of resp.results.sort((a, b) =>
          a.model_id.localeCompare(b.model_id)
        )) {
          const attrs = Object.entries(m.attributes)
            .map(([k, v]) => `${k}: ${v}`)
            .join("\n");
          table.push([
            m.type,
            m.model_id,
            attrs,
            m.model_path,
            humanizeBytes(m.total_bytes),
          ]);
        }

        console.log(table.toString());
      } catch (err) {
        console.error(err);
      } finally {
        await runtime.stop();
      }
    })
  )
  .addCommand(
    new Command("download")
      .argument("<model_id>")
      .addOption(
        new Option("-q, --quantization <quantization>").choices(["q4f16_1"])
      )
      .addOption(
        new Option("-d, --device <device>").choices(["metal", "vulkan"])
      )
      .action(async (model_id, options) => {
        const runtime = await startRuntime();
        try {
          const resp = (await runtime.call("download_model", {
            model_id,
            quantization: options.quantization,
            device: options.device,
          })) as DownloadModelResponse;

          console.log(
            `Successfully downloaded ${model_id} to ${resp.model_path}`
          );
        } catch (err) {
          console.error(err);
        } finally {
          await runtime.stop();
        }
      })
  )
  .addCommand(
    new Command("remove").argument("<model_id>").action(async (model_id) => {
      const runtime = await startRuntime();
      try {
        const resp = (await runtime.call("remove_model", {
          model_id,
        })) as RemoveModelResponse;
        if (resp.skipped) {
          console.log(`Skipped removing ${model_id}`);
        } else {
          console.log(
            `Successfully removed ${model_id} from ${resp.model_path}`
          );
        }
      } catch (err) {
        console.log("error raised: ", err);
        console.error(err);
      } finally {
        await runtime.stop();
      }
    })
  );

export default command;
