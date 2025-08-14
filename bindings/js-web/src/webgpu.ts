import { FeatureSupportError, WebGPUNotAvailableError } from "./llm/error";
import { detectGPUDevice } from "./tvmjs";

export async function isWebGPUSupported(
  powerPreference: "low-power" | "high-performance" = "high-performance",
  requiredFeatures: GPUFeatureName[] = ["shader-f16"]
): Promise<{
  supported: boolean;
  reason?: string;
}> {
  try {
    if (typeof navigator === "undefined" || navigator.gpu === undefined) {
      throw new WebGPUNotAvailableError();
    }

    const adapter = await navigator.gpu.requestAdapter({ powerPreference });
    if (adapter == null) {
      throw Error(
        "Unable to find a compatible GPU. This issue might be because your computer doesn't have a GPU, or your system settings are not configured properly. " +
          "Please check if your device has a GPU properly set up and if your your browser supports WebGPU. " +
          "You can also consult your browser's compatibility chart to see if it supports WebGPU. " +
          "For more information about WebGPU support in your browser, visit https://webgpureport.org/"
      );
    }
    const computeMB = (value: number) => {
      return Math.ceil(value / (1 << 20)) + "MB";
    };

    // more detailed error message
    let requiredMaxBufferSize = 1 << 30; // 1GB
    if (requiredMaxBufferSize > adapter.limits.maxBufferSize) {
      // If 1GB is too large, try 256MB (default size stated in WebGPU doc)
      const backupRequiredMaxBufferSize = 1 << 28; // 256MB
      console.log(
        `Requested maxBufferSize exceeds limit. \n` +
          `requested=${computeMB(requiredMaxBufferSize)}, \n` +
          `limit=${computeMB(adapter.limits.maxBufferSize)}. \n` +
          `WARNING: Falling back to ${computeMB(
            backupRequiredMaxBufferSize
          )}...`
      );
      requiredMaxBufferSize = backupRequiredMaxBufferSize;
      if (backupRequiredMaxBufferSize > adapter.limits.maxBufferSize) {
        // Fail if 256MB is still too big
        throw Error(
          `Cannot initialize runtime because of requested maxBufferSize ` +
            `exceeds limit. requested=${computeMB(
              backupRequiredMaxBufferSize
            )}, ` +
            `limit=${computeMB(adapter.limits.maxBufferSize)}. ` +
            `Consider upgrading your browser.`
        );
      }
    }

    let requiredMaxStorageBufferBindingSize = 1 << 30; // 1GB
    if (
      requiredMaxStorageBufferBindingSize >
      adapter.limits.maxStorageBufferBindingSize
    ) {
      // If 1GB is too large, try 128MB (default size stated in WebGPU doc)
      const backupRequiredMaxStorageBufferBindingSize = 1 << 27; // 128MB
      console.log(
        `Requested maxStorageBufferBindingSize exceeds limit. \n` +
          `requested=${computeMB(requiredMaxStorageBufferBindingSize)}, \n` +
          `limit=${computeMB(adapter.limits.maxStorageBufferBindingSize)}. \n` +
          `WARNING: Falling back to ${computeMB(
            backupRequiredMaxStorageBufferBindingSize
          )}...`
      );
      requiredMaxStorageBufferBindingSize =
        backupRequiredMaxStorageBufferBindingSize;
      if (
        backupRequiredMaxStorageBufferBindingSize >
        adapter.limits.maxStorageBufferBindingSize
      ) {
        // Fail if 128MB is still too big
        throw Error(
          `Cannot initialize runtime because of requested maxStorageBufferBindingSize ` +
            `exceeds limit. requested=${computeMB(
              backupRequiredMaxStorageBufferBindingSize
            )}, ` +
            `limit=${computeMB(adapter.limits.maxStorageBufferBindingSize)}. `
        );
      }
    }

    const requiredMaxComputeWorkgroupStorageSize = 32 << 10;
    if (
      requiredMaxComputeWorkgroupStorageSize >
      adapter.limits.maxComputeWorkgroupStorageSize
    ) {
      throw Error(
        `Cannot initialize runtime because of requested maxComputeWorkgroupStorageSize ` +
          `exceeds limit. requested=${requiredMaxComputeWorkgroupStorageSize}, ` +
          `limit=${adapter.limits.maxComputeWorkgroupStorageSize}. `
      );
    }

    const requiredMaxStorageBuffersPerShaderStage = 10; // default is 8
    if (
      requiredMaxStorageBuffersPerShaderStage >
      adapter.limits.maxStorageBuffersPerShaderStage
    ) {
      throw Error(
        `Cannot initialize runtime because of requested maxStorageBuffersPerShaderStage ` +
          `exceeds limit. requested=${requiredMaxStorageBuffersPerShaderStage}, ` +
          `limit=${adapter.limits.maxStorageBuffersPerShaderStage}. `
      );
    }

    for (const feature of requiredFeatures) {
      if (!adapter.features.has(feature)) {
        throw new FeatureSupportError(feature);
      }
    }

    return {
      supported: true,
    };
  } catch (err) {
    return {
      supported: false,
      reason: (err as Error).message,
    };
  }
}

export async function getGPUDevice(
  requiredFeatures: GPUFeatureName[] = ["shader-f16"]
) {
  const gpuDetectOutput = await detectGPUDevice();
  if (gpuDetectOutput == undefined) {
    throw new WebGPUNotAvailableError();
  }
  for (const feature of requiredFeatures) {
    if (!gpuDetectOutput.device.features.has(feature)) {
      throw new FeatureSupportError(feature);
    }
  }
  return gpuDetectOutput.device;
}
