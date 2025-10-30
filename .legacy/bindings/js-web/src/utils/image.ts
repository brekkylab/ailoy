import { Image } from "wasm-vips";

import { uint8ArrayToBase64 } from "./base64";

export function isVipsImage(obj: any): obj is Image {
  return (
    obj != null &&
    typeof obj === "object" &&
    typeof obj.width === "number" &&
    typeof obj.height === "number" &&
    typeof obj.bands === "number" &&
    typeof obj.delete === "function" &&
    typeof obj.writeToBuffer === "function" &&
    obj.constructor &&
    obj.constructor.name === "Image"
  );
}

export function vipsImageToBase64(image: Image) {
  try {
    const pngBuffer = image.writeToBuffer(".png");
    const base64String = uint8ArrayToBase64(pngBuffer);
    return `data:image/png;base64,${base64String}`;
  } catch (err) {
    throw new Error(
      `Failed to convert vips image to base64 PNG: ${(err as Error).message}`
    );
  }
}
