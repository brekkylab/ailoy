import type { Sharp } from "sharp";

export async function sharpImageToBase64(image: Sharp) {
  const buffer = await image.toBuffer();
  const metadata = await image.metadata();
  metadata.format;

  const mimeTypeMap = {
    avif: "image/avif",
    dz: "application/zip", // DeepZoom — usually ZIP-based tileset
    exr: "image/aces", // OpenEXR format
    fits: "image/fits",
    gif: "image/gif",
    heif: "image/heif",
    input: "application/octet-stream", // generic input placeholder
    jpeg: "image/jpeg",
    jpg: "image/jpeg",
    jp2: "image/jp2",
    jxl: "image/jxl",
    magick: "image/magick", // general ImageMagick input — ambiguous
    openslide: "application/vnd.openslide", // whole-slide imaging
    pdf: "application/pdf",
    png: "image/png",
    ppm: "image/x-portable-pixmap",
    rad: "image/vnd.radiance",
    raw: "application/octet-stream",
    svg: "image/svg+xml",
    tiff: "image/tiff",
    tif: "image/tiff",
    v: "application/octet-stream", // ambiguous; possibly raw/video?
    webp: "image/webp",
  };

  const mimeType = mimeTypeMap[metadata.format] || "application/octet-stream";
  return `data:${mimeType};base64,${buffer.toString("base64")}`;
}
