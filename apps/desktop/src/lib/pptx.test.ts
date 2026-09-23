import { describe, expect, it, vi } from "vitest";

import { attachMedia } from "@/lib/pptx";

const picture = (imagePath: string) => ({ type: "picture", imagePath });

describe("attachMedia", () => {
  it("gives every picture its data", async () => {
    const a = picture("ppt/media/image1.png");
    const b = picture("ppt/media/image2.png");
    const report = await attachMedia([{ elements: [a, b] }], (p) => Promise.resolve(`blob:${p}`));
    expect(report).toEqual({ filled: 2, missing: 0 });
    expect(a).toMatchObject({ imageData: "blob:ppt/media/image1.png" });
    expect(b).toMatchObject({ imageData: "blob:ppt/media/image2.png" });
  });

  it("reaches pictures inside a group", async () => {
    // A group's members are elements in their own right, and a deck puts most of its
    // logos in one.
    const inner = picture("ppt/media/logo.png");
    const group = { type: "group", children: [{ type: "group", children: [inner] }] };
    const report = await attachMedia([{ elements: [group] }], () => Promise.resolve("blob:x"));
    expect(report.filled).toBe(1);
    expect(inner).toMatchObject({ imageData: "blob:x" });
  });

  it("resolves a repeated path once, however many slides use it", async () => {
    // A logo on every slide would otherwise be read out of the archive every time.
    const resolve = vi.fn().mockResolvedValue("blob:x");
    const slides = [1, 2, 3].map(() => ({ elements: [picture("ppt/media/logo.png")] }));
    const report = await attachMedia(slides, resolve);
    expect(resolve).toHaveBeenCalledTimes(1);
    expect(report.filled).toBe(3);
  });

  it("counts a picture the archive has nothing for, and leaves it alone", async () => {
    // One unreadable picture costs its own box, not the deck.
    const gone = picture("ppt/media/missing.png");
    const report = await attachMedia([{ elements: [gone] }], () => Promise.resolve(undefined));
    expect(report).toEqual({ filled: 0, missing: 1 });
    expect(gone).not.toHaveProperty("imageData");
  });

  it("survives a lookup that throws", async () => {
    const report = await attachMedia([{ elements: [picture("a.png")] }], () =>
      Promise.reject(new Error("archive closed")),
    );
    expect(report).toEqual({ filled: 0, missing: 1 });
  });

  it("leaves data that is already there", async () => {
    const already = { type: "picture", imagePath: "a.png", imageData: "blob:kept" };
    const resolve = vi.fn().mockResolvedValue("blob:new");
    const report = await attachMedia([{ elements: [already] }], resolve);
    expect(resolve).not.toHaveBeenCalled();
    expect(already.imageData).toBe("blob:kept");
    expect(report).toEqual({ filled: 0, missing: 0 });
  });

  it("walks past everything that is not a picture", async () => {
    const elements = [{ type: "text" }, null, "stray", { type: "shape", imagePath: "" }];
    const report = await attachMedia([{ elements }], () => Promise.resolve("blob:x"));
    expect(report).toEqual({ filled: 0, missing: 0 });
  });
})
