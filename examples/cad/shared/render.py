"""Build a CadQuery model, check it, and draw it.

    python3 render.py MODEL.py --out /artifacts/NAME [--turntable] [--explode F] [--section AXIS[=V]]

MODEL.py is a CadQuery script, run as CQ-editor runs one: `cq` is imported, and each call of
`show_object(obj, name, options)` adds a part, with `options={"color": ...}`. A script that
calls it for none is taken for its `result`. A part is a Workplane, a Shape, an Assembly or a
list of them.

What is written under --out:

* `model.step` — every part, named and colored, as one STEP assembly;
* `model.glb` — the same as meshes, for a viewer;
* `parts/NAME.stl` — each part on its own, for a slicer;
* `views.png` — iso, front, right and top on one sheet, and each of them alone;
* `exploded.png`, `section.png`, `turntable.gif` — when asked for.

The report, on stdout as one JSON object, has each part's validity, solids, bounding box,
volume and area, and every pair of parts whose solids overlap, with the volume they share.
"""

import argparse
import json
import math
import runpy
import sys
import time
import traceback
from pathlib import Path

import cadquery as cq
import numpy as np
import trimesh
from OCP.BRep import BRep_Tool
from OCP.BRepMesh import BRepMesh_IncrementalMesh
from OCP.TopAbs import TopAbs_FACE, TopAbs_REVERSED
from OCP.TopExp import TopExp
from OCP.TopLoc import TopLoc_Location
from OCP.TopoDS import TopoDS
from OCP.TopTools import TopTools_IndexedMapOfShape
from PIL import Image, ImageColor, ImageDraw, ImageFont

# Colors for the parts that do not name one.
PALETTE = [
    "#4c78a8", "#f58518", "#54a24b", "#e45756", "#72b7b2",
    "#eeca3b", "#b279a2", "#ff9da6", "#9d755d", "#bab0ac",
]
BACKGROUND = "#f5f6f8"
EDGE = np.array([0.12, 0.13, 0.15, 1.0])

# Where each view looks from, and which way is up in it. Z is up in the model.
VIEWS = {
    "iso": ((1.0, -1.0, 0.8), (0, 0, 1)),
    "front": ((0.0, -1.0, 0.0), (0, 0, 1)),
    "right": ((1.0, 0.0, 0.0), (0, 0, 1)),
    "top": ((0.0, 0.0, 1.0), (0, 1, 0)),
}


class ModelError(Exception):
    pass


# ---- Loading the model ----------------------------------------------------------------------


def to_rgba(color, index):
    if color is None:
        color = PALETTE[index % len(PALETTE)]
    if isinstance(color, cq.Color):
        return tuple(color.toTuple())
    if isinstance(color, (tuple, list)):
        c = [float(x) for x in color]
        if max(c) > 1.0:
            c = [x / 255.0 for x in c]
        return tuple(c + [1.0] * (4 - len(c)))
    c = ImageColor.getrgb(color)
    return tuple(x / 255.0 for x in c) + ((1.0,) if len(c) == 3 else ())


def to_shapes(obj):
    """The shapes an object is made of, each with the color it carries, if any."""
    if isinstance(obj, (list, tuple)):
        return [s for o in obj for s in to_shapes(o)]
    if isinstance(obj, cq.Assembly):
        # Iterating an assembly yields each shape with its location and color from the root,
        # named by its path from the root, whose own name is a UUID unless it was given one.
        return [
            (shape.moved(loc), color.toTuple() if color else None, name.split("/", 1)[-1])
            for shape, name, loc, color in obj
        ]
    if isinstance(obj, cq.Workplane):
        shapes = [o for o in obj.vals() if isinstance(o, cq.Shape)]
        if not shapes:
            raise ModelError("a Workplane with no shapes on its stack")
        return [(cq.Compound.makeCompound(shapes) if len(shapes) > 1 else shapes[0], None, None)]
    if isinstance(obj, cq.Shape):
        return [(obj, None, None)]
    raise ModelError(f"cannot show a {type(obj).__name__}")


def load(model):
    shown = []

    def show_object(obj, name=None, options=None, **kwargs):
        options = dict(options or {}, **kwargs)
        shown.append((obj, name, options.get("color")))

    namespace = runpy.run_path(
        str(model),
        init_globals={"cq": cq, "show_object": show_object, "debug": lambda *a, **k: None},
        run_name="__main__",
    )
    if not shown:
        if "result" not in namespace:
            raise ModelError("the script neither called show_object nor set `result`")
        shown.append((namespace["result"], None, None))

    parts, names = [], set()
    for obj, name, color in shown:
        shapes = to_shapes(obj)
        for i, (shape, own_color, own_name) in enumerate(shapes):
            base = name or own_name or f"part{len(parts) + 1}"
            if len(shapes) > 1 and name:
                base = f"{name}-{own_name or i + 1}"
            label, n = base, 2
            while label in names:
                label, n = f"{base}-{n}", n + 1
            names.add(label)
            index = len(parts)
            parts.append({"name": label, "shape": shape, "rgba": to_rgba(color or own_color, index)})
    if not parts:
        raise ModelError("what was shown has no shapes in it")
    return parts


# ---- Checking it ----------------------------------------------------------------------------


def bbox_of(shape):
    b = shape.BoundingBox()
    return np.array([b.xmin, b.ymin, b.zmin]), np.array([b.xmax, b.ymax, b.zmax])


def describe(part):
    shape = part["shape"]
    lo, hi = bbox_of(shape)
    return {
        "name": part["name"],
        "valid": bool(shape.isValid()),
        "solids": len(shape.Solids()),
        "bbox": {"min": rounded(lo), "max": rounded(hi), "size": rounded(hi - lo)},
        "volume": round(shape.Volume(), 3),
        "area": round(shape.Area(), 3),
        "color": "#" + "".join(f"{round(x * 255):02x}" for x in part["rgba"][:3]),
        "triangles": len(part["mesh"].faces),
        "watertight": bool(part["mesh"].is_watertight),
    }


def interferences(parts, tolerance):
    """Every pair of parts whose solids share more than `tolerance` of volume."""
    found = []
    boxes = [bbox_of(p["shape"]) for p in parts]
    for i in range(len(parts)):
        for j in range(i + 1, len(parts)):
            (alo, ahi), (blo, bhi) = boxes[i], boxes[j]
            if np.any(ahi < blo) or np.any(bhi < alo):
                continue
            a, b = parts[i]["shape"], parts[j]["shape"]
            if not a.Solids() or not b.Solids():
                continue
            try:
                shared = a.intersect(b).Volume()
            except Exception as e:  # A boolean OCCT gives up on is worth saying, not fatal.
                found.append({"a": parts[i]["name"], "b": parts[j]["name"], "error": str(e)})
                continue
            if shared > tolerance:
                found.append({"a": parts[i]["name"], "b": parts[j]["name"], "volume": round(shared, 4)})
    return found


def rounded(v):
    return [round(float(x), 3) for x in v]


# ---- Meshing and exporting ------------------------------------------------------------------


def mesh_of(shape, tolerance, angular=0.2):
    """The shape as triangles, from OCCT's mesher. `Shape.tessellate` is the same, but slower
    by a hundred times on a shape of many faces."""
    BRepMesh_IncrementalMesh(shape.wrapped, tolerance, False, angular, True)
    faces = TopTools_IndexedMapOfShape()
    TopExp.MapShapes_s(shape.wrapped, TopAbs_FACE, faces)
    vertices, triangles, offset = [], [], 0
    for i in range(1, faces.Size() + 1):
        face = TopoDS.Face_s(faces.FindKey(i))
        loc = TopLoc_Location()
        poly = BRep_Tool.Triangulation_s(face, loc)
        if poly is None:
            continue
        trsf = loc.Transformation()
        n = poly.NbNodes()
        vertices.append(np.array([poly.Node(j).Transformed(trsf).Coord() for j in range(1, n + 1)]))
        tri = np.array([poly.Triangle(j).Get() for j in range(1, poly.NbTriangles() + 1)]) - 1
        if face.Orientation() == TopAbs_REVERSED:
            tri = tri[:, ::-1]
        triangles.append(tri + offset)
        offset += n
    if not triangles:
        return trimesh.Trimesh()
    mesh = trimesh.Trimesh(vertices=np.concatenate(vertices), faces=np.concatenate(triangles), process=True)
    # OCCT meshes each face on its own; merged, a closed solid's normals can be made to agree.
    trimesh.repair.fix_normals(mesh)
    return mesh


def export(parts, out):
    (out / "parts").mkdir(parents=True, exist_ok=True)
    assembly = cq.Assembly(name="model")
    scene = trimesh.Scene()
    for part in parts:
        assembly.add(part["shape"], name=part["name"], color=cq.Color(*part["rgba"]))
        mesh = part["mesh"].copy()
        mesh.visual.face_colors = (np.array(part["rgba"]) * 255).astype(np.uint8)
        scene.add_geometry(mesh, node_name=part["name"], geom_name=part["name"])
        part["mesh"].export(out / "parts" / f"{safe(part['name'])}.stl")
    assembly.export(str(out / "model.step"))
    scene.export(out / "model.glb")


def safe(name):
    return "".join(c if c.isalnum() or c in "-_." else "_" for c in name)


# ---- Drawing --------------------------------------------------------------------------------


def camera(direction, up):
    d = np.asarray(direction, float)
    d /= np.linalg.norm(d)
    right = np.cross(up, d)
    right /= np.linalg.norm(right)
    return d, right, np.cross(d, right)


def draw(meshes, direction, up, size, path, frame=None, title=None):
    """Draw meshes, orthographic, from `direction`, into a PNG of `size` pixels a side.

    `meshes` is a list of (Trimesh, rgba), rgba one color or one for each face. `frame` is the
    (center, half-width) of the drawing in view coordinates, fixed so the frames of a turntable
    do not jump; by default it fits.
    """
    d, right, upv = camera(direction, up)
    key = normalize(d + 0.6 * upv - 0.5 * right)
    fill = normalize(d - 0.8 * upv + 0.7 * right)
    half_vector = normalize(key + d)

    tris, colors, segments = [], [], []
    for mesh, rgba in meshes:
        if len(mesh.faces) == 0:
            continue
        v = mesh.vertices
        view = np.stack([v @ right, v @ upv, v @ d], axis=1)
        normals = mesh.face_normals
        # A closed mesh's normals point out; an open one's may not, so light either side.
        n = normals if mesh.is_watertight else normals * np.sign(normals @ d + 1e-12)[:, None]
        light = 0.30 + 0.62 * np.clip(n @ key, 0, None) + 0.18 * np.clip(n @ fill, 0, None)
        shine = 0.22 * np.clip(n @ half_vector, 0, None) ** 40
        base = np.broadcast_to(np.asarray(rgba, float)[..., :3], (len(normals), 3))
        color = np.clip(base * light[:, None] + shine[:, None], 0, 1)
        tris.append(view[mesh.faces])
        colors.append(color)

        # Creases, where faces turn more than 25°, and outlines, where one face of an edge
        # looks at the camera and the other away.
        if len(mesh.face_adjacency):
            facing = (normals @ d) > 0
            fa = mesh.face_adjacency
            keep = (mesh.face_adjacency_angles > math.radians(25)) | (facing[fa[:, 0]] != facing[fa[:, 1]])
            segments.append(view[mesh.face_adjacency_edges[keep]])

    # Twice the size, and scaled down at the end, for smooth edges.
    ss = 2
    px = size * ss
    image = np.empty((px, px, 3))
    image[:] = np.array(ImageColor.getrgb(BACKGROUND)) / 255.0
    if tris:
        tris = np.concatenate(tris)
        colors = np.concatenate(colors)
        if frame is None:
            xy = tris[:, :, :2].reshape(-1, 2)
            lo, hi = xy.min(axis=0), xy.max(axis=0)
            frame = ((lo + hi) / 2, max(hi - lo) / 2 * 1.08 + 1e-9)
        center, half = frame
        scale = px / (2 * half)

        def to_pixels(p):
            out = p.copy()
            out[..., 0] = (p[..., 0] - center[0]) * scale + px / 2
            out[..., 1] = px / 2 - (p[..., 1] - center[1]) * scale
            return out

        tris = to_pixels(tris)
        zbuf = np.full((px, px), -np.inf)
        rasterize(tris, colors, zbuf, image)

        if segments:
            segments = to_pixels(np.concatenate(segments))
            # How far behind the surface a line may be and still be drawn, in model units.
            slack = 2.0 / scale
            lines(segments, zbuf, image, slack, EDGE[:3], width=ss)

    picture = Image.fromarray((image * 255).astype(np.uint8)).resize((size, size), Image.LANCZOS)
    if title:
        pen = ImageDraw.Draw(picture)
        pen.text((size * 0.02, size * 0.02), title, fill="#333a44", font=font(max(12, size // 55)))
    picture.save(path)


def rasterize(tris, colors, zbuf, image):
    """Fill each triangle, flat, where it is nearer than what is already drawn."""
    h, w = zbuf.shape
    for (a, b, c), color in zip(tris, colors):
        x0, x1 = int(max(min(a[0], b[0], c[0]), 0)), int(min(max(a[0], b[0], c[0]) + 1, w))
        y0, y1 = int(max(min(a[1], b[1], c[1]), 0)), int(min(max(a[1], b[1], c[1]) + 1, h))
        if x0 >= x1 or y0 >= y1:
            continue
        area = (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
        if abs(area) < 1e-12:
            continue
        ys, xs = np.mgrid[y0:y1, x0:x1]
        xs = xs + 0.5
        ys = ys + 0.5
        w0 = ((b[0] - xs) * (c[1] - ys) - (b[1] - ys) * (c[0] - xs)) / area
        w1 = ((c[0] - xs) * (a[1] - ys) - (c[1] - ys) * (a[0] - xs)) / area
        w2 = 1 - w0 - w1
        inside = (w0 >= -1e-9) & (w1 >= -1e-9) & (w2 >= -1e-9)
        if not inside.any():
            continue
        z = w0 * a[2] + w1 * b[2] + w2 * c[2]
        region = zbuf[y0:y1, x0:x1]
        nearer = inside & (z > region)
        region[nearer] = z[nearer]
        image[y0:y1, x0:x1][nearer] = color


def lines(segments, zbuf, image, slack, color, width):
    """Draw segments where they are not behind the surface drawn there."""
    h, w = zbuf.shape
    length = np.linalg.norm(segments[:, 1, :2] - segments[:, 0, :2], axis=1)
    steps = np.maximum(np.ceil(length), 1).astype(int)
    which = np.repeat(np.arange(len(segments)), steps + 1)
    t = np.concatenate([np.linspace(0, 1, s + 1) for s in steps])
    p = segments[which, 0] + (segments[which, 1] - segments[which, 0]) * t[:, None]
    for dx in range(width):
        for dy in range(width):
            x = np.floor(p[:, 0]).astype(int) + dx - width // 2
            y = np.floor(p[:, 1]).astype(int) + dy - width // 2
            ok = (x >= 0) & (x < w) & (y >= 0) & (y < h)
            x, y, z = x[ok], y[ok], p[ok, 2]
            seen = z >= zbuf[y, x] - slack
            image[y[seen], x[seen]] = color


def font(size):
    try:
        return ImageFont.load_default(size)
    except TypeError:  # Pillow before 10.1 has one size.
        return ImageFont.load_default()


def normalize(v):
    return v / np.linalg.norm(v)


def sheet(paths, out):
    """The views two by two on one image, which is one read for the agent instead of four."""
    images = [Image.open(p).convert("RGB") for p in paths]
    w, h = images[0].size
    canvas = Image.new("RGB", (w * 2, h * 2), BACKGROUND)
    pen = ImageDraw.Draw(canvas)
    for i, image in enumerate(images):
        x, y = (i % 2) * w, (i // 2) * h
        canvas.paste(image, (x, y))
        pen.rectangle([x, y, x + w - 1, y + h - 1], outline="#d0d4da")
    canvas.save(out)


def explode(parts, factor):
    """Each part moved away from the middle of the model along the line through its own."""
    centers = [np.array(p["mesh"].bounds).mean(axis=0) for p in parts if len(p["mesh"].faces)]
    middle = np.mean(centers, axis=0) if centers else np.zeros(3)
    moved = []
    for p in parts:
        mesh = p["mesh"].copy()
        if len(mesh.faces):
            offset = (np.array(mesh.bounds).mean(axis=0) - middle) * (factor - 1.0)
            mesh.apply_translation(offset)
        moved.append((mesh, p["rgba"]))
    return moved


def section(parts, spec, tolerance):
    """Each part cut by a plane across `axis`, keeping the side toward the negative axis."""
    axis, _, value = spec.partition("=")
    axis = axis.strip().lower()
    if axis not in "xyz" or len(axis) != 1:
        raise ModelError(f"--section takes x, y or z, and optionally =VALUE, not {spec!r}")
    k = "xyz".index(axis)
    los, his = zip(*(bbox_of(p["shape"]) for p in parts))
    lo, hi = np.min(los, axis=0), np.max(his, axis=0)
    at = float(value) if value else (lo[k] + hi[k]) / 2
    span = float(np.max(hi - lo)) * 4 + 10
    corner = (lo + hi) / 2 - span / 2
    corner[k] = at - span
    cutter = cq.Solid.makeBox(span, span, span, pnt=cq.Vector(*corner))
    cut = []
    for p in parts:
        mesh = mesh_of(p["shape"].intersect(cutter), tolerance)
        colors = np.tile(np.array(p["rgba"]), (len(mesh.faces), 1))
        if len(mesh.faces):
            # The faces the plane made, lighter, so the cut shows as a cut.
            on_plane = (np.abs(mesh.triangles_center[:, k] - at) < tolerance * 2) & (
                np.abs(mesh.face_normals[:, k]) > 0.99
            )
            colors[on_plane, :3] = colors[on_plane, :3] * 0.45 + 0.55
        cut.append((mesh, colors))
    # Look at the cut face, from the side that was removed.
    direction = np.array([0.35, -0.35, 0.3])
    direction[k] = 1.0
    up = (0, 1, 0) if axis == "z" else (0, 0, 1)
    return cut, direction, up, axis, at


def turntable(meshes, size, path, frames):
    elevation = 0.55
    everything = trimesh.util.concatenate([m for m, _ in meshes if len(m.faces)])
    center = everything.bounds.mean(axis=0)
    radius = float(np.linalg.norm(everything.vertices - center, axis=1).max()) * 1.05
    shifted = []
    for mesh, rgba in meshes:
        m = mesh.copy()
        m.apply_translation(-center)
        shifted.append((m, rgba))
    tmp = path.parent / ".frames"
    tmp.mkdir(exist_ok=True)
    images = []
    for i in range(frames):
        a = 2 * math.pi * i / frames - math.pi / 4
        direction = (math.cos(a), math.sin(a), elevation)
        frame_path = tmp / f"{i:03d}.png"
        draw(shifted, direction, (0, 0, 1), size, frame_path, frame=(np.zeros(2), radius))
        images.append(Image.open(frame_path).convert("P", palette=Image.ADAPTIVE, colors=255))
    images[0].save(path, save_all=True, append_images=images[1:], duration=80, loop=0, disposal=2)
    for f in tmp.iterdir():
        f.unlink()
    tmp.rmdir()


# ---- Main -----------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("model", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--size", type=int, default=900, help="pixels a side of each view")
    parser.add_argument("--explode", type=float, help="draw the parts this many times as far apart")
    parser.add_argument("--section", help="draw the model cut across x, y or z, at =VALUE or the middle")
    parser.add_argument("--turntable", action="store_true", help="write a turning GIF")
    parser.add_argument("--frames", type=int, default=36)
    parser.add_argument("--no-export", action="store_true", help="draw only, for a quick look")
    parser.add_argument("--interference", type=float, default=1e-3,
                        help="the shared volume, in mm³, above which two parts are said to overlap")
    args = parser.parse_args()

    started = time.monotonic()
    timings = {}

    def lap(name):
        nonlocal started
        now = time.monotonic()
        timings[name] = round(now - started, 2)
        started = now

    try:
        parts = load(args.model)
    except Exception:
        print(json.dumps({"error": "the model script failed", "traceback": traceback.format_exc(limit=-6)}))
        sys.exit(1)
    lap("build")

    args.out.mkdir(parents=True, exist_ok=True)
    los, his = zip(*(bbox_of(p["shape"]) for p in parts))
    lo, hi = np.min(los, axis=0), np.max(his, axis=0)
    tolerance = max(float(np.linalg.norm(hi - lo)) * 5e-4, 0.005)
    for p in parts:
        p["mesh"] = mesh_of(p["shape"], tolerance)
    lap("mesh")

    report = {
        "parts": [describe(p) for p in parts],
        "bbox": {"min": rounded(lo), "max": rounded(hi), "size": rounded(hi - lo)},
        "units": "mm",
    }
    report["interferences"] = interferences(parts, args.interference)
    lap("check")

    files = []
    if not args.no_export:
        export(parts, args.out)
        files += ["model.step", "model.glb"] + [f"parts/{safe(p['name'])}.stl" for p in parts]
        lap("export")

    meshes = [(p["mesh"], p["rgba"]) for p in parts]
    size = hi - lo
    view_paths = []
    for name, (direction, up) in VIEWS.items():
        path = args.out / f"{name}.png"
        # `x`, not `×`: Pillow's own font has no glyph for it.
        dims = {"iso": "", "front": f"  {size[0]:.1f} x {size[2]:.1f}",
                "right": f"  {size[1]:.1f} x {size[2]:.1f}", "top": f"  {size[0]:.1f} x {size[1]:.1f}"}[name]
        draw(meshes, direction, up, args.size, path, title=f"{name}{dims} mm" if dims else name)
        view_paths.append(path)
        files.append(path.name)
    sheet(view_paths, args.out / "views.png")
    files.append("views.png")
    lap("views")

    if args.explode:
        draw(explode(parts, args.explode), VIEWS["iso"][0], VIEWS["iso"][1], args.size,
             args.out / "exploded.png", title=f"exploded x{args.explode:g}")
        files.append("exploded.png")
        lap("exploded")

    if args.section:
        try:
            cut, direction, up, axis, at = section(parts, args.section, tolerance)
        except ModelError as e:
            print(json.dumps({"error": str(e)}))
            sys.exit(2)
        draw(cut, direction, up, args.size, args.out / "section.png", title=f"section {axis} = {at:.2f} mm")
        files.append("section.png")
        lap("section")

    if args.turntable:
        turntable(meshes, min(args.size, 600), args.out / "turntable.gif", args.frames)
        files.append("turntable.gif")
        lap("turntable")

    report["out"] = str(args.out)
    report["files"] = files
    report["seconds"] = timings
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
