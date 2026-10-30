"""Run the code the agent sends in this Blender, on its main thread, while the person watches.

    blender --gpu-backend vulkan -P server.py

It listens on 127.0.0.1:8765. A request is one connection: `run.py` writes `{"code": ...}`
on one line, and reads back `{"ok": ..., "output": ...}` until the connection closes. Not a
half-close for the end of the request: a published port carries it as a close. Every request runs in one
namespace, so what one defines the next can use, and it starts with `bpy`, `Vector` and the
helpers below: `look`, `views`, `report`, `save` and `turntable`.
"""

import contextlib
import io
import json
import math
import os
import socket
import tempfile
import traceback

import bpy
import gpu
import numpy as np
from mathutils import Matrix, Vector

ADDRESS = ("127.0.0.1", 8765)

# Where each of `views` looks from, and which way is up in it. Z is up in the scene.
VIEWS = {
    "iso": ((1.0, -1.0, 0.8), (0, 0, 1)),
    "front": ((0.0, -1.0, 0.0), (0, 0, 1)),
    "right": ((1.0, 0.0, 0.0), (0, 0, 1)),
    "top": ((0.0, 0.0, 1.0), (0, 1, 0)),
}


def _bounds():
    """The world-space box around every visible mesh, curve and text, or a unit box if none."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    corners = [
        obj.matrix_world @ Vector(corner)
        for obj in bpy.context.scene.objects
        if obj.type in {"MESH", "CURVE", "SURFACE", "META", "FONT"} and obj.visible_get()
        for corner in obj.evaluated_get(depsgraph).bound_box
    ]
    if not corners:
        return Vector((-1, -1, -1)), Vector((1, 1, 1))
    lo = Vector(tuple(min(c[i] for c in corners) for i in range(3)))
    hi = Vector(tuple(max(c[i] for c in corners) for i in range(3)))
    return lo, hi


@contextlib.contextmanager
def _render_state(engine, width, height):
    """Set the engine and size for a render, and put back what the scene had."""
    scene = bpy.context.scene
    render = scene.render
    saved = (
        render.engine,
        render.resolution_x,
        render.resolution_y,
        render.resolution_percentage,
        render.filepath,
        render.image_settings.file_format,
        scene.camera,
    )
    render.engine = engine
    render.resolution_x, render.resolution_y, render.resolution_percentage = width, height, 100
    try:
        yield scene
    finally:
        (
            render.engine,
            render.resolution_x,
            render.resolution_y,
            render.resolution_percentage,
            render.filepath,
            render.image_settings.file_format,
            scene.camera,
        ) = saved


@contextlib.contextmanager
def _temporary_camera(direction, up, ortho):
    """A camera that frames the whole scene from `direction`, removed afterwards."""
    lo, hi = _bounds()
    center, radius = (lo + hi) / 2, (hi - lo).length / 2
    data = bpy.data.cameras.new("_ailoy_view")
    camera = bpy.data.objects.new("_ailoy_view", data)
    bpy.context.scene.collection.objects.link(camera)
    direction = Vector(direction).normalized()
    # A camera looks down its −Z with +Y up: its Z is `direction`, its Y the scene's `up`.
    right = Vector(up).cross(direction).normalized()
    camera_up = direction.cross(right)
    camera.rotation_euler = Matrix((right, camera_up, direction)).transposed().to_euler()
    data.clip_end = radius * 20
    if ortho:
        # Fit the box as the camera sees it, centred, with a margin.
        corners = [Vector((x, y, z)) for x in (lo.x, hi.x) for y in (lo.y, hi.y) for z in (lo.z, hi.z)]
        across = [max(c.dot(axis) for c in corners) - min(c.dot(axis) for c in corners) for axis in (right, camera_up)]
        data.type = "ORTHO"
        data.ortho_scale = max(across) * 1.1
        camera.location = center + direction * radius * 4
    else:
        data.lens = 50
        # Far enough that the sphere around the box fits the narrower side of a 50 mm lens.
        camera.location = center + direction * radius * 1.1 / math.tan(math.atan(18 / 50))
    try:
        yield camera
    finally:
        bpy.data.objects.remove(camera)
        bpy.data.cameras.remove(data)


def _render_to(path):
    bpy.context.scene.render.filepath = path
    bpy.context.scene.render.image_settings.file_format = "PNG"
    bpy.ops.render.render(write_still=True)


def look(path, width=1280, height=720, samples=32):
    """Render the scene through its camera with EEVEE, as the person will see it, to `path`.

    Without a camera, from a perspective one that frames everything, which is not kept.
    """
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with _render_state("BLENDER_EEVEE", width, height) as scene:
        scene.eevee.taa_render_samples = samples
        if scene.camera is None:
            with _temporary_camera(VIEWS["iso"][0], VIEWS["iso"][1], ortho=False) as camera:
                scene.camera = camera
                _render_to(path)
        else:
            _render_to(path)
    print(path)


def views(path, size=512):
    """Draw the scene in Workbench from iso, front, right and top, on one 2×2 sheet at `path`.

    Iso is top left, front (from −Y) top right, right (from +X) bottom left, top bottom right.
    Flat shading in each object's material color: for shape and placement, not for looks.
    """
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tiles = []
    with tempfile.TemporaryDirectory() as tmp, _render_state("BLENDER_WORKBENCH", size, size) as scene:
        shading = scene.display.shading
        saved = shading.light, shading.color_type, shading.show_cavity
        shading.light, shading.color_type, shading.show_cavity = "STUDIO", "MATERIAL", True
        try:
            for name, (direction, up) in VIEWS.items():
                with _temporary_camera(direction, up, ortho=name != "iso") as camera:
                    scene.camera = camera
                    tile = os.path.join(tmp, f"{name}.png")
                    _render_to(tile)
                image = bpy.data.images.load(tile)
                pixels = np.empty(size * size * 4, dtype=np.float32)
                image.pixels.foreach_get(pixels)
                tiles.append(pixels.reshape(size, size, 4))
                bpy.data.images.remove(image)
        finally:
            shading.light, shading.color_type, shading.show_cavity = saved
    # Blender's rows run bottom up, so the top row of the sheet is the second block.
    sheet = np.concatenate(
        [np.concatenate(tiles[2:], axis=1), np.concatenate(tiles[:2], axis=1)], axis=0
    )
    image = bpy.data.images.new("_ailoy_views", size * 2, size * 2, alpha=True)
    image.pixels.foreach_set(sheet.ravel())
    image.filepath_raw, image.file_format = path, "PNG"
    image.save()
    bpy.data.images.remove(image)
    print(path)


def report():
    """Print what is in the scene as JSON: each object, its size and materials, and the whole box."""
    depsgraph = bpy.context.evaluated_depsgraph_get()
    objects = []
    for obj in bpy.context.scene.objects:
        entry = {
            "name": obj.name,
            "type": obj.type,
            "location": [round(v, 4) for v in obj.matrix_world.translation],
            "dimensions": [round(v, 4) for v in obj.dimensions],
            "parent": obj.parent.name if obj.parent else None,
            "visible": obj.visible_get(),
        }
        if obj.type == "MESH":
            mesh = obj.evaluated_get(depsgraph).to_mesh()
            entry["vertices"], entry["faces"] = len(mesh.vertices), len(mesh.polygons)
            obj.evaluated_get(depsgraph).to_mesh_clear()
            entry["modifiers"] = [m.type for m in obj.modifiers]
            entry["materials"] = [s.material.name if s.material else None for s in obj.material_slots]
        elif obj.type == "LIGHT":
            entry["light"] = {"type": obj.data.type, "energy": obj.data.energy}
        objects.append(entry)
    lo, hi = _bounds()
    scene = bpy.context.scene
    print(
        json.dumps(
            {
                "objects": objects,
                "bbox": {"min": list(lo), "max": list(hi), "size": list(hi - lo)},
                "camera": scene.camera.name if scene.camera else None,
                "frames": [scene.frame_start, scene.frame_end],
                "unmaterialed": [
                    o["name"] for o in objects if o["type"] == "MESH" and not any(o["materials"])
                ],
            },
            indent=1,
        )
    )


def save(folder):
    """Write the scene to `folder` as `scene.blend` and `scene.glb`."""
    folder = os.path.abspath(folder)
    os.makedirs(folder, exist_ok=True)
    bpy.ops.wm.save_as_mainfile(filepath=os.path.join(folder, "scene.blend"), copy=True)
    bpy.ops.export_scene.gltf(filepath=os.path.join(folder, "scene.glb"), export_format="GLB")
    print(os.path.join(folder, "scene.blend"), os.path.join(folder, "scene.glb"), sep="\n")


def turntable(path, frames=72, width=640, height=360, samples=8):
    """Render the scene turning once around Z, through its camera, as an MP4 at `path`.

    The camera is parented to a pivot at the scene's centre for the render and put back.
    """
    path = os.path.abspath(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    scene = bpy.context.scene
    lo, hi = _bounds()
    with contextlib.ExitStack() as stack, _render_state("BLENDER_EEVEE", width, height):
        if scene.camera is None:
            scene.camera = stack.enter_context(_temporary_camera(VIEWS["iso"][0], VIEWS["iso"][1], ortho=False))
        camera = scene.camera
        pivot = bpy.data.objects.new("_ailoy_pivot", None)
        scene.collection.objects.link(pivot)
        pivot.location = (lo + hi) / 2
        saved_parent, saved_matrix = camera.parent, camera.matrix_world.copy()
        saved_frames = scene.frame_start, scene.frame_end
        settings = scene.render.image_settings
        saved_format = settings.media_type, scene.render.ffmpeg.format, scene.render.ffmpeg.codec
        try:
            camera.parent = pivot
            camera.matrix_parent_inverse = pivot.matrix_world.inverted()
            scene.frame_start, scene.frame_end = 1, frames
            # One turn over the frames, the last one short of where the first starts.
            pivot.driver_add("rotation_euler", 2).driver.expression = f"(frame - 1) * {2 * math.pi / frames}"
            scene.eevee.taa_render_samples = samples
            settings.media_type = "VIDEO"
            scene.render.ffmpeg.format, scene.render.ffmpeg.codec = "MPEG4", "H264"
            scene.render.filepath = path
            bpy.ops.render.render(animation=True)
        finally:
            camera.parent = saved_parent
            camera.matrix_world = saved_matrix
            bpy.data.objects.remove(pivot)
            scene.frame_start, scene.frame_end = saved_frames
            settings.media_type, scene.render.ffmpeg.format, scene.render.ffmpeg.codec = saved_format
    print(path)


def _redraw():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            area.tag_redraw()


def _frame_all():
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                region = next(r for r in area.regions if r.type == "WINDOW")
                with bpy.context.temp_override(window=window, area=area, region=region):
                    bpy.ops.view3d.view_all()


NAMESPACE = {
    "bpy": bpy,
    "Vector": Vector,
    "look": look,
    "views": views,
    "report": report,
    "save": save,
    "turntable": turntable,
    "__name__": "__agent__",
}


def _serve(listener):
    """Answer one waiting request, if there is one. A `bpy.app.timers` callback."""
    try:
        conn, _ = listener.accept()
    except BlockingIOError:
        return 0.1
    with conn:
        conn.setblocking(True)
        data = b""
        while not data.endswith(b"\n") and (chunk := conn.recv(1 << 16)):
            data += chunk
        before = len(bpy.context.scene.objects)
        out = io.StringIO()
        ok = True
        try:
            with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
                exec(compile(json.loads(data)["code"], "<agent>", "exec"), NAMESPACE)
        except Exception:
            out.write(traceback.format_exc())
            ok = False
        conn.sendall(json.dumps({"ok": ok, "output": out.getvalue()}).encode())
    try:
        if len(bpy.context.scene.objects) > before:
            _frame_all()
        _redraw()
    except Exception:
        traceback.print_exc()
    return 0.0


def _start():
    # An empty scene to build in, and the viewport in material colors so the person sees them.
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj)
    # llvmpipe, where the host GPU is not taken, draws the floor grid and axes as noise.
    software = "llvmpipe" in gpu.platform.renderer_get()
    for window in bpy.context.window_manager.windows:
        for area in window.screen.areas:
            if area.type == "VIEW_3D":
                space = area.spaces.active
                space.shading.color_type = "MATERIAL"
                if software:
                    space.overlay.show_floor = space.overlay.show_axis_x = space.overlay.show_axis_y = False
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(ADDRESS)
    listener.listen()
    listener.setblocking(False)
    bpy.app.timers.register(lambda: _serve(listener), persistent=True)
    print("ailoy: listening", flush=True)


# After the window is up, so the startup scene and its 3D view exist.
bpy.app.timers.register(lambda: _start() and None, first_interval=0.5)
