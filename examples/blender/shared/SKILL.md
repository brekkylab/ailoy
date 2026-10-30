---
name: blender
description: Build 3D scenes and assets in a live Blender — models, materials, lights, a camera — render them with EEVEE, look at them and fix them. Use it to make a 3D model, a scene, a product shot or a turntable, and deliver it as .blend, .glb and renders.
---

# Blender

A Blender 5.2 is running on a display the person is watching over VNC. You drive it with
Python: `run.py` sends a script to it, the script runs inside Blender with `bpy`, and what it
prints comes back. Every object you add appears in the person's viewport as you add it, so
build the scene in steps they can follow, not in one script at the end.

Blender runs on another machine than your shell. `/artifacts` and `/context` are the same
folders on both, so a path under them means the same file to Blender and to you. Anything
else — `/tmp`, your working directory — Blender cannot see, and you cannot see what Blender
writes there.

## The loop

1. Write a step of the scene as a script under `/artifacts/NAME/`, such as
   `/artifacts/ramen-shop/01_blockout.py`, and run it from this directory:

   ```sh
   python3 run.py /artifacts/ramen-shop/01_blockout.py
   ```

   Short checks can go inline: `python3 run.py -c 'report()'`.
2. Look at it. `views("/artifacts/NAME/views.png")` draws it from four sides in a second or
   two; `look("/artifacts/NAME/look.png")` renders it through the camera with EEVEE. Read
   the picture with your `imgread` tool and say to yourself what should be there and check
   that it is: every object, the right way round, the right size, nothing floating or
   sunk into the floor.
3. Fix and go again. Most scenes need several rounds: blockout, shapes, materials, lighting,
   camera.

Do not describe a scene you have not looked at. `report()` tells you what exists; only the
pictures show that a roof is upside down.

## What runs where

The script runs in Blender's own Python, on its main thread, in one namespace kept between
runs: a function or variable one script defines, the next can use. It starts with `bpy`,
`Vector` and these:

* `look(path, width=1280, height=720, samples=32)` — the scene through `scene.camera` with
  EEVEE, as the person will see it. Without a camera, from one that frames everything and is
  not kept.
* `views(path, size=512)` — iso (top left), front from −Y (top right), right from +X
  (bottom left) and top (bottom right) on one sheet, flat-shaded in each object's material
  color. For shape and placement, not looks: it ignores lights and the camera.
* `report()` — JSON of every object: type, location, dimensions, vertex and face counts,
  modifiers, materials and parent; the box around the whole scene; and `unmaterialed`, the
  meshes with no material.
* `save(folder)` — `scene.blend` and `scene.glb` in `folder`.
* `turntable(path, frames=72, width=640, height=360, samples=8)` — an MP4 of the scene
  turning once in front of the camera. It is the slowest thing here: make it last.

A render blocks Blender while it runs, so the person's view freezes for that long. Without
the host's GPU, EEVEE renders on the CPU: a `look` takes half a minute and a `turntable`
minutes. Use `views` while you iterate and `look` when a step is done.

## bpy

Use the data API, not operators, wherever you can: it does not depend on what is selected or
which area is under the mouse.

```python
import bmesh

mesh = bpy.data.meshes.new("counter")
bm = bmesh.new()
bmesh.ops.create_cube(bm, size=1.0)
bm.to_mesh(mesh)
bm.free()
counter = bpy.data.objects.new("counter", mesh)
bpy.context.scene.collection.objects.link(counter)
counter.location = (0, 0, 0.5)
counter.scale = (2.0, 0.6, 1.0)
```

* `bpy.ops.mesh.primitive_*_add(...)` works too, and leaves the new object in
  `bpy.context.object`. Give it a name right away: `bpy.context.object.name = "lamp"`.
* Units are metres and Z is up. Put the floor at `z = 0` and stand things on it.
* Group with collections: `col = bpy.data.collections.new("shop")`,
  `bpy.context.scene.collection.children.link(col)`, then `col.objects.link(obj)`.
* Modifiers do the modelling a person would do by hand: `obj.modifiers.new("bevel", "BEVEL")`
  (`width`, `segments`), `"SUBSURF"` (`levels`, `render_levels`), `"ARRAY"` (`count`,
  `relative_offset_displace`), `"MIRROR"`, `"SOLIDIFY"` (`thickness`), `"BOOLEAN"`
  (`object`, `operation`). Smooth a mesh with `obj.data.shade_smooth()`.
* To start a step over, remove what it made: `bpy.data.objects.remove(obj)`.

### Materials

```python
mat = bpy.data.materials.new("glazed tile")
bsdf = mat.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.1, 0.35, 0.6, 1)
bsdf.inputs["Roughness"].default_value = 0.25
mat.diffuse_color = (0.1, 0.35, 0.6, 1)  # what the viewport and `views` show
obj.data.materials.append(mat)
```

* Other inputs worth knowing: `"Metallic"`, `"Emission Color"` with `"Emission Strength"` for
  signs and lamps, `"Alpha"` for glass-like cut-outs, `"Transmission Weight"` for glass.
* A material per face: append several, then set `polygon.material_index`.
* Procedural texture: add a `ShaderNodeTexNoise` or `ShaderNodeTexVoronoi` with
  `mat.node_tree.nodes.new(...)` and link its output into an input with
  `mat.node_tree.links.new(src.outputs[0], bsdf.inputs["Base Color"])`, through a
  `ShaderNodeValToRGB` for a color ramp.

### Light, world and camera

* Lights: `bpy.data.lights.new("key", "AREA")` (`energy` in watts, a few hundred for a room;
  `size`), `"SUN"` (`energy` 2–5), `"POINT"`. Link an object made from it like a mesh. A key,
  a fill at a third of its energy and a rim behind is a good start.
* The world: `bpy.context.scene.world.node_tree.nodes["Background"].inputs["Color"]` and
  `["Strength"]` set the ambient light and the background.
* The camera: `bpy.data.cameras.new("cam")`, an object from it, `scene.camera = obj`. Aim it
  with `obj.rotation_euler = (target - obj.location).to_track_quat("-Z", "Y").to_euler()`.
  `data.lens` is the focal length in mm: 35–50 for a scene, 85 for a product.
* `scene.render.film_transparent = True` renders without the background, for a cut-out.

### Where it goes wrong

* `bpy.ops` that need a 3D view or a selection fail with `context is incorrect`. Use the data
  API instead, or `bpy.context.temp_override(...)`.
* An object made with `bpy.data.objects.new` is not in the scene until it is linked to a
  collection.
* Scale applied to an object, not its mesh, also scales its bevels and arrays. Keep scale at
  1 on objects with modifiers, and size the mesh instead, or apply the scale with
  `bpy.ops.object.transform_apply(scale=True)` on the selected object.
* An exception in a script stops it where it raised; what it made before is still in the
  scene. Check with `report()` before you run it again, or the second run makes doubles
  (`chair.001`).

## Delivering

In `/artifacts/NAME` leave the scripts, `save("/artifacts/NAME")`'s `scene.blend` and
`scene.glb`, the last `look.png`, and a `turntable.mp4` if one was asked for. In your
reply, say what is in the scene, what the pictures show, and which files are there. Blender
stays open for the person to look around in until they close the run.
