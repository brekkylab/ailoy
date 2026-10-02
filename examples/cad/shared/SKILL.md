---
name: cad
description: Model solid 3D parts and assemblies in CadQuery, check them and draw them. Use it to design something to 3D print or machine — an enclosure, a bracket, gears, a mechanism — and deliver it as STEP, STL and pictures of it.
---

# CAD

A model is a Python script that builds solids with [CadQuery](https://cadquery.readthedocs.io),
a parametric CAD library on the OpenCascade kernel. `render.py` runs the script, checks what it
built, exports it and draws it, so that you can look at what you made and fix it before you
hand it over.

## The loop

1. Write the model as `model.py` in a folder under `/artifacts` named for what it is, such as
   `/artifacts/gear-bearing/model.py`. The script is part of what you deliver: keep the
   dimensions as named parameters at its top.
2. Run it from this directory:

   ```sh
   python3 render.py /artifacts/gear-bearing/model.py --out /artifacts/gear-bearing
   ```

3. Read the report it prints, then look at `views.png` with your `read` tool. Say to yourself
   what should be there and check that it is: every feature, the right way round, the right
   size.
4. Fix and run again until the report is clean and the pictures show what was asked for.
   Most models need a few rounds. `--no-export` skips the files while you iterate.

Do not describe a model you have not looked at. The report catches what is invalid or
overlapping; only the pictures show that a hole is on the wrong face.

## The script

It runs as CQ-editor runs one: `cq` is already imported, and each part is handed over with
`show_object`.

```python
width, depth, height, wall = 60.0, 40.0, 25.0, 2.0

box = cq.Workplane("XY").box(width, depth, height).faces(">Z").shell(-wall)
lid = cq.Workplane("XY").box(width, depth, wall).translate((0, 0, height / 2 + wall / 2 + 0.4))

show_object(box, "box", {"color": "#4c78a8"})
show_object(lid, "lid", {"color": "#f58518"})
```

* `show_object(obj, name, options)` adds a part. `obj` is a Workplane, a Shape, an
  `cq.Assembly` or a list of them. `options["color"]` is a CSS color name, `#rrggbb`, a tuple of
  0–1 or 0–255 values, or a `cq.Color`. Parts that name no color get one of a palette.
* A script that calls `show_object` for nothing is taken for its `result` variable.
* Units are millimetres and Z is up. For a print, Z is off the bed: put the face it prints on
  at `z = 0`.
* Parts that are separate things — the planets of a gear set, a lid and its box — are
  separate `show_object` calls. The report checks each pair of them for overlap, and each is
  its own STL.

## render.py

```
python3 render.py MODEL.py --out DIR [--no-export] [--size PX]
                  [--explode F] [--section AXIS[=V]] [--turntable [--frames N]]
```

It writes into `DIR`:

* `views.png` — iso, front (from −Y), right (from +X) and top on one sheet, with the model's
  size in each; and `iso.png`, `front.png`, `right.png`, `top.png` alone.
* `model.step` — every part, named and colored, as one STEP assembly, for other CAD tools.
* `model.glb` — the same as colored meshes, for a viewer or a web page.
* `parts/NAME.stl` — each part on its own, for a slicer.
* `exploded.png` with `--explode F` — each part moved away from the middle of the model, F
  times as far. Parts on the model's centre do not move, so for a stack of concentric parts
  make an exploded copy in a script of its own, with `translate`.
* `section.png` with `--section z=5` — every part cut by the plane `z = 5` and seen from the
  side that was taken away, the cut faces lighter. Without `=V` it cuts through the middle.
  This is how to see inside: walls, bores, clearances between parts.
* `turntable.gif` with `--turntable` — the model turning once, 36 frames by default. The
  report is the same without it, so make it last, for the user.

The report on stdout is one JSON object:

* `parts` — for each: `valid` (OpenCascade's own check), `solids` (how many), `bbox` (`min`,
  `max`, `size`), `volume` in mm³, `area` in mm², `color`, `triangles` and `watertight`
  (whether its mesh is closed).
* `bbox` — of the whole model.
* `interferences` — every pair of parts whose solids share more than 0.001 mm³, with how
  much. Parts that only touch share none. `--interference V` changes the threshold.
* `files`, and `seconds` for each step.

A script that fails comes back as `{"error": ..., "traceback": ...}` with exit code 1.

What the report should say before you hand a model over:

* every part `valid`, with the number of `solids` it should have: one, for a part to print;
* every part `watertight`, or a slicer may not take it;
* `interferences` empty, unless two parts are meant to be one — then union them;
* `bbox` sizes that match what was asked for.

## CadQuery

A Workplane is a chain: a plane, 2D shapes on it, then operations that turn them into
solids and change them.

* **Sketching.** `rect`, `circle`, `polygon(n, d)`, `slot2D`, `ellipse`, and
  `polyline(points).close()` or `spline(points)` for outlines of your own. `pushPoints`,
  `rarray(xs, ys, nx, ny)` and `polarArray(r, 0, 360, n)` place copies of what follows.
* **Solids.** `extrude(h)` (`both=True` for both ways), `revolve(deg, (x0, y0, 0), (x1, y1, 0))`,
  `twistExtrude(h, deg)` for helical and herringbone teeth, `loft()` between stacked
  sketches, `sweep(path)` along a wire, with `cq.Wire.makeHelix(pitch, height, r)` for
  springs and threads. `box`, `cylinder`, `sphere` directly.
* **Selecting.** `faces(">Z")` is the top face, `faces("<Z")` the bottom, `edges("|Z")` the
  vertical edges, `faces(">Z").workplane()` a new plane on the top face to sketch on.
* **Features.** `hole(d)`, `cboreHole`, `cskHole`, `shell(-t)` to hollow (the selected faces
  are left open), `fillet(r)` and `chamfer(d)` on selected edges.
* **Booleans.** `union`, `cut`, `intersect`. Moving: `translate((x, y, z))`,
  `rotate((0, 0, 0), (0, 0, 1), deg)`, `mirror("XY", (0, 0, z))`.

Where it goes wrong:

* `polyline` with two points the same in a row fails with `BRep_API: command not done`. So
  does a profile that crosses itself. Drop repeated points before you close it.
* A `fillet` too big for the edges it is on fails with `StdFail_NotDone`, or makes an invalid
  solid. Fillet last, select edges narrowly, and try a smaller radius.
* A boolean between faces that lie exactly on each other can come back invalid. Let the
  cutter stick out past the face it cuts, by 0.01 mm or more.
* Hundreds of small `union`s or `cut`s in a loop are slow. Draw the repeated feature in 2D
  and extrude it once, or place copies with `pushPoints` or `polarArray`.
* `twistExtrude` twists about the workplane's origin: centre the profile on it.

## Gears

For spur gears with module `m`, pressure angle `α` (20°) and `z` teeth: the pitch radius is
`m·z/2`, the base radius `m·z/2·cos α`, the tip `m` further out, the root `1.25·m` further in.
A tooth's flank is the involute of the base circle, at angle `inv(φ) = tan φ − φ` for the
radius `r_b / cos φ`. Two external gears mesh at centre distance `m·(z1 + z2)/2`. A ring gear
is a disc with an external gear of its tooth count cut out of it, and then a circle of radius
`m·z/2 − m` cut out too: without that the ring's teeth are as long as the gear's gaps were
deep, and run into the roots of the planets.

A planetary set has `z_ring = z_sun + 2·z_planet`, and `n` planets spaced evenly only fit when
`(z_sun + z_ring) / n` is a whole number. Each planet then has to be turned so that its teeth
fall into the sun's gaps and the ring's: check it with the report's `interferences` and a
`--section` through the teeth, not by eye.

Parts that turn against each other need a clearance. For a print that comes off the bed
assembled, 0.3–0.5 mm between the flanks — make each tooth thinner at the pitch circle by half
of it, so two teeth in mesh leave all of it — and the same between any faces that slide.
Teeth in mesh with no clearance show as a small overlap in `interferences`, a fraction of a
mm³ per pair; a tooth in the wrong place shows as hundreds.

## Printing

* Put the largest flat face on `z = 0`, and keep overhangs under 45° from vertical, or they
  need supports.
* Walls 1.2 mm or thicker, and holes that take a screw 0.2–0.3 mm bigger than the screw.
* One solid per part: a part that is two solids is two parts, or a part the union missed.

## Delivering

In `/artifacts/NAME` leave the script, `model.step`, the STLs, `model.glb` and the pictures
that show it best. In your reply, say what the model is, its main dimensions, what the
report said, and which files are there.
