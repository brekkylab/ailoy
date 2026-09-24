"""Download the SAM3 ONNX encoders and convert them into ncnn models.

    uv run prepare_model.py [DATA_DIR]

DATA_DIR is `data/` beside this file by default. The ONNX files go into `DATA_DIR/onnx`,
from `wkentaro/sam3-onnx-models` on the Hugging Face Hub at a pinned revision, and for each
of `sam3_image_encoder` and `sam3_language_encoder` this writes into `DATA_DIR/ncnn`

* `NAME.ncnn.param`, `NAME.ncnn.bin` -- the model;
* `NAME.json` -- its ncnn input and output blob names, and what was peeled off it (below);
* `NAME.TENSOR.npy` -- a lookup table the caller runs before the model, or an output that
  is the same for every input.

and `tokenizer.json`, CLIP's tokenizer for the language encoder's prompts, from
`openai/clip-vit-base-patch32` at a pinned revision. It gives the same ids as the
`SimpleTokenizer` SAM3 tokenizes with, and needs `tokenizers` alone.

The decoder is neither downloaded nor converted: it takes scalar and bool inputs pnnx does not accept, and
answers with as many masks as it found (`NonZero`), which ncnn has no shape for.

**Peeled inputs.** ncnn has no integer or bool tensors, so what the ONNX graphs do to their
raw inputs first is taken off and left to the caller:

* a `Cast` straight off an input -- the image encoder's `uint8` image. The model takes the
  cast's output instead: the same pixels, as floats in 0..255, with a batch dim in front.
* a `Gather` from a weight table by an input -- the language encoder's token embeddings.
  The model takes the embeddings, and the table is saved next to it as `.npy`.
* an output computed from the raw input alone -- the language encoder's padding mask,
  `tokens == 0`. Dropped, and listed in the `.json`.
* an output computed from no input at all -- the image encoder's position encodings. Run
  once here and saved as `.npy`.

**Pieces.** pnnx infers shapes by making every intermediate tensor a graph output and
running onnxruntime once. For ViT-H at 1008² that is ~73 GB held at the same time -- each
global-attention layer alone is 16 x 5184² floats -- so a graph is cut between blocks, where
the residual stream is the one tensor crossing the cut, each piece is converted on its own,
and the pieces are stitched back into one ncnn model. ncnn reads a `.bin` layer by layer in
`.param` order, so concatenating params and bins in the same order is the whole of the merge.
"""

import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import NamedTuple

import numpy as np
import onnx
import onnxruntime as ort
from onnx import numpy_helper, shape_inference
from huggingface_hub import hf_hub_download
from onnx.utils import Extractor

MODELS = ["sam3_language_encoder", "sam3_image_encoder"]

REPO = "wkentaro/sam3-onnx-models"
REVISION = "8e0e8d84459144ad442e1ef514b4f29a221f5097"
TOKENIZER_REPO = "openai/clip-vit-base-patch32"
TOKENIZER_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"

# The activation bytes one piece may hold during pnnx's shape inference. An image-encoder
# block is ~1.9 GB and a global-attention one ~4.4 GB, so this is one or two blocks a piece;
# the whole language encoder fits in one. Peak memory is roughly twice this.
PIECE_BUDGET = int(float(os.environ.get("SAM3_PIECE_BUDGET_GB", "5")) * 10**9)

ELEM_SIZE = {1: 4, 2: 1, 6: 4, 7: 8, 9: 1, 10: 2, 11: 8}


def consumers(graph: onnx.GraphProto) -> dict[str, list[onnx.NodeProto]]:
    out: dict[str, list[onnx.NodeProto]] = {}
    for node in graph.node:
        for i in node.input:
            out.setdefault(i, []).append(node)
    return out


def reachable(graph: onnx.GraphProto, sources: set[str], skip=frozenset()) -> set[str]:
    """Every tensor computed from `sources`, not passing through the nodes in `skip`."""
    seen = set(sources)
    for node in graph.node:  # ONNX nodes are in topological order
        if node.name not in skip and any(i in seen for i in node.input):
            seen.update(node.output)
    return seen


# What an attention mask's `-inf` becomes. Past any score a softmax sees here, so the weight
# it gives is 0 all the same, and inside fp16's range.
MASKED = -1e4


def finite_masks(model: onnx.ModelProto):
    """Replace `-inf` in the weights with MASKED.

    The language encoder's causal mask is `-inf` above the diagonal. onnxruntime and ncnn on
    macOS take that through the softmax as 0, but ncnn's Linux arm64 and Vulkan kernels make
    NaN of it, and every row after it goes with it.
    """
    for i, w in enumerate(model.graph.initializer):
        a = numpy_helper.to_array(w)
        if a.dtype.kind == "f" and np.isneginf(a).any():
            model.graph.initializer[i].CopyFrom(numpy_helper.from_array(np.where(np.isneginf(a), MASKED, a).astype(a.dtype), w.name))


def batch_inputs(model: onnx.ModelProto):
    """Give an unbatched input its batch dim straight off its Cast.

    The image comes in as `[3, H, W]`, is normalized, and only then unsqueezed to
    `[1, 3, H, W]` for the patch embedding. pnnx takes the leading dim of what it is given as
    the batch, so it keeps that late `1` as a dim of its own, and the Conv reads a 4-D blob
    with the channels in the wrong place. Unsqueezing first gives the same numbers --
    the normalization's constants broadcast over the new axis -- and a batch pnnx can drop.
    """
    g = model.graph
    use = consumers(g)
    weights = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    for x in g.input:
        (cast, *rest) = use.get(x.name, [None])
        if rest or cast is None or cast.op_type != "Cast":
            continue
        chain, t = [], cast.output[0]
        while len(use.get(t, [])) == 1 and use[t][0].op_type in ("Mul", "Sub", "Div", "Add"):
            chain.append(use[t][0])
            t = chain[-1].output[0]
        (unsqueeze, *rest) = use.get(t, [None])
        if rest or unsqueeze is None or unsqueeze.op_type != "Unsqueeze" or not chain:
            continue
        if list(weights.get(unsqueeze.input[1], [])) != [0]:
            continue
        batched = f"{cast.output[0]}_batched"
        chain[0].input[list(chain[0].input).index(cast.output[0])] = batched
        chain[-1].output[0] = unsqueeze.output[0]
        unsqueeze.input[0], unsqueeze.output[0] = cast.output[0], batched
        nodes = [n for n in g.node if n is not unsqueeze]
        nodes.insert(nodes.index(cast) + 1, unsqueeze)
        del g.node[:]
        g.node.extend(nodes)
        # The chain's tensors changed rank; `lower_rank` infers them afresh.
        del g.value_info[:]


def peel(model: onnx.ModelProto, extractor: Extractor, name: str, ncnn_dir: Path) -> dict:
    """Take off what ncnn cannot run on the raw inputs; see the module docs."""
    g = model.graph
    weights = {i.name: i for i in g.initializer}
    raw = [i.name for i in g.input]
    inputs, tables, peeled = [], {}, set()
    for x in raw:
        for node in consumers(g).get(x, []):
            if node.op_type == "Cast":
                out = node.output[0]
                # Past the batch dim `batch_inputs` put straight after it, if it did.
                after = consumers(g).get(out, [])
                if len(after) == 1 and after[0].op_type == "Unsqueeze":
                    peeled.add(after[0].name)
                    out = after[0].output[0]
                inputs.append(out)
            elif node.op_type == "Gather" and node.input[0] in weights and node.input[1] == x:
                table = numpy_helper.to_array(weights[node.input[0]]).astype(np.float16)
                path = f"{name}.{node.output[0]}.npy"
                np.save(ncnn_dir / path, table)
                tables[node.output[0]] = {"table": path, "indices": x}
                inputs.append(node.output[0])
            else:
                continue
            peeled.add(node.name)
        if not any(x in n.input for n in g.node if n.name in peeled):
            inputs.append(x)

    from_raw = reachable(g, set(raw), skip=peeled)
    dropped = [o.name for o in g.output if o.name in from_raw]

    # Computed once here, since nothing about them changes from one input to the next.
    constant = [o.name for o in g.output if o.name not in reachable(g, set(raw))]
    constants = {}
    if constant:
        sub = extractor.extract_model([], constant)
        values = ort.InferenceSession(sub.SerializeToString()).run(None, {})
        for o, value in zip(constant, values):
            constants[o] = f"{name}.{o}.npy"
            np.save(ncnn_dir / constants[o], value)

    outputs = [o.name for o in g.output if o.name not in from_raw and o.name not in constants]
    return {
        "inputs": inputs,
        "outputs": outputs,
        "tables": tables,
        "constants": constants,
        "dropped": dropped,
    }


def inferred(model: onnx.ModelProto) -> onnx.GraphProto:
    """`model`'s graph with every shape inferred -- worked out on a copy without its weights.

    Shape inference serializes the model it is handed and hands back a parsed copy, which for
    the image encoder is 1.8 GB each way. It only ever needs a weight's dims, and the values
    of the small ones (the shapes a Reshape is given), so the large ones go in as dims alone.
    """
    skeleton = onnx.ModelProto(ir_version=model.ir_version, opset_import=model.opset_import)
    graph = skeleton.graph
    graph.node.extend(model.graph.node)
    graph.input.extend(model.graph.input)
    graph.output.extend(model.graph.output)
    graph.value_info.extend(model.graph.value_info)
    for w in model.graph.initializer:
        if w.ByteSize() > 4096:
            w = onnx.TensorProto(name=w.name, data_type=w.data_type, dims=w.dims)
        graph.initializer.append(w)
    return shape_inference.infer_shapes(skeleton).graph


def shapes_of(model: onnx.ModelProto) -> dict[str, list[int]]:
    g = inferred(model)
    out = {v.name: [d.dim_value for d in v.type.tensor_type.shape.dim] for v in [*g.input, *g.value_info, *g.output]}
    out.update({i.name: list(i.dims) for i in g.initializer})
    return out


def merged(shape: list[int], axis: int) -> tuple[list[int], int]:
    """`shape` folded to at most four dims around `axis`: everything before its neighbour,
    its neighbour, the axis, everything after -- and where that axis ends up."""
    before, after = int(np.prod(shape[: axis - 1])) if axis > 0 else 1, int(np.prod(shape[axis + 1 :]))
    dims = ([before] if axis > 1 else []) + ([shape[axis - 1]] if axis > 0 else []) + [shape[axis]]
    new_axis = len(dims) - 1
    return dims + ([after] if after != 1 else []), new_axis


def lower_rank(model: onnx.ModelProto):
    """Rewrite every op on a tensor of more than four dims into ones on four or fewer.

    ncnn's tensors stop at four dims (c, d, h, w) and pnnx writes a shapeless Reshape for
    anything past that, which crashes ncnn when it runs. SAM3's ViT goes to five and six in
    its window partition, its qkv split and its RoPE -- all only moving data around, so each
    has a lower-rank spelling:

    * Transpose: drop the size-1 axes and merge the runs that stay adjacent under the perm.
    * Split of a Transpose: split first, then transpose the pieces.
    * Split, Gather, Concat: merge the axes on either side of the one they work on.
    * Squeeze, Unsqueeze, Reshape: all a Reshape, and a chain of Reshapes is one Reshape --
      which is what makes the high-rank tensors between the rewrites disappear.

    Shapes are static throughout, which is what makes every one of these a plain Reshape.
    """
    g = model.graph
    shape = shapes_of(model)
    rank = lambda t: len(shape.get(t, [])) if t else 0
    counter = iter(range(10**9))
    # The Reshapes made here, whose shapes are spelled out in full. Only these fold: one of
    # the graph's own may say `0` for "as the input has it", which folding would change.
    static: set[str] = set()

    def fresh(dims: list[int]) -> str:
        name = f"lower_{next(counter)}"
        shape[name] = dims
        return name

    def const(values: list[int]) -> str:
        name = fresh([len(values)])
        g.initializer.append(numpy_helper.from_array(np.array(values, np.int64), name))
        return name

    def reshape(x: str, dims: list[int], out: str | None = None) -> tuple[onnx.NodeProto, str]:
        out = out or fresh(dims)
        node = onnx.helper.make_node("Reshape", [x, const(dims)], [out], name=f"{out}_reshape")
        static.add(node.name)
        return node, out

    def axis_of(node: onnx.NodeProto, r: int) -> int:
        a = next((a.i for a in node.attribute if a.name == "axis"), 0)
        return a % r

    # Split of a high-rank Transpose becomes a Transpose of each piece of a Split.
    used_by: dict[str, list[onnx.NodeProto]] = consumers(g)
    made = {o: n for n in g.node for o in n.output}
    nodes = []
    for node in g.node:
        src = made.get(node.input[0]) if node.input else None
        if node.op_type == "Split" and src is not None and src.op_type == "Transpose" and rank(node.input[0]) > 4 and len(used_by[node.input[0]]) == 1:
            perm = list(next(a.ints for a in src.attribute if a.name == "perm"))
            a = axis_of(node, len(perm))
            pieces = []
            for o in node.output:
                dims = list(shape[src.input[0]])
                dims[perm[a]] = shape[o][a]
                pieces.append(fresh(dims))
            split = onnx.helper.make_node("Split", [src.input[0], *node.input[1:]], pieces, name=f"{node.name}_early")
            split.attribute.extend([a_ for a_ in node.attribute if a_.name != "axis"])
            split.attribute.append(onnx.helper.make_attribute("axis", perm[a]))
            nodes = [n for n in nodes if n is not src]
            nodes.append(split)
            for piece, o in zip(pieces, node.output):
                nodes.append(onnx.helper.make_node("Transpose", [piece], [o], name=f"{o}_transpose", perm=perm))
            continue
        nodes.append(node)

    lowered = []
    for node in nodes:
        ios = [*node.input, *node.output]
        if not any(rank(t) > 4 for t in ios) or node.op_type == "Constant":
            lowered.append(node)
            continue
        op, x, y = node.op_type, node.input[0], node.output[0]

        if op in ("Reshape", "Squeeze", "Unsqueeze", "Flatten"):
            lowered.append(reshape(x, shape[y], y)[0])

        elif op == "Transpose":
            perm = list(next(a.ints for a in node.attribute if a.name == "perm"))
            dims = shape[x]
            kept = [a for a in range(len(dims)) if dims[a] != 1]
            order = [kept.index(a) for a in perm if dims[a] != 1]
            kdims = [dims[a] for a in kept]
            groups: list[list[int]] = []
            for a in order:
                if groups and groups[-1][-1] + 1 == a:
                    groups[-1].append(a)
                else:
                    groups.append([a])
            by_input = sorted(groups)
            red_in = [int(np.prod([kdims[a] for a in grp])) for grp in by_input]
            red_perm = [by_input.index(grp) for grp in groups]
            assert len(red_in) <= 4, f"{node.name}: transpose {dims} {perm} does not lower"
            n1, x1 = reshape(x, red_in)
            t = fresh([red_in[p] for p in red_perm])
            n2 = onnx.helper.make_node("Transpose", [x1], [t], name=f"{t}_transpose", perm=red_perm)
            lowered += [n1, n2, reshape(t, shape[y], y)[0]]

        elif op in ("Split", "Gather", "Concat"):
            a = axis_of(node, rank(x))
            if op == "Gather":
                assert rank(node.input[1]) == 0, f"{node.name}: only a scalar index lowers"
                dims, na = merged(shape[x], a)
                n1, x1 = reshape(x, dims)
                out = fresh(dims[:na] + dims[na + 1 :])
                n2 = onnx.helper.make_node("Gather", [x1, node.input[1]], [out], name=f"{out}_gather", axis=na)
                lowered += [n1, n2, reshape(out, shape[y], y)[0]]
            elif op == "Split":
                dims, na = merged(shape[x], a)
                n1, x1 = reshape(x, dims)
                outs = []
                for o in node.output:
                    d = list(dims)
                    d[na] = shape[o][a]
                    outs.append(fresh(d))
                n2 = onnx.helper.make_node("Split", [x1, *node.input[1:]], outs, name=f"{outs[0]}_split")
                n2.attribute.extend([a_ for a_ in node.attribute if a_.name != "axis"])
                n2.attribute.append(onnx.helper.make_attribute("axis", na))
                lowered += [n1, n2, *(reshape(o1, shape[o], o)[0] for o1, o in zip(outs, node.output))]
            else:
                ins = []
                for i in node.input:
                    n1, x1 = reshape(i, merged(shape[i], a)[0])
                    lowered.append(n1)
                    ins.append(x1)
                dims, na = merged(shape[y], a)
                out = fresh(dims)
                lowered.append(onnx.helper.make_node("Concat", ins, [out], name=f"{out}_concat", axis=na))
                lowered.append(reshape(out, shape[y], y)[0])
        else:
            raise SystemExit(f"{node.name}: no lower-rank form for {op} on {[shape.get(t) for t in ios]}")

    # A Reshape of a Reshape reads the first one's input, and what nobody reads goes.
    source = {}
    for node in lowered:
        if node.name in static:
            x = node.input[0]
            source[node.output[0]] = source.get(x, x)
            node.input[0] = source.get(x, x)
    wanted = {o.name for o in g.output}
    kept = []
    for node in reversed(lowered):
        if any(o in wanted for o in node.output):
            kept.append(node)
            wanted.update(node.input)
    kept.reverse()

    live = {t for n in kept for t in [*n.input, *n.output]}
    high = sorted({t for t in live if rank(t) > 4})
    assert not high, f"still above four dims: {high[:5]}"
    names = {i for n in kept for i in n.input}
    initializers = [i for i in g.initializer if i.name in names]
    del g.initializer[:]
    g.initializer.extend(initializers)
    del g.node[:]
    g.node.extend(kept)
    # What the old shapes said about tensors that are gone or changed, inferred afresh --
    # the Extractor cuts pieces by these.
    del g.value_info[:]
    g.value_info.extend(inferred(model).value_info)


def cut_points(model: onnx.ModelProto, inputs: list[str]) -> list[str]:
    """Tensors to cut at, each the only input-derived tensor alive at that point, spaced so
    that no piece holds more than PIECE_BUDGET of activations."""
    g = inferred(model)
    info = {v.name: v.type.tensor_type for v in list(g.value_info) + list(g.output)}

    def nbytes(name: str) -> int:
        if name not in info:
            return 0
        t = info[name]
        return int(np.prod([d.dim_value for d in t.shape.dim])) * ELEM_SIZE.get(t.elem_type, 4)

    dynamic = reachable(g, set(inputs))
    last_use = {o.name: len(g.node) for o in g.output}
    for i, node in enumerate(g.node):
        for t in node.input:
            last_use[t] = max(last_use.get(t, -1), i)

    cuts, live, held, candidate = [], set(), 0, None
    for i, node in enumerate(g.node):
        for o in node.output:
            held += nbytes(o)
            if o in dynamic and last_use.get(o, -1) > i:
                live.add(o)
        live = {t for t in live if last_use[t] > i}
        if held > PIECE_BUDGET and candidate is not None:
            cuts.append(candidate[0])
            held -= candidate[1]
            candidate = None
        if len(live) == 1:
            candidate = (next(iter(live)), held)
    return cuts


def pnnx_binary() -> str:
    """The pnnx executable inside its pip package.

    Called directly rather than through the `pnnx` script on PATH, which goes by way of the
    package's Python half and imports torch first -- seconds a piece, for a path that never
    uses it. Found without importing the package, for the same reason.
    """
    spec = importlib.util.find_spec("pnnx")
    assert spec and spec.origin, "pnnx is not installed"
    return str(Path(spec.origin).parent / "pnnx")


class Piece(NamedTuple):
    param: Path
    bin: Path
    # Which ONNX axis pnnx dropped from each input and output blob, or None for none. ncnn
    # has no batch dimension, so pnnx drops one where it can find it -- and where it cannot,
    # keeps the tensor whole. Its own test script is the only place it says which it did.
    squeezed_in: list[int | None]
    squeezed_out: list[int | None]


def pnnx(onnx_path: Path) -> Piece:
    """Run pnnx on one ONNX file and hand back the ncnn pair it wrote beside it."""
    work, stem = onnx_path.parent.resolve(), onnx_path.stem
    param, bin_, test = work / f"{stem}.ncnn.param", work / f"{stem}.ncnn.bin", work / f"{stem}_ncnn.py"
    out = {
        "pnnxparam": f"{stem}.pnnx.param",
        "pnnxbin": f"{stem}.pnnx.bin",
        "pnnxpy": f"{stem}_pnnx.py",
        "pnnxonnx": f"{stem}.pnnx.onnx",
        "ncnnparam": param.name,
        "ncnnbin": bin_.name,
        "ncnnpy": test.name,
    }
    done = subprocess.run(
        [pnnx_binary(), onnx_path.name, "fp16=1", *(f"{k}={v}" for k, v in out.items())],
        capture_output=True,
        text=True,
        # pnnx resolves an ONNX file's external data against its working directory.
        cwd=work,
    )
    # pnnx narrates every pass; only a failure's account is worth showing.
    if done.returncode != 0:
        sys.stderr.write(done.stdout[-4000:] + done.stderr[-4000:])
        raise SystemExit(f"pnnx {onnx_path.name}: exit {done.returncode}")
    script = test.read_text()
    axis = lambda m: int(m) if m else None
    squeezed_in = [axis(m) for m in re.findall(r"ncnn\.Mat\(in\d+(?:\.squeeze\((\d+)\))?\.numpy", script)]
    squeezed_out = [axis(m) for m in re.findall(r"np\.array\(out\d+\)\)(?:\.unsqueeze\((\d+)\))?", script)]
    # Only the ncnn pair is wanted; the pnnx intermediates are as large as the weights.
    for leftover in [*work.glob(f"{stem}.pnnx*"), *work.glob(f"{stem}_*.py")]:
        leftover.unlink()
    return Piece(param, bin_, squeezed_in, squeezed_out)


def ncnn_shape(shape: list[int], squeezed: int | None) -> list[int]:
    return [d for i, d in enumerate(shape) if i != squeezed]


def reshape_layer(name: str, src: str, dst: str, shape: list[int]) -> str:
    """An ncnn Reshape to `shape`, which ncnn spells innermost first: w, h, d, c."""
    keys = {1: ["0"], 2: ["1", "0"], 3: ["2", "1", "0"], 4: ["2", "11", "1", "0"]}[len(shape)]
    return f"Reshape {name} 1 1 {src} {dst} " + " ".join(f"{k}={d}" for k, d in zip(keys, shape))


def read_param(path: Path) -> list[list[str]]:
    lines = path.read_text().split("\n")
    assert lines[0].strip() == "7767517", f"{path}: not an ncnn param"
    return [line.split() for line in lines[2:] if line.strip()]


def merge(pieces: list[Piece], names_in, names_out, shapes, param_out: Path, bin_out: Path):
    """Stitch ncnn pieces in order into one model.

    pnnx names a piece's inputs `in0..` and outputs `out0..` and every other blob by number,
    so each piece's own blobs are prefixed to keep them apart, a piece's `in0` is the previous
    piece's `out0`, and the Input layer that declared it is dropped. Where the two sides of a
    cut disagree on which axis ncnn drops, a Reshape between them makes up the difference.
    The model's own inputs and outputs get their ONNX names back.
    """
    layers, blobs = [], set()
    for k, piece in enumerate(pieces):
        first, final = k == 0, k == len(pieces) - 1
        entry = f"p{k - 1}_out0"
        if not first:
            before, after = pieces[k - 1].squeezed_out[0], piece.squeezed_in[0]
            if before != after:
                shape = ncnn_shape(shapes[k - 1], after)
                layers.append(reshape_layer(f"p{k}_cut", entry, f"p{k}_cut", shape))
                blobs.add(f"p{k}_cut")
                entry = f"p{k}_cut"

        def rename(blob: str) -> str:
            if blob.startswith("in") and blob[2:].isdigit():
                return names_in[int(blob[2:])] if first else entry
            if blob.startswith("out") and blob[3:].isdigit() and final:
                return names_out[int(blob[3:])]
            return f"p{k}_{blob}"

        for fields in read_param(piece.param):
            kind, name, nb, nt = fields[0], fields[1], int(fields[2]), int(fields[3])
            if kind == "Input" and not first:
                continue
            ios = [rename(b) for b in fields[4 : 4 + nb + nt]]
            blobs.update(ios)
            layers.append(" ".join([kind, f"p{k}_{name}", str(nb), str(nt), *ios, *fields[4 + nb + nt :]]))

    unsupported = {l.split()[0] for l in layers if "." in l.split()[0]}
    if unsupported:
        # pnnx leaves what it cannot lower under its torch name, and ncnn refuses to load it.
        raise SystemExit(f"{param_out.name}: pnnx left ops ncnn does not have: {unsupported}")

    param_out.write_text(f"7767517\n{len(layers)} {len(blobs)}\n" + "\n".join(layers) + "\n")
    # The `.bin` is what says a model is done, so it appears whole or not at all.
    partial = bin_out.with_suffix(".bin.part")
    with open(partial, "wb") as out:
        for piece in pieces:
            with open(piece.bin, "rb") as f:
                shutil.copyfileobj(f, out)
    partial.rename(bin_out)


def convert(name: str, onnx_dir: Path, ncnn_dir: Path, work: Path):
    model = onnx.load(onnx_dir / f"{name}.onnx")
    finite_masks(model)
    batch_inputs(model)
    lower_rank(model)
    extractor = Extractor(model)
    io = peel(model, extractor, name, ncnn_dir)
    cuts = cut_points(model, io["inputs"])
    shape = shapes_of(model)
    print(f"  inputs {io['inputs']}, outputs {io['outputs']}, {len(cuts) + 1} piece(s)", flush=True)

    bounds = [io["inputs"], *[[c] for c in cuts], io["outputs"]]
    pieces = []
    for k, (ins, outs) in enumerate(zip(bounds, bounds[1:])):
        path = work / f"piece{k:02}.onnx"
        onnx.save(
            extractor.extract_model(ins, outs),
            path,
            save_as_external_data=True,
            location=path.name + ".data",
        )
        print(f"  pnnx piece {k}: {ins} -> {outs}", flush=True)
        pieces.append(pnnx(path))
        path.unlink()
        Path(f"{path}.data").unlink()
    del extractor, model

    # What a caller needs to feed the model and read it back: the shape each blob has in
    # ncnn, and the ONNX shape to reshape an output to. Written before the `.bin`, which is
    # what says the model is done.
    names_in, names_out = io["inputs"], io["outputs"]
    io["inputs"] = [
        {"name": n, "shape": shape[n], "ncnn_shape": ncnn_shape(shape[n], a)}
        for n, a in zip(names_in, pieces[0].squeezed_in)
    ]
    io["outputs"] = [
        {"name": n, "shape": shape[n], "ncnn_shape": ncnn_shape(shape[n], a)}
        for n, a in zip(names_out, pieces[-1].squeezed_out)
    ]
    (ncnn_dir / f"{name}.json").write_text(json.dumps(io, indent=2) + "\n")
    merge(
        pieces,
        names_in,
        names_out,
        [shape[c] for c in cuts],
        ncnn_dir / f"{name}.ncnn.param",
        ncnn_dir / f"{name}.ncnn.bin",
    )


def download(names: list[str], onnx_dir: Path):
    """Each encoder's graph and its external weights, ~3.4 GB for both.

    The Hub client resumes a partial file and checks what it got against the Hub's digest,
    and one already in `onnx_dir` is not fetched again.
    """
    for name in names:
        for f in (f"{name}.onnx", f"{name}.onnx.data"):
            hf_hub_download(REPO, f, revision=REVISION, local_dir=onnx_dir)


def main():
    data = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data"
    onnx_dir, ncnn_dir, work = data / "onnx", data / "ncnn", data / "ncnn" / "work"
    ncnn_dir.mkdir(parents=True, exist_ok=True)

    todo = [name for name in MODELS if not (ncnn_dir / f"{name}.ncnn.bin").exists()]
    for name in MODELS:
        if name not in todo:
            print(f"{name}: already converted")
    download(todo, onnx_dir)
    hf_hub_download(TOKENIZER_REPO, "tokenizer.json", revision=TOKENIZER_REVISION, local_dir=ncnn_dir)
    for name in todo:
        print(f"{name}: converting", flush=True)
        shutil.rmtree(work, ignore_errors=True)
        work.mkdir(parents=True)
        convert(name, onnx_dir, ncnn_dir, work)
    shutil.rmtree(work, ignore_errors=True)


if __name__ == "__main__":
    main()
