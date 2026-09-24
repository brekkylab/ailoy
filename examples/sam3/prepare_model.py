"""Download the SAM3 checkpoint and convert it into ncnn models.

    uv run prepare_model.py [DATA_DIR]

DATA_DIR is `data/` beside this file by default. The checkpoint goes into `DATA_DIR/checkpoint`,
from a mirror of `facebook/sam3` on the Hugging Face Hub at a pinned revision, and this writes
into `DATA_DIR/ncnn`

* `sam3_PIECE.ncnn.param`, `sam3_PIECE.ncnn.bin` -- each piece below;
* `sam3.json` -- every piece's ncnn input and output blobs, what they are, and the settings the
  steps around the models need;
* `sam3.npz` -- the weights those steps use: the prompt encoder's, the memory bank's
  embeddings, the text token table;
* `tokenizer.json` -- the checkpoint's CLIP tokenizer, for `tokenizers` alone.

SAM3 is one ViT backbone and three heads on it. The detector finds every instance of a text
prompt, or of example boxes. The tracker, SAM 2's mask decoder, segments one object from
points, a box or a mask. The video tracker is the same decoder on features conditioned on a
memory of earlier frames. The pieces:

* `vision` -- the backbone and both of its necks, the detector's and the tracker's: an image to
  the features every head takes. Run once an image or a frame.
* `text` -- CLIP's text encoder, from token embeddings to the detector's prompt features.
* `detector` -- the fusion encoder, the DETR decoder and the mask head: 200 queries' scores,
  boxes and masks for a prompt.
* `geometry` -- the detector's box prompts, as prompt features to go beside the text's.
* `tracker` -- the mask decoder: four masks for a prompt, their predicted IoUs, the object score
  and each mask token's object pointer.
* `mask_prompt` -- a mask prompt, as the dense embedding the tracker takes.
* `memory_encoder` -- a frame's features and its predicted mask, as a memory for later frames.
* `memory_attention` -- a frame's features, conditioned on the memories of earlier frames.

What the transformers implementation does around them -- the prompt encoder's points and boxes,
thresholds, which memories a frame attends to -- is left to the caller, `run_sam3.py`.

**What ncnn needs changed.** Each piece is rewritten from the transformers modules, with their
weights, so that the graph pnnx traces is one ncnn can run:

* no tensor past four dims: window partitions, RoPE's pairwise rotation and the detector's box
  bias are respelled on four, and a batch of one is never a dim of its own;
* no op ncnn lacks: `sign`, strided slices, `einsum`, `repeat_interleave`;
* no bool or int tensors: masks come in as floats, 1 for a valid token and 0 for padding, and
  the text's token lookup is left to the caller;
* what depends on no input -- position encodings, RoPE's angles -- computed here once.

Where a count varies -- the detector's prompt tokens, the tracker's points, the memory
attention's memories -- the piece is traced at two lengths, and ncnn takes any.
"""

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from huggingface_hub import snapshot_download
from torch import nn

# A mirror of `facebook/sam3`, which is gated; the SAM License lets it be redistributed with the
# license, which it is.
REPO = "jetjodh/sam3"
REVISION = "1aa50ce07302cb375f85d8084b68a0fb378b8d85"
FILES = [
    "model.safetensors",
    "config.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "vocab.json",
    "merges.txt",
    "LICENSE",
]

# What a masked attention score becomes: past anything a softmax sees here, so the weight it
# gives is 0 all the same, and inside bf16's range.
MASKED = -1e4

IMAGE = 1008
GRID = 72  # IMAGE / patch size, the backbone's and every head's coarsest grid
TOKENS = 32  # the text's context


def rotation(dim: int) -> torch.Tensor:
    """R with x @ R == rotate_pairwise(x): (x0, x1, ...) -> (-x1, x0, ...)."""
    r = torch.zeros(dim, dim)
    even = torch.arange(0, dim, 2)
    r[even + 1, even] = -1
    r[even, even + 1] = 1
    return r


def rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, turn: torch.Tensor) -> torch.Tensor:
    return x * cos + (x @ turn) * sin


def attention(q, k, v, mask=None):
    return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)


def ones(like: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """Ones to matmul `like` with, of its rank: pnnx takes a leading 1 for the batch and drops
    it, and a matrix of ones that lost it would be a vector to ncnn."""
    return torch.ones(*([1] * (like.dim() - 2)), rows, cols)


def outer(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a [..., X, 1] + b [..., 1, Y] as [..., X, Y].

    ncnn's Vulkan binary ops go wrong where both sides broadcast; a matmul with ones broadcasts
    each alone, and a product with 1 is exact."""
    return a @ ones(a, 1, b.shape[-1]) + ones(b, a.shape[-2], 1) @ b


def sine_table(inv: torch.Tensor):
    """The phase that turns sin into cos on odd features, for `sin(x * inv + phase)`: what
    `stack((sin(a[0::2]), cos(a[1::2])))` interleaves, without its strided slices."""
    return torch.tensor([0.0, math.pi / 2]).repeat(inv.numel() // 2)


# --- The backbone and its necks -------------------------------------------------------------


class Vision(nn.Module):
    """Pixels in 0..255 to the detector's three FPN levels and the tracker's.

    The tracker's first two levels come out through the mask decoder's `conv_s0` and `conv_s1`,
    as transformers precomputes them, and its last as the neck gives it: the image tracker adds
    `no_memory_embedding` to it, the video tracker conditions it on memories first.
    """

    def __init__(self, video):
        super().__init__()
        det = video.detector_model
        vit = det.vision_encoder.backbone
        emb = vit.embeddings
        self.patch = emb.patch_embeddings.projection
        pos = emb._tile_position_embeddings(emb.position_embeddings, GRID, GRID)
        self.register_buffer("pos", pos.detach().reshape(1, GRID, GRID, -1))
        self.pre_norm = vit.layer_norm
        self.layers = vit.layers
        self.heads = vit.config.num_attention_heads
        head_dim = vit.config.hidden_size // self.heads
        self.register_buffer("turn", rotation(head_dim))
        for i, layer in enumerate(self.layers):
            cos, sin = layer.rotary_emb(pos, layer.position_ids)
            self.register_buffer(f"cos{i}", cos.detach())
            self.register_buffer(f"sin{i}", sin.detach())
        self.det_neck = det.vision_encoder.neck.fpn_layers[:3]
        self.trk_neck = video.tracker_neck.fpn_layers[:3]
        self.conv_s0 = video.tracker_model.mask_decoder.conv_s0
        self.conv_s1 = video.tracker_model.mask_decoder.conv_s1

    def block(self, i: int, x: torch.Tensor) -> torch.Tensor:
        layer = self.layers[i]
        c = x.shape[-1]
        h = layer.layer_norm1(x)
        ws = layer.window_size or GRID
        n = GRID // ws
        seq, heads = ws * ws, self.heads
        # [H, W, C] -> [windows, window tokens, C], windows row-major: (i, r, j, c) -> (i, j, r, c)
        h = h.reshape(1, n, ws, n, ws * c).permute(0, 1, 3, 2, 4).reshape(1, n * n, seq, c)
        att = layer.attention
        # windows and heads as one dim, behind the batch of one ncnn drops
        split = lambda t: t.reshape(1, n * n, seq, heads, c // heads).permute(0, 1, 3, 2, 4).reshape(1, n * n * heads, seq, -1)
        q, k, v = split(att.q_proj(h)), split(att.k_proj(h)), split(att.v_proj(h))
        cos, sin = getattr(self, f"cos{i}"), getattr(self, f"sin{i}")
        q, k = rope(q, cos, sin, self.turn), rope(k, cos, sin, self.turn)
        h = attention(q, k, v).reshape(1, n * n, heads, seq, -1).permute(0, 1, 3, 2, 4).reshape(1, n * n, seq, c)
        h = att.o_proj(h)
        h = h.reshape(1, n, n, ws, ws * c).permute(0, 1, 3, 2, 4).reshape(1, GRID, GRID, c)
        x = x + h
        return x + layer.mlp(layer.layer_norm2(x))

    def forward(self, pixels):
        x = pixels / 127.5 - 1.0
        x = self.patch(x).permute(0, 2, 3, 1) + self.pos
        x = self.pre_norm(x)
        for i in range(len(self.layers)):
            x = self.block(i, x)
        x = x.permute(0, 3, 1, 2)
        det = [layer(x) for layer in self.det_neck]
        trk = [layer(x) for layer in self.trk_neck]
        return det[0], det[1], det[2], self.conv_s0(trk[0]), self.conv_s1(trk[1]), trk[2]


# --- The text encoder -----------------------------------------------------------------------


class Text(nn.Module):
    """CLIP's text transformer, from token plus position embeddings to the detector's prompt
    features. The attention is causal alone: past its end of text a prompt's padding changes no
    token before it, and the detector masks the padding out."""

    def __init__(self, video):
        super().__init__()
        det = video.detector_model
        text = det.text_encoder.text_model
        self.layers = text.encoder.layers
        self.final_norm = text.final_layer_norm
        self.projection = det.text_projection
        self.heads = text.config.num_attention_heads
        causal = torch.full((TOKENS, TOKENS), MASKED).triu(1)
        self.register_buffer("causal", causal[None])

    def forward(self, embeds):
        x = embeds
        c = x.shape[-1]
        shape = (1, TOKENS, self.heads, c // self.heads)
        for layer in self.layers:
            h = layer.layer_norm1(x)
            att = layer.self_attn
            q = att.q_proj(h).reshape(shape).permute(0, 2, 1, 3)
            k = att.k_proj(h).reshape(shape).permute(0, 2, 1, 3)
            v = att.v_proj(h).reshape(shape).permute(0, 2, 1, 3)
            h = attention(q, k, v, self.causal).permute(0, 2, 1, 3).reshape(1, TOKENS, c)
            x = x + att.out_proj(h)
            x = x + layer.mlp(layer.layer_norm2(x))
        return self.projection(self.final_norm(x))


# --- The detector ---------------------------------------------------------------------------


def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    return torch.log(x.clamp(min=eps) / (1 - x).clamp(min=eps))


def cxcywh_to_xyxy(b):
    cx, cy, w, h = b[..., 0:1], b[..., 1:2], b[..., 2:3], b[..., 3:4]
    return torch.cat([cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h], dim=-1)


class Detector(nn.Module):
    """FPN levels and a prompt to 200 queries' logits, xyxy boxes and mask logits, the presence
    logit and the semantic segmentation. The prompt is the text's features, with the geometry's
    after them for box prompts, and `valid` says which of its tokens are not padding."""

    def __init__(self, video):
        super().__init__()
        det = video.detector_model
        self.encoder = det.detr_encoder.layers
        dec = det.detr_decoder
        self.decoder = dec
        self.scoring = det.dot_product_scoring
        self.mask_decoder = det.mask_decoder
        pos = det.vision_encoder.neck.position_encoding((1, 256, GRID, GRID), "cpu", torch.float32)
        self.register_buffer("vision_pos", pos.flatten(2).transpose(1, 2).detach())
        coords = torch.arange(GRID, dtype=torch.float32) / GRID
        self.register_buffer("coords", coords.reshape(1, 1, GRID, 1).expand(1, dec.config.num_queries, GRID, 2).contiguous())
        self.register_buffer("vision_rows", torch.ones(1, GRID * GRID, 1))
        self.register_buffer("query_rows", torch.ones(1, dec.config.num_queries + 1, 1))
        # The decoder's box sine embedding: `encode_boxes` as `sin(x * inv + phase)`.
        pe = dec.position_encoding
        n = pe.num_position_features
        dim_t = pe.temperature ** (2 * torch.div(torch.arange(n), 2, rounding_mode="floor") / n)
        self.register_buffer("box_inv", (pe.scale / dim_t).float().reshape(1, 1, 1, n))
        self.register_buffer("box_phase", sine_table(dim_t))

    def encode_boxes(self, boxes):
        # [1, Q, 4] -> [1, Q, 4, n], then (y, x, w, h) as `encode_boxes` orders them
        a = torch.sin(boxes.unsqueeze(-1) @ self.box_inv + self.box_phase)
        q = boxes.shape[1]
        return torch.cat([a[:, :, 1], a[:, :, 0], a[:, :, 2], a[:, :, 3]], dim=-1).reshape(1, q, -1)

    def rpb(self, boxes):
        """`_get_rpb_matrix`, on four dims and without `sign`: [1, heads, Q, H*W]."""
        dec = self.decoder
        q = boxes.shape[1]
        xyxy = cxcywh_to_xyxy(boxes).reshape(1, q, 1, 4)
        spread = ones(xyxy, GRID, 1)  # each box's edges down the grid, [1, Q, H, 2]
        dy = self.coords - spread @ torch.cat([xyxy[..., 1:2], xyxy[..., 3:4]], dim=-1)
        dx = self.coords - spread @ torch.cat([xyxy[..., 0:1], xyxy[..., 2:3]], dim=-1)

        def log_scale(d):
            d = d * 8
            # sign(d) * log2(|d| + 1) / log2(8); where |d| < 1e-6 the log is under 2e-6 anyway
            return (d * 1e6).clamp(-1, 1) * torch.log(d.abs() + 1.0) / math.log(8)

        ey = dec.box_rpb_embed_y(log_scale(dy)).permute(0, 3, 1, 2)  # [1, heads, Q, H]
        ex = dec.box_rpb_embed_x(log_scale(dx)).permute(0, 3, 1, 2)
        heads = ey.shape[1]
        m = outer(ey.unsqueeze(-1), ex.unsqueeze(-2)).reshape(1, heads, q, GRID * GRID)
        # the presence token attends to every location alike
        return torch.cat([torch.zeros(1, heads, 1, GRID * GRID), m], dim=2)

    def forward(self, fpn0, fpn1, fpn2, prompt, valid):
        # ncnn's attention reads a mask row by row, one row a query: spelled out in full
        row = ((1 - valid) * MASKED).unsqueeze(1)
        mask = self.vision_rows @ row
        queries_mask = self.query_rows @ row
        vision = fpn2.flatten(2).transpose(1, 2)
        for layer in self.encoder:
            vision = layer(vision, prompt_feats=prompt, vision_pos_encoding=self.vision_pos, prompt_cross_attn_mask=mask)

        dec = self.decoder
        boxes = dec.reference_points.weight.unsqueeze(0).sigmoid()
        hidden = torch.cat([dec.presence_token.weight, dec.query_embed.weight], dim=0).unsqueeze(0)
        for layer in dec.layers:
            query_pos = dec.ref_point_head(self.encode_boxes(boxes))
            hidden = layer(
                hidden,
                query_pos=query_pos,
                text_features=prompt,
                vision_features=vision,
                vision_pos_encoding=self.vision_pos,
                text_cross_attn_mask=queries_mask,
                vision_cross_attn_mask=self.rpb(boxes),
            )
            queries = dec.output_layer_norm(hidden[:, 1:])
            boxes = (dec.box_head(queries) + inverse_sigmoid(boxes)).sigmoid()
        presence = dec.presence_head(dec.presence_layer_norm(hidden[:, :1])).squeeze(-1)
        presence = presence.clamp(-dec.clamp_presence_logit_max_val, dec.clamp_presence_logit_max_val)

        s = self.scoring
        text = s.text_mlp_out_norm(s.text_mlp(prompt) + prompt)
        w = valid.unsqueeze(-1)
        pooled = (text * w).sum(dim=1) / w.sum(dim=1).clamp(min=1.0)
        logits = (s.query_proj(queries) @ s.text_proj(pooled).unsqueeze(-1)).squeeze(-1) * s.scale
        logits = logits.clamp(-s.clamp_max_val, s.clamp_max_val)

        md = self.mask_decoder
        attn, _ = md.prompt_cross_attn(query=md.prompt_cross_attn_norm(vision), key=prompt, value=prompt, attention_mask=mask)
        vision = vision + attn
        feats = [fpn0, fpn1, vision.transpose(1, 2).reshape(1, -1, GRID, GRID)]
        pixel = md.pixel_decoder(feats)
        inst = md.instance_projection(pixel)
        q = queries.shape[1]
        masks = (md.mask_embedder(queries) @ inst.flatten(2)).reshape(1, q, pixel.shape[-2], pixel.shape[-1])
        semantic = md.semantic_projection(pixel)
        return logits, cxcywh_to_xyxy(boxes), presence, masks, semantic


class Geometry(nn.Module):
    """The detector's box prompts to the prompt features that go after the text's.

    The caller gives each box as cxcywh in 0..1, with the sine embedding of its center and its
    size (`_encode_box_coordinates`), its label's embedding, and the features ROI-aligned from
    the layer-normed last FPN level, flattened: a 7x7 conv over a 7x7 input is a linear layer.
    """

    def __init__(self, video):
        super().__init__()
        det = video.detector_model
        self.g = g = det.geometry_encoder
        w = g.boxes_pool_project.weight
        self.pool = nn.Linear(w.shape[1] * w.shape[2] * w.shape[3], w.shape[0])
        with torch.no_grad():
            self.pool.weight.copy_(w.reshape(w.shape[0], -1))
            self.pool.bias.copy_(g.boxes_pool_project.bias)
        pos = det.vision_encoder.neck.position_encoding((1, 256, GRID, GRID), "cpu", torch.float32)
        self.register_buffer("vision_pos", pos.flatten(2).transpose(1, 2).detach())

    def forward(self, boxes, box_pos, labels, pooled, fpn2):
        g = self.g
        x = g.boxes_direct_project(boxes) + self.pool(pooled) + g.boxes_pos_enc_project(box_pos) + labels
        x = torch.cat([x, g.cls_embed.weight.unsqueeze(0)], dim=1)
        x = g.prompt_layer_norm(g.final_proj(x))
        vision = fpn2.flatten(2).transpose(1, 2)
        for layer in g.layers:
            x = layer(prompt_feats=x, vision_feats=vision, vision_pos_encoding=self.vision_pos, prompt_mask=None)
        return g.output_layer_norm(x)


# --- The tracker ----------------------------------------------------------------------------


class Tracker(nn.Module):
    """SAM 2's mask decoder: the last-level features, the dense prompt embedding, the first two
    levels and the sparse prompt's point embeddings, to all four masks' logits, their IoUs, the
    object score logit, and the object pointer each mask token makes. Which mask to take is the
    caller's, as `_dynamic_multimask_via_stability` and multimask output decide it."""

    def __init__(self, video):
        super().__init__()
        t = video.tracker_model
        self.d = t.mask_decoder
        self.pointer = t.object_pointer_proj
        self.register_buffer("image_pe", t.get_image_wide_positional_embeddings().detach())

    def forward(self, pix, dense, s0, s1, sparse):
        d = self.d
        n = d.num_mask_tokens
        out = torch.cat([d.obj_score_token.weight, d.iou_token.weight, d.mask_tokens.weight], dim=0).unsqueeze(0)
        tokens = torch.cat([out, sparse], dim=1).unsqueeze(1)
        q, k = d.transformer(
            point_embeddings=tokens,
            image_embeddings=pix + dense,
            image_positional_embeddings=self.image_pe,
            attention_similarity=None,
        )
        q = q.squeeze(1)
        mask_tokens = q[:, 2 : 2 + n]
        k = k.transpose(2, 3).reshape(1, -1, GRID, GRID)
        up = d.activation(d.upscale_layer_norm(d.upscale_conv1(k) + s1))
        up = d.activation(d.upscale_conv2(up) + s0)
        hyper = torch.cat([d.output_hypernetworks_mlps[i](mask_tokens[:, i : i + 1]) for i in range(n)], dim=1)
        size = up.shape[-1]
        masks = (hyper @ up.flatten(2)).reshape(1, n, size, size)
        iou = d.iou_prediction_head(q[:, 1])
        score = d.pred_obj_score_head(q[:, 0])
        return masks, iou, score, self.pointer(mask_tokens)


class MaskPrompt(nn.Module):
    """A mask prompt at 4x the grid, as logits, to the dense embedding the tracker takes."""

    def __init__(self, video):
        super().__init__()
        self.embed = video.tracker_model.prompt_encoder.mask_embed

    def forward(self, mask):
        return self.embed(mask)


class MemoryEncoder(nn.Module):
    """A frame's tracker features and its mask, already at 16x the grid and scaled as the
    memory takes it, to the memory's features. Their position encoding is the same for every
    frame and is in `sam3.npz`."""

    def __init__(self, video):
        super().__init__()
        self.m = video.tracker_model.memory_encoder

    def forward(self, pix, mask):
        m = self.m
        x = m.feature_projection(pix) + m.mask_downsampler(mask)
        return m.projection(m.memory_fuser(x))


class MemoryAttention(nn.Module):
    """A frame's tracker features, conditioned on memories: the spatial memories of earlier
    frames, with their position encodings, one frame's after another, and the object pointers,
    with theirs. The spatial memories' keys are rotated as the frame's queries are, frame by
    frame; the pointers' are not."""

    def __init__(self, video):
        super().__init__()
        t = video.tracker_model
        ma = t.memory_attention
        self.layers = ma.layers
        self.norm = ma.layer_norm
        pos = video.tracker_neck.position_encoding((1, 256, GRID, GRID), "cpu", torch.float32)
        self.register_buffer("vision_pos", pos.flatten(2).transpose(1, 2).detach())
        cos, sin = ma.rotary_emb(pos, ma.position_ids)
        self.register_buffer("cos", cos.detach())
        self.register_buffer("sin", sin.detach())
        self.register_buffer("turn", rotation(cos.shape[-1]))

    def turned(self, x):
        """RoPE on each frame's tokens alike: [1, frames * H*W, C]."""
        c = x.shape[-1]
        return rope(x.reshape(1, -1, GRID * GRID, c), self.cos, self.sin, self.turn).reshape(1, -1, c)

    def forward(self, current, memory, memory_pos, pointers, pointers_pos):
        x = current + 0.1 * self.vision_pos
        for layer in self.layers:
            h = layer.layer_norm1(x)
            att = layer.self_attn
            q = self.turned(att.q_proj(h))
            k = self.turned(att.k_proj(h))
            h = attention(q.unsqueeze(1), k.unsqueeze(1), att.v_proj(h).unsqueeze(1)).squeeze(1)
            x = x + att.o_proj(h)

            h = layer.layer_norm2(x)
            att = layer.cross_attn_image
            q = self.turned(att.q_proj(h))
            k = torch.cat([self.turned(att.k_proj(memory + memory_pos)), att.k_proj(pointers + pointers_pos)], dim=1)
            v = torch.cat([att.v_proj(memory), att.v_proj(pointers)], dim=1)
            h = attention(q.unsqueeze(1), k.unsqueeze(1), v.unsqueeze(1)).squeeze(1)
            x = x + att.o_proj(h)

            h = layer.layer_norm3(x)
            x = x + layer.linear2(layer.activation(layer.linear1(h)))
        return self.norm(x)


# --- The pieces, as they are traced -------------------------------------------------------


def pieces(video) -> dict:
    """Each piece: its module, what its inputs and outputs are, and example inputs at two
    lengths where a length varies. Shapes are with the batch of one ncnn drops."""
    g = torch.Generator().manual_seed(0)
    r = lambda *s: torch.randn(*s, generator=g)
    fpn = [r(1, 256, 288, 288), r(1, 256, 144, 144), r(1, 256, GRID, GRID)]
    trk = [r(1, 32, 288, 288), r(1, 64, 144, 144), r(1, 256, GRID, GRID)]
    hw = GRID * GRID

    def prompt(n):
        return r(1, n, 256), torch.cat([torch.ones(1, n - 5), torch.zeros(1, 5)], dim=1)

    def boxes(n):
        return r(1, n, 4).sigmoid(), r(1, n, 258), r(1, n, 256), r(1, n, 256 * 49), fpn[2]

    def memory(frames, pointers):
        return r(1, hw, 256), r(1, frames * hw, 64), r(1, frames * hw, 64), r(1, pointers, 64), r(1, pointers, 64)

    return {
        "vision": dict(
            module=Vision(video),
            inputs=[("pixels", "image as RGB floats in 0..255, channels first, at IMAGE x IMAGE")],
            outputs=["det_fpn0", "det_fpn1", "det_fpn2", "trk_s0", "trk_s1", "trk_fpn2"],
            example=(torch.rand(1, 3, IMAGE, IMAGE, generator=g) * 255,),
        ),
        "text": dict(
            module=Text(video),
            inputs=[("embeds", "token embeddings plus position embeddings, [TOKENS, 1024]")],
            outputs=["features"],
            example=(r(1, TOKENS, 1024),),
        ),
        "detector": dict(
            module=Detector(video),
            inputs=[
                ("fpn0", "det_fpn0"),
                ("fpn1", "det_fpn1"),
                ("fpn2", "det_fpn2"),
                ("prompt", "prompt features: the text's, then the geometry's, [P, 256]"),
                ("valid", "1 for each prompt token that is not padding, else 0, [P]"),
            ],
            outputs=["logits", "boxes", "presence", "masks", "semantic"],
            example=(*fpn, *prompt(TOKENS)),
            example2=(*fpn, *prompt(TOKENS + 3)),
        ),
        "geometry": dict(
            module=Geometry(video),
            inputs=[
                ("boxes", "cxcywh in 0..1, [N, 4]"),
                ("box_pos", "sine embedding of each box's center, then its h and w, [N, 258]"),
                ("labels", "each box's label embedding, label_embed[1] positive, [0] negative, [N, 256]"),
                ("pooled", "roi_align of the layer-normed det_fpn2, flattened, [N, 256 * 49]"),
                ("fpn2", "det_fpn2"),
            ],
            outputs=["prompt"],
            example=boxes(1),
            example2=boxes(2),
        ),
        "tracker": dict(
            module=Tracker(video),
            inputs=[
                ("pix", "trk_fpn2, plus no_memory_embedding or conditioned on memories"),
                ("dense", "the dense prompt embedding, [256, GRID, GRID]"),
                ("s0", "trk_s0"),
                ("s1", "trk_s1"),
                ("sparse", "the points' and boxes' embeddings, [N, 256]"),
            ],
            outputs=["masks", "iou", "score", "pointers"],
            example=(trk[2], r(1, 256, GRID, GRID), trk[0], trk[1], r(1, 2, 256)),
            example2=(trk[2], r(1, 256, GRID, GRID), trk[0], trk[1], r(1, 3, 256)),
        ),
        "mask_prompt": dict(
            module=MaskPrompt(video),
            inputs=[("mask", "mask logits at 4 * GRID, [1, 288, 288]")],
            outputs=["dense"],
            example=(r(1, 1, 4 * GRID, 4 * GRID),),
        ),
        "memory_encoder": dict(
            module=MemoryEncoder(video),
            inputs=[("pix", "trk_fpn2"), ("mask", "the mask as the memory takes it, [1, 1152, 1152]")],
            outputs=["features"],
            example=(trk[2], r(1, 1, 16 * GRID, 16 * GRID)),
        ),
        "memory_attention": dict(
            module=MemoryAttention(video),
            inputs=[
                ("current", "trk_fpn2 as tokens, [H*W, 256]"),
                ("memory", "the spatial memories, frame after frame, [F*H*W, 64]"),
                ("memory_pos", "their position encodings, temporal included, [F*H*W, 64]"),
                ("pointers", "the object pointers, [P, 64]"),
                ("pointers_pos", "their temporal position encodings, [P, 64]"),
            ],
            outputs=["conditioned"],
            example=memory(1, 4),
            example2=memory(2, 8),
        ),
    }


def export(name: str, spec: dict, ncnn_dir: Path, work: Path) -> dict:
    import pnnx

    stem = f"sam3_{name}"
    with torch.no_grad():
        outs = spec["module"](*spec["example"])
        pnnx.export(
            spec["module"],
            str(work / f"{stem}.pt"),
            spec["example"],
            spec.get("example2"),
            ncnnparam=str(ncnn_dir / f"{stem}.ncnn.param"),
            ncnnbin=str(work / f"{stem}.ncnn.bin"),
            ncnnpy=str(work / f"{stem}_ncnn.py"),
            pnnxparam=str(work / f"{stem}.pnnx.param"),
            pnnxbin=str(work / f"{stem}.pnnx.bin"),
            pnnxpy=str(work / f"{stem}_pnnx.py"),
            pnnxonnx=str(work / f"{stem}.pnnx.onnx"),
            fp16=True,
        )
    dynamic = spec.get("example2")
    io = {
        "inputs": [
            {
                "name": f"in{i}",
                "role": role,
                "about": about,
                "shape": list(x.shape[1:]) if dynamic is None or x.shape == dynamic[i].shape else None,
            }
            for i, ((role, about), x) in enumerate(zip(spec["inputs"], spec["example"]))
        ],
        "outputs": [
            {"name": f"out{i}", "role": role, "shape": list(o.shape[1:]) if dynamic is None else None}
            for i, (role, o) in enumerate(zip(spec["outputs"], outs))
        ],
    }
    # The `.bin` is what says a piece is done, so it appears once its param is in place.
    (work / f"{stem}.ncnn.bin").rename(ncnn_dir / f"{stem}.ncnn.bin")
    return io


# --- What the steps around the pieces need --------------------------------------------------


def constants(video) -> dict:
    det, t = video.detector_model, video.tracker_model
    text = det.text_encoder.text_model.embeddings
    pe = t.prompt_encoder
    _, mem_pos = t.memory_encoder(torch.zeros(1, 256, GRID, GRID), torch.zeros(1, 1, 16 * GRID, 16 * GRID))
    a = lambda x: x.detach().float().numpy()
    return {
        # The token table is what the text piece's lookup was; half precision, as the pieces'
        # weights are.
        "text_tokens": a(text.token_embedding.weight).astype(np.float16),
        "text_positions": a(text.position_embedding.weight),
        "geometry_norm_weight": a(det.geometry_encoder.vision_layer_norm.weight),
        "geometry_norm_bias": a(det.geometry_encoder.vision_layer_norm.bias),
        "geometry_label_embed": a(det.geometry_encoder.label_embed.weight),
        "prompt_gaussian": a(pe.shared_embedding.positional_embedding),
        "point_embed": a(pe.point_embed.weight),
        "not_a_point_embed": a(pe.not_a_point_embed.weight[0]),
        "no_mask_embed": a(pe.no_mask_embed.weight[0]),
        "no_memory_embedding": a(t.no_memory_embedding.reshape(-1)),
        "no_object_pointer": a(t.no_object_pointer.reshape(-1)),
        "occlusion_embedding": a(t.occlusion_spatial_embedding_parameter.reshape(-1)),
        "memory_temporal": a(t.memory_temporal_positional_encoding.reshape(t.num_maskmem, -1)),
        "memory_pos": a(mem_pos.flatten(2)[0].T),
        "pointer_pos_weight": a(t.temporal_positional_encoding_projection_layer.weight),
        "pointer_pos_bias": a(t.temporal_positional_encoding_projection_layer.bias),
        "mask_downsample_weight": a(t.mask_downsample.weight.reshape(4, 4)),
        "mask_downsample_bias": a(t.mask_downsample.bias),
    }


def settings(video) -> dict:
    c = video.config
    tc = c.tracker_config
    md = tc.mask_decoder_config
    return {
        "image_size": IMAGE,
        "grid": GRID,
        "tokens": TOKENS,
        "masked": MASKED,
        "geometry_norm_eps": video.detector_model.geometry_encoder.vision_layer_norm.eps,
        "detector": {
            "score_threshold": c.score_threshold_detection,
            "nms_threshold": c.det_nms_thresh,
        },
        "tracker": {
            "multimask_output_in_sam": tc.multimask_output_in_sam,
            "multimask_output_for_tracking": tc.multimask_output_for_tracking,
            "multimask_min_pt_num": tc.multimask_min_pt_num,
            "multimask_max_pt_num": tc.multimask_max_pt_num,
            "stability_delta": md.dynamic_multimask_stability_delta,
            "stability_thresh": md.dynamic_multimask_stability_thresh,
            "num_maskmem": tc.num_maskmem,
            "max_cond_frame_num": tc.max_cond_frame_num,
            "max_object_pointers": tc.max_object_pointers_in_encoder,
            "sigmoid_scale_for_mem_enc": tc.sigmoid_scale_for_mem_enc,
            "sigmoid_bias_for_mem_enc": tc.sigmoid_bias_for_mem_enc,
            "mem_dim": tc.memory_encoder_output_channels,
        },
    }


# --- Checking ncnn on the host against PyTorch ----------------------------------------------


def cos(a: np.ndarray, b: np.ndarray) -> float:
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))


def check(ncnn_dir: Path, specs: dict, ios: dict):
    """Each piece, on its example inputs, against the PyTorch module it was traced from."""
    import ncnn

    for name, spec in specs.items():
        net = ncnn.Net()
        net.opt.use_vulkan_compute = ncnn.get_gpu_count() > 0
        net.opt.use_fp16_storage = False
        net.opt.use_fp16_packed = False
        net.opt.use_fp16_arithmetic = False
        net.load_param(str(ncnn_dir / f"sam3_{name}.ncnn.param"))
        net.load_model(str(ncnn_dir / f"sam3_{name}.ncnn.bin"))
        for example in [spec["example"], spec.get("example2")]:
            if example is None:
                continue
            with torch.no_grad():
                want = spec["module"](*example)
            ex = net.create_extractor()
            for inp, x in zip(ios[name]["inputs"], example):
                a = np.ascontiguousarray(x.numpy()[0], np.float32)
                ex.input(inp["name"], ncnn.Mat(a).clone())
            line = []
            for out, w in zip(ios[name]["outputs"], want):
                ret, m = ex.extract(out["name"])
                assert ret == 0, f"{name}: extract {out['name']} failed: {ret}"
                line.append(f"{out['role']} {cos(np.array(m), w.numpy()):.5f}")
            print(f"  {name}: " + ", ".join(line), flush=True)


def download(checkpoint: Path):
    """The checkpoint at the repo root; the Hub client skips what is already there."""
    snapshot_download(REPO, revision=REVISION, allow_patterns=FILES, local_dir=checkpoint)


def main():
    from transformers import Sam3VideoModel

    data = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(__file__).parent / "data"
    checkpoint, ncnn_dir = data / "checkpoint", data / "ncnn"
    work = ncnn_dir / "work"
    ncnn_dir.mkdir(parents=True, exist_ok=True)
    names = ["vision", "text", "detector", "geometry", "tracker", "mask_prompt", "memory_encoder", "memory_attention"]
    todo = [n for n in names if not (ncnn_dir / f"sam3_{n}.ncnn.bin").exists()]
    if not todo and (ncnn_dir / "sam3.npz").exists():
        print("sam3: already converted")
        return
    download(checkpoint)
    video = Sam3VideoModel.from_pretrained(checkpoint, attn_implementation="sdpa").eval()
    specs = pieces(video)
    assert list(specs) == names

    shutil.rmtree(work, ignore_errors=True)
    work.mkdir(parents=True)
    ios = json.loads((ncnn_dir / "sam3.json").read_text())["pieces"] if (ncnn_dir / "sam3.json").exists() else {}
    for name in specs:
        if name in todo or name not in ios:
            print(f"sam3_{name}: converting", flush=True)
            ios[name] = export(name, specs[name], ncnn_dir.resolve(), work.resolve())
            (ncnn_dir / "sam3.json").write_text(json.dumps({"pieces": ios}, indent=2) + "\n")
    shutil.rmtree(work, ignore_errors=True)

    np.savez(ncnn_dir / "sam3.npz", **constants(video))
    shutil.copy(checkpoint / "tokenizer.json", ncnn_dir / "tokenizer.json")
    (ncnn_dir / "sam3.json").write_text(json.dumps({"pieces": ios, **settings(video)}, indent=2) + "\n")
    print("checking ncnn on the host against PyTorch", flush=True)
    check(ncnn_dir, specs, ios)


if __name__ == "__main__":
    main()
