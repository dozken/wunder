"""Hand-built ONNX graph for the GRU predictor: no glue ops.

torch.onnx.export wraps each GRU in Transpose/Squeeze/Slice/Concat and each
Gemm in Reshapes; at one row per call that glue costs more than the GRU
itself. This builds the same function with onnx.helper: standardise (Mul,
Add, Clip) -> Gemm -> Gelu -> LayerNorm -> Unsqueeze -> GRU x L -> Reshape ->
Concat -> LayerNorm -> Gemm -> Gelu -> Gemm. One state input/output per GRU
layer, named state_0.. (solution.py treats every non-feature input as state).

    python src/export_slim.py runs/<tag>/best.pt src/<name>.onnx
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnx
import torch
from onnx import TensorProto, helper, numpy_helper

sys.path.insert(0, str(Path(__file__).resolve().parent))
from export import load_checkpoint  # noqa: E402

OPSET = 17


def gru_weights(rnn: torch.nn.GRU, layer: int):
    """PyTorch gate order (r, z, n) -> ONNX (z, r, h); linear_before_reset=1 matches torch."""
    def reorder(w):
        r, z, n = w.chunk(3, dim=0)
        return torch.cat([z, r, n], dim=0)
    w_ih = reorder(getattr(rnn, f"weight_ih_l{layer}").detach())
    w_hh = reorder(getattr(rnn, f"weight_hh_l{layer}").detach())
    b_ih = reorder(getattr(rnn, f"bias_ih_l{layer}").detach())
    b_hh = reorder(getattr(rnn, f"bias_hh_l{layer}").detach())
    W = w_ih.unsqueeze(0).numpy()                       # (1, 3H, in)
    R = w_hh.unsqueeze(0).numpy()                       # (1, 3H, H)
    B = torch.cat([b_ih, b_hh]).unsqueeze(0).numpy()    # (1, 6H)
    return W.astype(np.float32), R.astype(np.float32), B.astype(np.float32)


def build(model) -> onnx.ModelProto:
    cfg = model.cfg
    if cfg.rnn != "gru" or cfg.proj <= 0:
        raise ValueError("slim export covers GRU with an input projection")
    H, P, L = cfg.hidden, cfg.proj, cfg.layers
    inits, nodes = [], []

    def const(name, arr):
        inits.append(numpy_helper.from_array(np.ascontiguousarray(arr, dtype=np.float32), name))
        return name

    def iconst(name, arr):
        inits.append(numpy_helper.from_array(np.asarray(arr, dtype=np.int64), name))
        return name

    inv_std = (1.0 / model.std).numpy()
    const("inv_std", inv_std.reshape(1, 112))
    const("neg_mean_scaled", (-(model.mean / model.std)).numpy().reshape(1, 112))
    const("clip_lo", np.array(-cfg.input_clip, dtype=np.float32))
    const("clip_hi", np.array(cfg.input_clip, dtype=np.float32))
    nodes.append(helper.make_node("Mul", ["features", "inv_std"], ["x_scaled"]))
    nodes.append(helper.make_node("Add", ["x_scaled", "neg_mean_scaled"], ["x_shift"]))
    nodes.append(helper.make_node("Clip", ["x_shift", "clip_lo", "clip_hi"], ["x"]))
    gemm_in = "x"
    if cfg.diff:
        # previous standardised row rides in `prev`; the first row of a sequence
        # sees prev = 0 exactly like the torch model's zero-initialised carry
        nodes.append(helper.make_node("Sub", ["x", "prev"], ["dx"]))
        nodes.append(helper.make_node("Concat", ["x", "dx"], ["x_cat"], axis=1))
        nodes.append(helper.make_node("Identity", ["x"], ["next_prev"]))
        gemm_in = "x_cat"

    lin, ln = model.inproj[0], model.inproj[2]
    const("W_in", lin.weight.detach().numpy())                       # (P, 112)
    const("b_in", lin.bias.detach().numpy().reshape(1, P))
    nodes.append(helper.make_node("Gemm", [gemm_in, "W_in", "b_in"], ["z_pre"], transB=1))
    nodes.append(helper.make_node("Gelu", ["z_pre"], ["z_act"], domain="com.microsoft"))
    const("ln_in_w", ln.weight.detach().numpy()); const("ln_in_b", ln.bias.detach().numpy())
    nodes.append(helper.make_node("LayerNormalization", ["z_act", "ln_in_w", "ln_in_b"], ["z"],
                                  axis=-1, epsilon=float(ln.eps)))
    iconst("axes0", [0])
    nodes.append(helper.make_node("Unsqueeze", ["z", "axes0"], ["z3"]))                 # (1, 1, P)

    x_in = "z3"
    state_in, state_out = [], []
    for layer in range(L):
        W, R, B = gru_weights(model.rnn, layer)
        const(f"W_{layer}", W); const(f"R_{layer}", R); const(f"B_{layer}", B)
        state_in.append(f"state_{layer}")
        state_out.append(f"next_state_{layer}")
        nodes.append(helper.make_node(
            "GRU", [x_in, f"W_{layer}", f"R_{layer}", f"B_{layer}", "", f"state_{layer}"],
            ["", f"next_state_{layer}"], hidden_size=H, linear_before_reset=1, direction="forward"))
        x_in = f"next_state_{layer}"                                                   # (1, 1, H) feeds the next layer

    iconst("shape_h", [1, H])
    nodes.append(helper.make_node("Reshape", [x_in, "shape_h"], ["h2"]))
    nodes.append(helper.make_node("Concat", ["h2", "z"], ["cat"], axis=1))              # (1, H + P)
    hn = model.head_norm
    const("ln_head_w", hn.weight.detach().numpy()); const("ln_head_b", hn.bias.detach().numpy())
    nodes.append(helper.make_node("LayerNormalization", ["cat", "ln_head_w", "ln_head_b"], ["cat_n"],
                                  axis=-1, epsilon=float(hn.eps)))
    h1, h2 = model.head[0], model.head[3]
    const("W_h1", h1.weight.detach().numpy()); const("b_h1", h1.bias.detach().numpy().reshape(1, -1))
    const("W_h2", h2.weight.detach().numpy()[:2]); const("b_h2", h2.bias.detach().numpy()[:2].reshape(1, -1))  # drop any aux channel
    nodes.append(helper.make_node("Gemm", ["cat_n", "W_h1", "b_h1"], ["head_pre"], transB=1))
    nodes.append(helper.make_node("Gelu", ["head_pre"], ["head_act"], domain="com.microsoft"))
    nodes.append(helper.make_node("Gemm", ["head_act", "W_h2", "b_h2"], ["prediction"], transB=1))

    inputs = [helper.make_tensor_value_info("features", TensorProto.FLOAT, [1, 112])]
    inputs += [helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, 1, H]) for n in state_in]
    outputs = [helper.make_tensor_value_info("prediction", TensorProto.FLOAT, [1, 2])]
    outputs += [helper.make_tensor_value_info(n, TensorProto.FLOAT, [1, 1, H]) for n in state_out]
    if cfg.diff:
        inputs.append(helper.make_tensor_value_info("prev", TensorProto.FLOAT, [1, 112]))
        outputs.append(helper.make_tensor_value_info("next_prev", TensorProto.FLOAT, [1, 112]))
    graph = helper.make_graph(nodes, "predictor_slim", inputs, outputs, initializer=inits)
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", OPSET), helper.make_opsetid("com.microsoft", 1)])
    m.ir_version = 9
    return m


@torch.no_grad()
def verify(model, path: Path, rows: int = 3000, seed: int = 0) -> float:
    import onnxruntime as ort
    rng = np.random.default_rng(seed)
    x = (rng.standard_normal((rows, 112)) * 3).astype(np.float32)
    x[:, 104:] *= 4                                                     # exercise the clip
    ref, _ = model(torch.from_numpy(x).unsqueeze(0), model.initial_state(1))
    ref = ref.numpy()[0][:, :2]
    s = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    state_inputs = [i for i in s.get_inputs() if i.name != "features"]
    states = {i.name: np.zeros(i.shape, np.float32) for i in state_inputs}
    out = np.zeros((rows, 2), np.float32)
    for t in range(rows):
        res = s.run(None, {"features": x[t:t + 1], **states})
        out[t] = res[0][0]
        for k, i in enumerate(state_inputs):          # outputs follow input order after `prediction`
            states[i.name] = res[1 + k]
    return float(np.abs(out - ref).max())


def bench(path: Path, rows: int = 20000) -> float:
    import onnxruntime as ort
    opts = ort.SessionOptions(); opts.intra_op_num_threads = 1; opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.add_session_config_entry("session.intra_op.allow_spinning", "0")
    s = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    ins = s.get_inputs()
    feed = {i.name: np.zeros([d for d in i.shape], np.float32) for i in ins}
    state_names = [i.name for i in ins if i.name != "features"]
    t = time.perf_counter()
    for _ in range(rows):
        res = s.run(None, feed)
        for k, n in enumerate(state_names):
            feed[n] = res[1 + k]
    return (time.perf_counter() - t) / rows * 1e6


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()
    model, ckpt = load_checkpoint(args.checkpoint)
    m = build(model)
    onnx.checker.check_model(m)
    onnx.save(m, str(args.out))
    print(f"wrote {args.out} ({args.out.stat().st_size / 1e6:.2f} MB, {len(m.graph.node)} nodes) config={ckpt['config']}")
    diff = verify(model, args.out)
    print(f"torch vs slim onnx max abs diff over 3000 rows: {diff:.2e}")
    if diff > 1e-3:
        print("MISMATCH", file=sys.stderr)
        return 1
    print(f"latency {bench(args.out):.1f} us/row (this machine, current load)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
