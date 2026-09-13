"""Rewrite the ONNX GRU/LSTM ops of an exported graph as explicit one-step cells.

    python src/unroll.py src/model_a.onnx src/model_a_x86.onnx [--quant dynamic|nbits8|nbits4]

Why: at batch 1 and sequence length 1 ONNX Runtime's fused GRU/LSTM kernels
are slow on x86 (their per-call overhead dominates, and the int8 LSTM kernel
that helps on Apple silicon does nothing there). Unrolling one step into
MatMul + elementwise ops is numerically identical in fp32 and, more
importantly, lets the recurrent MatMuls be quantised: dynamic int8 MatMuls
run 1.6-3x faster than the fused kernels on a cloud Xeon because the weight
bytes, not the FLOPs, are what a single step pays for.

The rewritten graph keeps the same inputs/outputs, so solution.py is unchanged.
Run with --check to confirm the fp32 rewrite matches the source graph and to
measure how far the quantised variant drifts over a long synthetic sequence.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import helper as H, numpy_helper as NH


def _inits(model):
    return {i.name: i for i in model.graph.initializer}


def _const_nodes(model):
    out = {}
    for n in model.graph.node:
        if n.op_type == "Constant":
            out[n.output[0]] = NH.to_array(n.attribute[0].t)
    return out


def _fetch(model, name):
    inits = _inits(model)
    if name in inits:
        return NH.to_array(inits[name])
    consts = _const_nodes(model)
    if name in consts:
        return consts[name]
    raise KeyError(f"{name} is neither an initializer nor a Constant; weights must be static")


def _consumers(model, name):
    return [n for n in model.graph.node if name in n.input]


def _rewire(model, old, new):
    for n in model.graph.node:
        for i, inp in enumerate(n.input):
            if inp == old:
                n.input[i] = new
    for o in model.graph.output:
        if o.name == old:
            o.name = new


def fold_standardise(model: onnx.ModelProto) -> onnx.ModelProto:
    """Fold features -> Mul(inv_std) -> Add(shift) -> Clip -> Gemm(W, b, transB=1)
    into a single Gemm. The Clip is dropped: it sits at +-10 standardised units
    and the features are bounded at +-5.2 by construction, so it never fires."""
    model = onnx.ModelProto.FromString(model.SerializeToString())
    nodes = list(model.graph.node)
    try:
        mul = next(n for n in nodes if n.op_type == "Mul" and n.input[0] == "features")
        add = next(n for n in nodes if n.op_type == "Add" and n.input[0] == mul.output[0])
        clip = next(n for n in nodes if n.op_type == "Clip" and n.input[0] == add.output[0])
        gemm = next(n for n in nodes if n.op_type == "Gemm" and n.input[0] == clip.output[0])
    except StopIteration:
        return model
    attrs = {a.name: H.get_attribute_value(a) for a in gemm.attribute}
    if attrs.get("transB", 0) != 1:
        return model
    inv_std = _fetch(model, mul.input[1]).reshape(-1)
    shift = _fetch(model, add.input[1]).reshape(-1)
    W = _fetch(model, gemm.input[1])                     # (P, 112)
    b = _fetch(model, gemm.input[2]).reshape(-1)         # (P,)
    W2 = (W * inv_std[None, :]).astype(np.float32)
    b2 = (b + W @ shift).astype(np.float32).reshape(1, -1)
    inits = _inits(model)
    for name, arr in ((gemm.input[1], W2), (gemm.input[2], b2)):
        old = inits[name]
        old.CopyFrom(NH.from_array(arr, name))
    gemm.input[0] = "features"
    for n in (mul, add, clip):
        model.graph.node.remove(n)
    return model


def states_2d(model: onnx.ModelProto) -> onnx.ModelProto:
    """Slim graphs carry (1, 1, H) states with an Unsqueeze before the first
    GRU and a Reshape after the last. Make the states (1, H) and drop both."""
    model = onnx.ModelProto.FromString(model.SerializeToString())
    changed = False
    for vi in list(model.graph.input) + list(model.graph.output):
        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        if vi.name != "features" and vi.name != "prediction" and len(dims) == 3 and dims[:2] == [1, 1]:
            del vi.type.tensor_type.shape.dim[0]
            changed = True
    if not changed:
        return model
    for n in list(model.graph.node):
        if n.op_type == "Unsqueeze" and any(c.op_type in ("GRU", "LSTM") for c in _consumers(model, n.output[0])):
            _rewire(model, n.output[0], n.input[0])
            model.graph.node.remove(n)
        elif n.op_type == "Reshape" and any(n.input[0] == o.name for o in model.graph.output):
            _rewire(model, n.output[0], n.input[0])
            model.graph.node.remove(n)
    # the fused ops still expect 3D; the unroll pass replaces them with 2D-agnostic MatMuls.
    return model


def unroll(model: onnx.ModelProto) -> onnx.ModelProto:
    """Return a copy with every GRU/LSTM node replaced by explicit ops."""
    model = onnx.ModelProto.FromString(model.SerializeToString())
    new_nodes, new_inits, k = [], [], 0

    def add_init(name, arr):
        new_inits.append(NH.from_array(np.ascontiguousarray(arr), name))
        return name

    for node in model.graph.node:
        if node.op_type not in ("GRU", "LSTM"):
            new_nodes.append(node)
            continue
        attrs = {a.name: H.get_attribute_value(a) for a in node.attribute}
        if attrs.get("direction", b"forward") not in (b"forward", "forward"):
            raise ValueError("only forward RNNs are supported")
        hidden = attrs["hidden_size"]
        x, w_name, r_name, b_name = node.input[0], node.input[1], node.input[2], node.input[3]
        W = _fetch(model, w_name)[0]            # (G*H, in)
        R = _fetch(model, r_name)[0]            # (G*H, H)
        G = W.shape[0] // hidden
        B = _fetch(model, b_name)[0] if b_name else np.zeros(2 * G * hidden, np.float32)
        Wb, Rb = B[:G * hidden], B[G * hidden:]
        p = f"cell{k}_"
        k += 1
        Wt = add_init(p + "Wt", W.T)            # (in, G*H)
        Rt = add_init(p + "Rt", R.T)            # (H, G*H)

        def n(op, ins, outs, **kw):
            new_nodes.append(H.make_node(op, ins, outs, name=p + outs[0], **kw))

        if node.op_type == "GRU":
            if attrs.get("linear_before_reset", 0) != 1:
                raise ValueError("GRU must use linear_before_reset=1 (PyTorch export does)")
            h0 = node.input[5]
            y_out = node.output[0]
            yh_out = node.output[1] if len(node.output) > 1 and node.output[1] else p + "yh"
            # gi = x W^T + Wb ; gh = h R^T + Rb  (gate order z, r, n)
            n("MatMul", [x, Wt], [p + "gi0"])
            n("Add", [p + "gi0", add_init(p + "bi", Wb)], [p + "gi"])
            n("MatMul", [h0, Rt], [p + "gh0"])
            n("Add", [p + "gh0", add_init(p + "bh", Rb)], [p + "gh"])
            sp = add_init(p + "split2", np.array([2 * hidden, hidden], np.int64))
            n("Split", [p + "gi", sp], [p + "gi_zr", p + "gi_n"], axis=-1)
            n("Split", [p + "gh", sp], [p + "gh_zr", p + "gh_n"], axis=-1)
            n("Add", [p + "gi_zr", p + "gh_zr"], [p + "zr0"])
            n("Sigmoid", [p + "zr0"], [p + "zr"])
            sp1 = add_init(p + "split1", np.array([hidden, hidden], np.int64))
            n("Split", [p + "zr", sp1], [p + "z", p + "r"], axis=-1)
            n("Mul", [p + "r", p + "gh_n"], [p + "rn"])
            n("Add", [p + "gi_n", p + "rn"], [p + "n0"])
            n("Tanh", [p + "n0"], [p + "nn"])
            # h' = n + z * (h - n)  ==  (1 - z) * n + z * h
            n("Sub", [h0, p + "nn"], [p + "hmn"])
            n("Mul", [p + "z", p + "hmn"], [p + "zh"])
            n("Add", [p + "nn", p + "zh"], [yh_out])
            if y_out:
                n("Unsqueeze", [yh_out, add_init(p + "ax1", np.array([1], np.int64))], [y_out])
        else:  # LSTM, gate order i, o, f, c
            h0, c0 = node.input[5], node.input[6]
            y_out = node.output[0]
            yh_out = node.output[1] if len(node.output) > 1 and node.output[1] else p + "yh"
            yc_out = node.output[2] if len(node.output) > 2 and node.output[2] else p + "yc"
            n("MatMul", [x, Wt], [p + "gi0"])
            n("MatMul", [h0, Rt], [p + "gh0"])
            n("Add", [p + "gi0", p + "gh0"], [p + "g0"])
            n("Add", [p + "g0", add_init(p + "b", Wb + Rb)], [p + "g"])
            sp = add_init(p + "split4", np.array([hidden] * 4, np.int64))
            n("Split", [p + "g", sp], [p + "i0", p + "o0", p + "f0", p + "c0"], axis=-1)
            n("Sigmoid", [p + "i0"], [p + "i"])
            n("Sigmoid", [p + "o0"], [p + "o"])
            n("Sigmoid", [p + "f0"], [p + "f"])
            n("Tanh", [p + "c0"], [p + "cc"])
            n("Mul", [p + "f", c0], [p + "fc"])
            n("Mul", [p + "i", p + "cc"], [p + "ic"])
            n("Add", [p + "fc", p + "ic"], [yc_out])
            n("Tanh", [yc_out], [p + "tc"])
            n("Mul", [p + "o", p + "tc"], [yh_out])
            if y_out:
                n("Unsqueeze", [yh_out, add_init(p + "ax1", np.array([1], np.int64))], [y_out])

    del model.graph.node[:]
    model.graph.node.extend(new_nodes)
    model.graph.initializer.extend(new_inits)
    # drop the now-unused fused weights
    used = {i for nd in model.graph.node for i in nd.input}
    keep = [i for i in model.graph.initializer if i.name in used]
    del model.graph.initializer[:]
    model.graph.initializer.extend(keep)
    onnx.checker.check_model(model)
    return model


def quantize(src: Path, dst: Path, mode: str) -> None:
    """dynamic: int8 weights + per-call int8 activations on every MatMul (fastest).
    nbits8/nbits4: weight-only block quantisation, fp32 activations (ORT MatMulNBits)."""
    if mode == "dynamic":
        from onnxruntime.quantization import QuantType, quantize_dynamic
        quantize_dynamic(str(src), str(dst), weight_type=QuantType.QInt8,
                         extra_options={"DefaultTensorType": onnx.TensorProto.FLOAT})
        return
    if mode in ("nbits8", "nbits4"):
        import logging
        from onnxruntime.quantization.matmul_nbits_quantizer import (
            DefaultWeightOnlyQuantConfig, MatMulNBitsQuantizer)
        logging.getLogger("onnxruntime.quantization.matmul_nbits_quantizer").setLevel(logging.WARNING)
        bits = 8 if mode == "nbits8" else 4
        cfg = DefaultWeightOnlyQuantConfig(block_size=32, is_symmetric=True, accuracy_level=4, bits=bits)
        q = MatMulNBitsQuantizer(onnx.load(str(src)), algo_config=cfg)
        q.process()
        q.model.save_model_to_file(str(dst), False)
        return
    raise ValueError(mode)


def _session(path):
    import onnxruntime as ort
    o = ort.SessionOptions()
    o.intra_op_num_threads = 1
    o.inter_op_num_threads = 1
    o.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    o.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    o.add_session_config_entry("session.intra_op.allow_spinning", "0")
    return ort.InferenceSession(str(path), sess_options=o, providers=["CPUExecutionProvider"])


def synthetic_sequence(rows: int, seed: int = 0) -> np.ndarray:
    """Rank-transformed-looking features: AR(1) per column, clipped to +-2.32."""
    rng = np.random.default_rng(seed)
    x = np.zeros((rows, 112), np.float32)
    x[0] = rng.standard_normal(112)
    noise = rng.standard_normal((rows, 112)).astype(np.float32) * 0.3
    for t in range(1, rows):
        x[t] = 0.95 * x[t - 1] + noise[t]
    return np.clip(x, -2.32, 2.32)


def run_sequence(path, x: np.ndarray):
    """Drive a graph row by row. Works for one packed state (export.py) and for
    one state per layer (export_slim.py): every non-feature input is state."""
    sess = _session(path)
    inputs = sess.get_inputs()
    feat = next(i for i in inputs if i.name == "features")
    fshape = tuple(d if isinstance(d, int) else 1 for d in feat.shape)
    state_names = [i.name for i in inputs if i.name != "features"]
    states = {i.name: np.zeros(tuple(d if isinstance(d, int) else 1 for d in i.shape), np.float32)
              for i in inputs if i.name != "features"}
    next_names = [o.name for o in sess.get_outputs() if o.name != "prediction"]
    out_names = ["prediction"] + next_names
    out = np.zeros((len(x), 2), np.float32)
    t0 = time.perf_counter()
    for t in range(len(x)):
        res = sess.run(out_names, {"features": x[t].reshape(fshape), **states})
        out[t] = res[0].reshape(-1)
        for name, val in zip(state_names, res[1:]):
            states[name] = val
    us = (time.perf_counter() - t0) / len(x) * 1e6
    return out, us


def compare(ref, out):
    r = [float(np.corrcoef(ref[:, j], out[:, j])[0, 1]) for j in range(2)]
    return {"max_abs": float(np.abs(ref - out).max()),
            "rel_rms": float(np.sqrt(((ref - out) ** 2).mean()) / np.sqrt((ref ** 2).mean())),
            "corr_t0": r[0], "corr_t1": r[1]}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", type=Path)
    ap.add_argument("dst", type=Path)
    ap.add_argument("--quant", choices=["dynamic", "nbits8", "nbits4"], default=None)
    ap.add_argument("--check", action="store_true", help="parity + drift on a 20k-row synthetic sequence")
    ap.add_argument("--rows", type=int, default=20000)
    ap.add_argument("--no-lean", action="store_true",
                    help="skip folding the standardisation into the first Gemm and the 2D-state rewrite")
    args = ap.parse_args()

    model = onnx.load(str(args.src))
    if not args.no_lean:
        model = states_2d(fold_standardise(model))
    model = unroll(model)
    print(f"{len(model.graph.node)} nodes after rewrite")
    fp32_path = args.dst if args.quant is None else args.dst.with_suffix(".fp32.onnx")
    onnx.save(model, str(fp32_path))
    if args.quant:
        quantize(fp32_path, args.dst, args.quant)
    print(f"wrote {args.dst} ({args.dst.stat().st_size / 1e6:.2f} MB)")

    if args.check:
        x = synthetic_sequence(args.rows)
        ref, us_ref = run_sequence(args.src, x)
        print(f"source graph        {us_ref:6.1f} us/row")
        fp, us_fp = run_sequence(fp32_path, x)
        print(f"unrolled fp32       {us_fp:6.1f} us/row  parity max_abs={np.abs(ref - fp).max():.2e}")
        if args.quant:
            q, us_q = run_sequence(args.dst, x)
            d = compare(ref, q)
            print(f"unrolled {args.quant:9s}{us_q:6.1f} us/row  drift {d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
