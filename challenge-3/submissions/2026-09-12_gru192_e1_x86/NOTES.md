# gru192_e1_x86 — the same model, repackaged for the scorer's x86 vCPU

Same weights as `../2026-09-12_gru192_e1` (GRU 192×2, proj 64, EMA after
epoch 1 of the first full-data run, held-out WP 0.5955). Only the graph and
the wrapper changed:

- `src/unroll.py` rewrote the two fused ONNX `GRU` ops as explicit
  MatMul + elementwise cells (fp32 parity with the source graph: 6e-7), then
  quantised every MatMul weight to 8-bit blocks with ORT's `MatMulNBits`
  kernel (block 32, symmetric, int8 compute).
- `solution.py` binds inputs/outputs once and ping-pongs the recurrent state
  between two buffers (IO binding); it is bit-identical to the old wrapper.

Why: on a cloud Xeon (Sapphire Rapids, 2.1 GHz, KVM, 1 pinned core) the
original zip measured **100 µs/row = 62.6 min, 0.96× headroom** — over the
60 min budget. The fused GRU kernel is ~2.3× slower there than on Apple
silicon and the int8 LSTM kernel that helps on the Mac does nothing on x86.

Measured on that Xeon, one row per call, 20k-row sequence, best of 3:

| variant | raw ORT µs/row | end-to-end µs/row | projected | headroom |
|---|---:|---:|---:|---:|
| original fp32 fused GRU (`../2026-09-12_gru192_e1`) | 82 | 100 | 62.6 min | 0.96× |
| unrolled fp32 | 100 | – | – | – |
| unrolled + dynamic int8 (`--quant dynamic`) | 55 | – | – | – |
| unrolled + 4-bit weights (`--quant nbits4`) | 52 | – | – | – |
| **unrolled + 8-bit weights (`--quant nbits8`), this zip** | **38** | **47** | **29.5 min** | **2.03×** |

Drift of the 8-bit graph vs fp32 over a 20k-row synthetic AR(1) sequence:
prediction correlation 0.99997 on both targets, relative RMS error 0.7 %,
max abs 0.03. 4-bit drifts to 0.996 and is not faster, so it is out.
The WP impact on real validation data has **not** been measured yet (no data
in the machine this was built on); run `python src/score.py --strict` on the
Mac before relying on it, but a 0.99997 correlation cannot move WP by more
than a few 1e-4.

Provided baseline for reference: 0.5896 on the full validation set.
