# Wunder Challenges

My solutions for [Wunder Fund](https://wundernn.io) machine learning challenges.

| Challenge | Task | Result | Solution |
|---|---|---|---|
| [Challenge 3 — Alpha Connectome (WNN33)](challenge-3/) | Same targets, two instruments, 112 features, 20,000-step sequences, hidden scoring mask | **active** — public 0.6221 (#11–12), held-out 0.666 | GRU 192×2, sequence-level Pearson loss, soft-masked rows, ONNX |
| [Challenge 2 — Predictorium](challenge-2/) | Predict future price movement indicators from Limit Order Book sequences | Validation WPC **0.2574** | LSTM V5, 5-fold ensemble, ONNX |

Each challenge lives in its own directory with its own `mise.toml`, Docker scorer
environment and README — run tasks from inside that directory:

```bash
cd challenge-3
mise run bench          # inference latency of src/solution.py
mise run docker-test    # 1-CPU scorer container
```

`CLAUDE.md` holds the working rules for the active challenge.

## License

MIT — see [LICENSE](LICENSE).
