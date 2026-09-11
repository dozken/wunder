# Wunder Challenges

My solutions for [Wunder Fund](https://wundernn.io) machine learning challenges.

| Challenge | Task | Result | Solution |
|---|---|---|---|
| [Challenge 3 — Alpha Connectome (WNN33)](challenge-3/) | Same targets, two instruments, 112 features, 20,000-step sequences | in progress (baseline 0.5896 WP) | — |
| [Challenge 2 — Predictorium](challenge-2/) | Predict future price movement indicators from Limit Order Book sequences | Validation WPC **0.2574** | LSTM V5, 5-fold ensemble, ONNX |

Each challenge lives in its own directory with its own `mise.toml`, Docker scorer
environment and README — run tasks from inside that directory:

```bash
cd challenge-2
mise run test
mise run score
mise run docker-test
```

## License

MIT — see [LICENSE](LICENSE).
