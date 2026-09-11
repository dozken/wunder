# Quick start guide

> Mirrored from https://wundernn.io/connectome/docs/quick_start (fetched 2026-09-11).
> Formatting is approximate; the live page is authoritative.

This guide will help you get started with the Alpha Connectome ML Competition.
1. Get the starter pack
curl -L https://files.wundernn.io/wnn_connectome_starterpack.tar.gz | tar -xz && \
cd wnn_connectome_starterpack
or download the archive manually:
 wnn_connectome_starterpack.zip 
 wnn_connectome_starterpack.tar.gz 
slow download? VPN enjoyer? try mirror-1/mirror-2
What's inside
wnn_connectome_starterpack/
├── baseline
│ ├── README.md
│ ├── baseline.onnx
│ ├── baseline_submission.zip
│ └── solution.py
├── datasets
│ ├── train.parquet
│ ├── valid.parquet
│ └── valid_mask.parquet
├── docs
├── METRIC.md
├── README.md
├── requirements.txt
└── utils.py
2. Set up your environment and packages
We strongly recommend using a virtual environment to keep your project dependencies tidy and avoid conflicts. Probably any Python 3.10+ will do. Submissions will be evaluated in a Docker container based on python:3.11-slim-bookworm image.
Set up the env and then install the required packages:
pip install -r requirements.txt
3. Explore the data
The dataset is provided in Parquet format.
datasets/train.parquet: 10,607 sequences.
datasets/valid.parquet: 1,873 sequences.
Each sequence contains 20,000 rows. Steps 0–98 are warm-up; predictions are required at steps 99–19,999. Inputs contain 112 features.
import pyarrow.parquet as pq
dataset = pq.ParquetFile('datasets/train.parquet')
sequence = dataset.read_row_group(0)
print(sequence.shape)
print(sequence.column_names)
The details are in Data overview.
4. Run the example solution
We provided a baseline solution in baseline folder.
python baseline/solution.py --validation datasets/valid.parquet
This script will load the validation data and the pre-trained baseline.onnx model, run inference, and calculate the Weighted Pearson score.
This baseline is a stateful vanilla GRU.
It scores 0.589595 WP on the complete validation set.
The ONNX Runtime session uses one CPU compute thread.
5. Submit
Zip your solution.py and model artifacts:
cd baseline
zip -r submission.zip solution.py baseline.onnx
Go to the submit page and send your solution for scoring.
🎉 ta-daa, you're awesome
6. Create your own model
Create your own solution.py implementing the PredictionModel interface. Use Submission guide for reference.
Train your model using datasets/train.parquet.
Validate using datasets/valid.parquet.
The data and metric is described in details in Data overview. The starterpack's METRIC.md contains the full scoring reference.
Good luck with the challenge! We're excited to see what you build.
