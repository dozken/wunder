# Data Overview

Understanding the data is key to success in this challenge. Here’s a detailed breakdown of the dataset format, structure, and key properties.

To get the data you need to sign up and go to [quick start page](./01_quick_start.md).

## Data format

The entire dataset is provided as a single table in a Parquet file. Each row represents a specific Limit Order Book (LOB) snapshot along with the trade information accumulated since the previous snapshot.

The table has the following columns:

*   `seq_ix`: The ID of the sequence. This is an integer that identifies which sequence the row belongs to.
*   `step_in_seq`: An integer representing the step number within a sequence, from 0 to 999.
*   `need_prediction`: A boolean (True or False). If True, you need to provide a prediction for the next step.

### Feature columns

The numeric features provide a structured representation of the market state, derived from the Limit Order Book (LOB) and recent trade activity. While the data is anonymized, the structure follows standard market data conventions:

**Price features**
*   `p0...p5` — **bid prices features**: Anonymized features derived from the bid side of the Limit Order Book representing prices at different levels.
*   `p6...p11` — **ask prices features**: Anonymized features derived from the ask side of the Limit Order Book representing prices at different levels.

**Volume features**
*   `v0...v5` — **bid volume features**: Anonymized features representing volumes corresponding to price levels on the bid side.
*   `v6...v11` — **ask volume features**: Anonymized features representing volumes corresponding to price levels on the ask side.

**Trades features**
*   `dp0...dp3` — **trade prices features**: Anonymized features representing trade prices (derived from both bid and ask trades).
*   `dv0...dv3` — **trade volumes features**: Anonymized features representing trade volumes (derived from both bid and ask trades).

### Target columns

*   `t0`, `t1`: The targets represent two different types of future price movement indicators. These values are also anonymized. Your task is to predict these values.

## The sequences

The data is organized into many independent sequences.

*   **Sequence length**: Each sequence is exactly 1000 steps long (from `step_in_seq` 0 to 999).
*   **Independence**: Each sequence is completely independent of the others. The market history from one sequence does not carry over to the next. When `seq_ix` changes, you are starting fresh.
*   **Warm-up period**: The first 99 steps (0-98) of every sequence are a "warm-up" period. You can use this data to build up your model's internal state (e.g., for an LSTM or Transformer), but you will not be scored on any predictions for these steps.
*   **Scored predictions**: Your score is based on predictions for steps 99 to 999 (inclusive). These are the steps where `need_prediction` will be True.

### Data ordering

*   **Inside a sequence**: Rows are always ordered chronologically. `step_in_seq == 1` always comes after `step_in_seq == 0`.
*   **Between sequences**: The sequences themselves are shuffled. `seq_ix == 10` is not related to `seq_ix == 11`.

## Dataset sizes & validation

*   **Train**: `datasets/train.parquet` — training dataset containing 10,721 sequences, provided.
*   **Validation**: `datasets/valid.parquet` — validation dataset containing 1,444 sequences, provided.
*   **Test**: a test dataset that is used for public leaderboard scoring, hidden.
*   **Final**: a final dataset that will be used for private leadearboard, hidden.

**Test & Final sets**: For both the Public and Private Leaderboards, you can expect the respective datasets to contain approximately 1,500 sequences, similar to the Validation dataset.

> **TIP: Validation strategy**
>
> We have already prepared `valid.parquet` for you to use as a consistent local validation set. However, you are completely free to:
>
> *   Use your own validation strategy.
> *   Combine `train.parquet` and `valid.parquet` into a larger training set.
> *   Create custom splits (e.g., K-Fold) based on `seq_ix`.
>
> Since all sequences are independent and shuffled, we believe that random splitting by `seq_ix` is likely a valid and robust approach for this task.

## Evaluation metric

We evaluate predictions using the **Weighted Pearson Correlation Coefficient** score:

![Score Formula](image.png)

### More details about the score

This metric weights samples by the amplitude of price change (`abs(target)`) and clips predictions to `[-6, 6]`. You can find implementation details at `utils.weighted_pearson_correlation`.

The final score is the average of the correlation scores across the two targets (`t0`, `t1`).

**A higher correlation score is better.**
