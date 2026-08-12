"""Train a Transformer for the WHEN (time-to-next-outage) problem.

The public ``train_transformer_when`` function is intentionally stateless and
returns only the held-out mean absolute error (MAE), in seconds.  This makes it
straightforward to call from a grid-search loop, for example::

    from sklearn.model_selection import ParameterGrid
    from src.train_transformer_when import train_transformer_when

    grid = ParameterGrid({
        "learning_rate": [1e-5, 5e-5],
        "transformer_dim": [128, 256],
        "num_heads": [4, 8],
        "dropout": [0.1, 0.2],
    })
    validation_split = {"train_folds": (0, 1, 2, 3), "test_fold": 4}
    results = [
        (params, train_transformer_when(**params, **validation_split))
        for params in grid
    ]

By default, folds 0--4 are used for training and fold -1 is used for testing,
matching the existing predictive-maintenance notebooks in this repository.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset


BASE_DIR = Path(__file__).resolve().parent
DEFAULT_SEQUENCE_DATASET_PATH = BASE_DIR / "output" / "dataset_winsize1h_where.csv"
DEFAULT_WHEN_DATASET_PATH = BASE_DIR / "output" / "dataset_winsize1h_when.csv"
TARGET_COLUMN = "label_time_to_event_seconds"
OUTAGE_TYPE_TO_ID = {"Planned": 0, "Auto": 1}


def _set_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for repeatable grid-search trials."""
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)


def _load_data(
    sequence_dataset_path: str | Path,
    when_dataset_path: str | Path,
    train_folds: Sequence[int],
    test_fold: int,
) -> tuple[pd.DataFrame, pd.DataFrame, int, int]:
    sequence_dataset_path = Path(sequence_dataset_path)
    when_dataset_path = Path(when_dataset_path)

    sequence_df = pd.read_csv(
        sequence_dataset_path,
        dtype={"fold_id": "int64"},
        converters={"window": json.loads},
    )
    when_df = pd.read_csv(when_dataset_path, dtype={"fold_id": "int64"})

    required_sequence_columns = {"fold_id", "window", "is_bg"}
    required_when_columns = {"fold_id", "is_bg", TARGET_COLUMN}
    missing_sequence = required_sequence_columns.difference(sequence_df.columns)
    missing_when = required_when_columns.difference(when_df.columns)
    if missing_sequence:
        raise ValueError(
            f"{sequence_dataset_path} is missing columns: {sorted(missing_sequence)}"
        )
    if missing_when:
        raise ValueError(
            f"{when_dataset_path} is missing columns: {sorted(missing_when)}"
        )

    if len(sequence_df) != len(when_df):
        raise ValueError(
            "Sequence and WHEN datasets have different row counts: "
            f"{len(sequence_df)} != {len(when_df)}"
        )
    if not np.array_equal(sequence_df["fold_id"], when_df["fold_id"]):
        raise ValueError("Fold IDs are not row-aligned between the two datasets.")
    if not np.array_equal(sequence_df["is_bg"], when_df["is_bg"]):
        raise ValueError("Background labels are not row-aligned between the datasets.")

    data_df = sequence_df.loc[~when_df["is_bg"].astype(bool)].copy()
    data_df["target_seconds"] = when_df.loc[data_df.index, TARGET_COLUMN].astype(float)
    data_df.reset_index(drop=True, inplace=True)

    if data_df["target_seconds"].isna().any():
        raise ValueError("The non-background data contains missing time targets.")
    if (~np.isfinite(data_df["target_seconds"])).any():
        raise ValueError("The non-background data contains non-finite time targets.")
    if (data_df["target_seconds"] < 0).any():
        raise ValueError("Time-to-event targets must be non-negative.")

    train_folds = tuple(int(fold) for fold in train_folds)
    if not train_folds:
        raise ValueError("train_folds must contain at least one fold ID.")
    if test_fold in train_folds:
        raise ValueError("test_fold must not also appear in train_folds.")

    train_df = data_df[data_df["fold_id"].isin(train_folds)].copy()
    test_df = data_df[data_df["fold_id"] == test_fold].copy()
    if train_df.empty:
        raise ValueError(f"No training rows found for folds {train_folds}.")
    if test_df.empty:
        raise ValueError(f"No test rows found for fold {test_fold}.")

    all_events = [event for window in data_df["window"] for event in window]
    if not all_events:
        raise ValueError("The sequence dataset contains no outage events.")

    required_event_fields = {
        "from_zone_index",
        "to_zone_index",
        "outage_type",
        "time_interval_index",
    }
    for row_number, window in enumerate(data_df["window"]):
        if not window:
            raise ValueError(f"Sequence row {row_number} has an empty event window.")
        for event in window:
            missing_fields = required_event_fields.difference(event)
            if missing_fields:
                raise ValueError(
                    f"Sequence row {row_number} has an event missing fields: "
                    f"{sorted(missing_fields)}"
                )
            if event["outage_type"] not in OUTAGE_TYPE_TO_ID:
                raise ValueError(
                    f"Unknown outage type {event['outage_type']!r} in row {row_number}."
                )
            if min(
                event["from_zone_index"],
                event["to_zone_index"],
                event["time_interval_index"],
            ) < 0:
                raise ValueError(
                    f"Sequence row {row_number} contains a negative index."
                )

    num_zones = (
        max(
            max(event["from_zone_index"], event["to_zone_index"])
            for event in all_events
        )
        + 1
    )
    num_time_intervals = max(
        event["time_interval_index"] for event in all_events
    ) + 1
    return train_df, test_df, num_zones, num_time_intervals


class WhenOutageDataset(Dataset):
    """Pre-tensorized variable-length outage sequences and regression targets."""

    def __init__(self, dataframe: pd.DataFrame) -> None:
        self.samples = []
        for row in dataframe.itertuples(index=False):
            window = row.window
            self.samples.append(
                {
                    "outage_type": torch.tensor(
                        [OUTAGE_TYPE_TO_ID[event["outage_type"]] for event in window],
                        dtype=torch.long,
                    ),
                    "from_zone_indices": torch.tensor(
                        [event["from_zone_index"] for event in window], dtype=torch.long
                    ),
                    "to_zone_indices": torch.tensor(
                        [event["to_zone_index"] for event in window], dtype=torch.long
                    ),
                    "time_interval_index": torch.tensor(
                        [event["time_interval_index"] for event in window],
                        dtype=torch.long,
                    ),
                    # log1p stabilizes regression while supporting a zero target.
                    "target_log_seconds": torch.tensor(
                        np.log1p(row.target_seconds), dtype=torch.float32
                    ),
                    "target_seconds": torch.tensor(
                        row.target_seconds, dtype=torch.float64
                    ),
                }
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        return self.samples[index]


def _collate_samples(samples: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    lengths = torch.tensor(
        [len(sample["outage_type"]) for sample in samples], dtype=torch.long
    )
    max_length = int(lengths.max())
    padding_mask = torch.arange(max_length).unsqueeze(0) >= lengths.unsqueeze(1)

    batch = {
        key: pad_sequence(
            [sample[key] for sample in samples], batch_first=True, padding_value=0
        )
        for key in (
            "outage_type",
            "from_zone_indices",
            "to_zone_indices",
            "time_interval_index",
        )
    }
    batch.update(
        {
            "lengths": lengths,
            "padding_mask": padding_mask,
            "target_log_seconds": torch.stack(
                [sample["target_log_seconds"] for sample in samples]
            ),
            "target_seconds": torch.stack(
                [sample["target_seconds"] for sample in samples]
            ),
        }
    )
    return batch


class WhenTransformer(nn.Module):
    """Transformer regressor over the outage events preceding a prediction."""

    def __init__(
        self,
        *,
        num_zones: int,
        num_time_intervals: int,
        zone_embedding_dim: int,
        outage_type_embedding_dim: int,
        time_interval_embedding_dim: int,
        transformer_dim: int,
        num_heads: int,
        num_layers: int,
        feedforward_dim: int,
        dropout: float,
        pooling: str,
    ) -> None:
        super().__init__()
        self.transformer_dim = transformer_dim
        self.pooling = pooling
        self.zone_embedding = nn.Embedding(num_zones, zone_embedding_dim)
        self.outage_type_embedding = nn.Embedding(
            len(OUTAGE_TYPE_TO_ID), outage_type_embedding_dim
        )
        self.time_interval_embedding = nn.Embedding(
            num_time_intervals, time_interval_embedding_dim
        )
        input_dim = (
            2 * zone_embedding_dim
            + outage_type_embedding_dim
            + time_interval_embedding_dim
        )
        self.input_projection = nn.Linear(input_dim, transformer_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=transformer_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.output_norm = nn.LayerNorm(transformer_dim)
        self.regression_head = nn.Linear(transformer_dim, 1)

    def _positional_encoding(
        self, sequence_length: int, x: torch.Tensor
    ) -> torch.Tensor:
        """Create sinusoidal positions dynamically for variable-length windows."""
        position = torch.arange(
            sequence_length, device=x.device, dtype=x.dtype
        ).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, self.transformer_dim, 2, device=x.device, dtype=x.dtype)
            * (-math.log(10_000.0) / self.transformer_dim)
        )
        angles = position * frequencies.unsqueeze(0)
        encoding = torch.zeros(
            sequence_length, self.transformer_dim, device=x.device, dtype=x.dtype
        )
        encoding[:, 0::2] = torch.sin(angles)
        encoding[:, 1::2] = torch.cos(angles[:, : encoding[:, 1::2].shape[1]])
        return encoding.unsqueeze(0)

    def forward(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        x = torch.cat(
            (
                self.zone_embedding(batch["from_zone_indices"]),
                self.zone_embedding(batch["to_zone_indices"]),
                self.outage_type_embedding(batch["outage_type"]),
                self.time_interval_embedding(batch["time_interval_index"]),
            ),
            dim=-1,
        )
        x = self.input_projection(x)
        x = x + self._positional_encoding(x.shape[1], x)
        x = self.encoder(x, src_key_padding_mask=batch["padding_mask"])

        if self.pooling == "last":
            pooled = x[
                torch.arange(x.shape[0], device=x.device), batch["lengths"] - 1
            ]
        else:
            valid_tokens = (~batch["padding_mask"]).unsqueeze(-1)
            pooled = (x * valid_tokens).sum(dim=1) / batch["lengths"].unsqueeze(1)

        return self.regression_head(self.output_norm(pooled)).squeeze(-1)


def _validate_hyperparameters(
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    zone_embedding_dim: int,
    outage_type_embedding_dim: int,
    time_interval_embedding_dim: int,
    transformer_dim: int,
    num_heads: int,
    num_layers: int,
    feedforward_dim: int,
    dropout: float,
    pooling: str,
    gradient_clip_norm: float | None,
    num_workers: int,
) -> None:
    positive_integers = {
        "epochs": epochs,
        "batch_size": batch_size,
        "zone_embedding_dim": zone_embedding_dim,
        "outage_type_embedding_dim": outage_type_embedding_dim,
        "time_interval_embedding_dim": time_interval_embedding_dim,
        "transformer_dim": transformer_dim,
        "num_heads": num_heads,
        "num_layers": num_layers,
        "feedforward_dim": feedforward_dim,
    }
    for name, value in positive_integers.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}.")
    if transformer_dim % num_heads != 0:
        raise ValueError("transformer_dim must be divisible by num_heads.")
    if learning_rate <= 0:
        raise ValueError("learning_rate must be positive.")
    if weight_decay < 0:
        raise ValueError("weight_decay must be non-negative.")
    if not 0 <= dropout < 1:
        raise ValueError("dropout must be in the interval [0, 1).")
    if pooling not in {"last", "mean"}:
        raise ValueError("pooling must be either 'last' or 'mean'.")
    if gradient_clip_norm is not None and gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be positive when provided.")
    if num_workers < 0:
        raise ValueError("num_workers must be non-negative.")


def train_transformer_when(
    sequence_dataset_path: str | Path = DEFAULT_SEQUENCE_DATASET_PATH,
    when_dataset_path: str | Path = DEFAULT_WHEN_DATASET_PATH,
    *,
    epochs: int = 16,
    batch_size: int = 32,
    learning_rate: float = 1e-5,
    weight_decay: float = 1e-5,
    zone_embedding_dim: int = 64,
    outage_type_embedding_dim: int = 32,
    time_interval_embedding_dim: int = 32,
    transformer_dim: int = 256,
    num_heads: int = 4,
    num_layers: int = 1,
    feedforward_dim: int = 1024,
    dropout: float = 0.15,
    pooling: str = "last",
    gradient_clip_norm: float | None = 1.0,
    train_folds: Sequence[int] = (0, 1, 2, 3, 4),
    test_fold: int = -1,
    random_state: int = 42,
    device: str | torch.device | None = None,
    num_workers: int = 0,
) -> float:
    """Train a WHEN Transformer and return the held-out MAE in seconds.

    The model minimizes MAE after a ``log1p`` target transform, then predictions
    are transformed back to seconds for the returned test metric.  No model,
    history, files, or auxiliary metrics are returned or saved.

    For hyperparameter selection without repeatedly inspecting the final test
    fold, use one of folds 0--4 as ``test_fold`` and remove it from
    ``train_folds``.  Once parameters are selected, call the defaults to train
    on folds 0--4 and report fold -1 exactly once.
    """
    _validate_hyperparameters(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        zone_embedding_dim=zone_embedding_dim,
        outage_type_embedding_dim=outage_type_embedding_dim,
        time_interval_embedding_dim=time_interval_embedding_dim,
        transformer_dim=transformer_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        feedforward_dim=feedforward_dim,
        dropout=dropout,
        pooling=pooling,
        gradient_clip_norm=gradient_clip_norm,
        num_workers=num_workers,
    )
    _set_seed(random_state)

    if device is None:
        resolved_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        resolved_device = torch.device(device)
    if resolved_device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("A CUDA device was requested, but CUDA is not available.")

    train_df, test_df, num_zones, num_time_intervals = _load_data(
        sequence_dataset_path,
        when_dataset_path,
        train_folds,
        test_fold,
    )
    train_dataset = WhenOutageDataset(train_df)
    test_dataset = WhenOutageDataset(test_df)

    loader_generator = torch.Generator().manual_seed(random_state)
    loader_options = {
        "batch_size": batch_size,
        "collate_fn": _collate_samples,
        "num_workers": num_workers,
        "pin_memory": resolved_device.type == "cuda",
    }
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        generator=loader_generator,
        **loader_options,
    )
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)

    model = WhenTransformer(
        num_zones=num_zones,
        num_time_intervals=num_time_intervals,
        zone_embedding_dim=zone_embedding_dim,
        outage_type_embedding_dim=outage_type_embedding_dim,
        time_interval_embedding_dim=time_interval_embedding_dim,
        transformer_dim=transformer_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        feedforward_dim=feedforward_dim,
        dropout=dropout,
        pooling=pooling,
    ).to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )

    model.train()
    for _ in range(epochs):
        for batch in train_loader:
            batch = {
                key: value.to(resolved_device, non_blocking=True)
                for key, value in batch.items()
            }
            optimizer.zero_grad(set_to_none=True)
            prediction_log_seconds = model(batch)
            loss = torch.abs(
                prediction_log_seconds - batch["target_log_seconds"]
            ).mean()
            loss.backward()
            if gradient_clip_norm is not None:
                nn.utils.clip_grad_norm_(model.parameters(), gradient_clip_norm)
            optimizer.step()

    absolute_error_sum = 0.0
    sample_count = 0
    model.eval()
    with torch.no_grad():
        for batch in test_loader:
            batch = {
                key: value.to(resolved_device, non_blocking=True)
                for key, value in batch.items()
            }
            prediction_log_seconds = model(batch).clamp_min(0.0)
            prediction_seconds = torch.expm1(prediction_log_seconds).double()
            target_seconds = batch["target_seconds"]
            absolute_error_sum += torch.abs(
                prediction_seconds - target_seconds
            ).sum().item()
            sample_count += target_seconds.numel()

    return float(absolute_error_sum / sample_count)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train the WHEN Transformer and print only test MAE in seconds.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--sequence-dataset", type=Path, default=DEFAULT_SEQUENCE_DATASET_PATH
    )
    parser.add_argument("--when-dataset", type=Path, default=DEFAULT_WHEN_DATASET_PATH)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--zone-embedding-dim", type=int, default=64)
    parser.add_argument("--outage-type-embedding-dim", type=int, default=32)
    parser.add_argument("--time-interval-embedding-dim", type=int, default=32)
    parser.add_argument("--transformer-dim", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=4)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--feedforward-dim", type=int, default=1024)
    parser.add_argument("--dropout", type=float, default=0.15)
    parser.add_argument("--pooling", choices=("last", "mean"), default="last")
    parser.add_argument("--gradient-clip-norm", type=float, default=1.0)
    parser.add_argument("--train-folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--test-fold", type=int, default=-1)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--device", default=None)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = vars(_parse_args())
    arguments["sequence_dataset_path"] = arguments.pop("sequence_dataset")
    arguments["when_dataset_path"] = arguments.pop("when_dataset")
    print(train_transformer_when(**arguments))
