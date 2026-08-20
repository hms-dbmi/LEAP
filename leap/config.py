import argparse
import os
from typing import Iterable, Optional

import torch
from omegaconf import DictConfig, OmegaConf

from leap.runs import run_timestamp


def build_parser(description: str) -> argparse.ArgumentParser:
    """An argument parser with --config and --set, used by every entry point."""
    parser = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config",
        required=True,
        metavar="PATH",
        help="path to this task's YAML config (see configs/ for a template)",
    )
    parser.add_argument(
        "--set",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a config key, dotted path; repeatable "
             "(e.g. --set train.epochs=5 --set data.patches_per_slide=50)",
    )
    return parser


def load_config(path: str, overrides: Optional[Iterable[str]] = None) -> DictConfig:
    """Load a YAML config and apply `KEY=VALUE` overrides."""
    if not os.path.exists(path):
        raise SystemExit(f"config not found: {path}")
    cfg = OmegaConf.load(path)
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(list(overrides)))
    return cfg


def require(cfg: DictConfig, *keys: str):
    """Return the values of the given dotted config keys, raising if any is missing or blank.

    The shipped configs are blank templates, so this turns an unfilled key into a message
    naming the key instead of a downstream TypeError.
    """
    missing, values = [], []
    for key in keys:
        value = OmegaConf.select(cfg, key)
        if value is None or (isinstance(value, str) and not value.strip()):
            missing.append(key)
        values.append(value)
    if missing:
        raise SystemExit(
            "the following config keys must be set before running:\n  "
            + "\n  ".join(missing)
        )
    return values[0] if len(values) == 1 else tuple(values)


def option(cfg: DictConfig, key: str, default=None):
    """A config value, treating a key that is present but blank as absent.

    The shipped configs list every key with an empty value, so `cfg.get(key, default)` would
    return None rather than the default for any key the user left blank.
    """
    value = OmegaConf.select(cfg, key)
    if value is None or (isinstance(value, str) and not value.strip()):
        return default
    return value


def resolve_device(requested: Optional[str] = None) -> torch.device:
    """The device to run on: `requested` if usable, else CUDA when available, else CPU."""
    if requested:
        if str(requested).startswith("cuda") and not torch.cuda.is_available():
            print("[device] CUDA requested but unavailable; falling back to CPU")
            return torch.device("cpu")
        return torch.device(str(requested))
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def experiment_id(cfg: DictConfig, *parts: str) -> str:
    """A run identifier: the configured `experiment_name` if set, else a timestamped default.

    Extra `parts` are appended with dots, so an id records the architecture it belongs to.
    """
    name = cfg.get("experiment_name") or f"{run_timestamp()}_leap"
    suffix = ".".join(str(p) for p in parts if p)
    return f"{name}.{suffix}" if suffix else str(name)
