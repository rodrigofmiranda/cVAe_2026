# -*- coding: utf-8 -*-
"""Simple GPU warning gate for canonical CLI entrypoints."""

from __future__ import annotations

from typing import List


def check_tensorflow_gpu() -> List[object]:
    """Return visible TensorFlow GPUs."""
    import tensorflow as tf

    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except Exception:
                pass
    return gpus


def warn_if_no_gpu_and_confirm(context: str = "runtime") -> List[object]:
    """Warn and ask the user whether to continue when TensorFlow sees no GPU.

    Set environment variable ``FORCE_CPU=1`` to skip the interactive prompt
    and proceed on CPU automatically (useful for non-interactive evaluation).
    """
    import os

    gpus = check_tensorflow_gpu()
    if gpus:
        print(f"✓ GPU check before {context}: TensorFlow sees {len(gpus)} GPU(s).")
        return gpus

    if os.environ.get("FORCE_CPU", "").strip() in ("1", "true", "yes"):
        print(f"⚠️  No GPU detected for {context} — continuing on CPU (FORCE_CPU=1).")
        return gpus

    print("\n" + "!" * 80)
    print("!!! INICIANDO SEM GPU, CONTINUAR? !!!")
    print(f"Contexto: {context}")
    print("!" * 80)
    try:
        answer = input("[y/N]: ").strip().lower()
    except EOFError as exc:
        raise RuntimeError(
            "GPU is not working and no interactive confirmation was possible. Aborting."
        ) from exc

    if answer not in {"y", "yes"}:
        raise RuntimeError("Aborted by user because GPU is unavailable.")
    return gpus
