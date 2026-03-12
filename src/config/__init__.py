# -*- coding: utf-8 -*-
"""src.config — Configuration defaults, schemas and I/O."""

from src.config.defaults import (  # noqa: F401 — public re-exports
    TRAINING_DEFAULTS,
    MODEL_DEFAULTS,
    DATA_REDUCTION_DEFAULTS,
    ANALYSIS_DEFAULTS,
    FALLBACK_STATE_RUN,
)
from src.config.io import (  # noqa: F401
    load_config,
    save_json,
    merge_overrides,
    ensure_state_run_compat,
)
from src.config.schema import (  # noqa: F401
    TrainConfig,
    DataConfig,
    AnalysisConfig,
    EvalProtocolConfig,
    RunMeta,
)
from src.config.overrides import RunOverrides  # noqa: F401
from src.config.runtime import (  # noqa: F401
    TrainingRuntime,
    EvaluationRuntime,
    build_training_runtime,
    build_evaluation_runtime,
)
