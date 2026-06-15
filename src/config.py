"""Experiment configuration: YAML schema, dataclasses, loader, and validation.

Loads a YAML config from ``configs/`` into a strongly-typed
:class:`ExperimentConfig` dataclass and applies the validation rules described
in ``specs/03_config_yaml_schema.md``. CLI overrides from ``train.py`` are
merged on top of the YAML values before validation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.utils.exceptions import ConfigValidationError

logger = logging.getLogger(__name__)


LABEL2ID_5TAG: Dict[str, int] = {
    "O": 0,
    "B-Chemical": 1,
    "I-Chemical": 2,
    "B-Disease": 3,
    "I-Disease": 4,
}
LABEL2ID_9TAG: Dict[str, int] = {
    "O": 0,
    "B-Chemical": 1,
    "I-Chemical": 2,
    "B-Disease": 3,
    "I-Disease": 4,
    "B-Virus": 5,
    "I-Virus": 6,
    "B-Gene": 7,
    "I-Gene": 8,
}
LABEL_SPACES: Dict[str, Dict[str, int]] = {
    "5tag": LABEL2ID_5TAG,
    "9tag": LABEL2ID_9TAG,
}

VALID_STRATEGIES = {"single", "sequential", "joint_uniform", "joint_noise_aware"}
VALID_BACKBONES = {
    "bert-base-uncased",
    "dmis-lab/biobert-base-cased-v1.2",
    "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext",
}


@dataclass
class DataSource:
    """A single training, validation, or test data source."""

    name: str
    path: str
    label_field: str
    source_tag: str


@dataclass
class ModelConfig:
    """Backbone selection and tokenization limits."""

    backbone: str
    num_labels: int
    max_length: int = 512


@dataclass
class DataConfig:
    """Dataset wiring for one experiment."""

    label_space: str
    train_sources: List[DataSource]
    validation_source: DataSource
    test_sources: List[DataSource]


@dataclass
class PhaseConfig:
    """Phase-specific training overrides for the sequential strategy."""

    train_source: str
    validation_source: str
    epochs: int
    learning_rate: float
    load_from_phase1: bool = False


@dataclass
class TrainingConfig:
    """Top-level training hyperparameters."""

    strategy: str
    epochs: int = 5
    batch_size: int = 16
    gradient_accumulation_steps: int = 1
    learning_rate: float = 2.0e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    lr_scheduler_type: str = "linear"
    fp16: bool = True
    gradient_checkpointing: bool = False
    phase1: Optional[PhaseConfig] = None
    phase2: Optional[PhaseConfig] = None
    source_weights: Optional[Dict[str, float]] = None


@dataclass
class EvaluationConfig:
    """Evaluation metrics and which test sets to score."""

    metrics: List[str] = field(default_factory=lambda: ["precision", "recall", "f1"])
    per_entity: bool = True
    test_sets_to_evaluate: List[str] = field(default_factory=list)


@dataclass
class OutputConfig:
    """Where checkpoints, logs and result JSON files land."""

    base_dir: str = "outputs"
    save_total_limit: int = 2
    logging_steps: int = 50
    eval_steps: int = 0
    save_strategy: str = "best"


@dataclass
class RuntimeConfig:
    """DataLoader and device knobs."""

    num_workers: int = 4
    pin_memory: bool = True
    device: str = "auto"


@dataclass
class ExperimentInfo:
    """Human-readable identification for the experiment."""

    id: int
    name: str
    description: str


@dataclass
class ExperimentConfig:
    """Fully resolved experiment configuration."""

    experiment: ExperimentInfo
    model: ModelConfig
    data: DataConfig
    training: TrainingConfig
    evaluation: EvaluationConfig
    output: OutputConfig
    runtime: RuntimeConfig
    seed: int = 42
    config_path: Optional[str] = None

    @property
    def label2id(self) -> Dict[str, int]:
        """Return the active label-to-id mapping based on the label space."""
        return dict(LABEL_SPACES[self.data.label_space])

    @property
    def id2label(self) -> Dict[int, str]:
        """Return the inverse mapping for the active label space."""
        return {v: k for k, v in self.label2id.items()}

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable view of the full config."""
        return asdict(self)


def _build_data_source(raw: Dict[str, Any], context: str) -> DataSource:
    """Construct a :class:`DataSource` and verify the required fields."""
    missing = [k for k in ("name", "path", "label_field", "source_tag") if k not in raw]
    if missing:
        raise ConfigValidationError(
            f"DataSource ({context}) is missing required keys: {missing}"
        )
    return DataSource(
        name=raw["name"],
        path=raw["path"],
        label_field=raw["label_field"],
        source_tag=raw["source_tag"],
    )


def _build_phase(raw: Optional[Dict[str, Any]], name: str) -> Optional[PhaseConfig]:
    if raw is None:
        return None
    missing = [k for k in ("train_source", "validation_source", "epochs", "learning_rate") if k not in raw]
    if missing:
        raise ConfigValidationError(
            f"PhaseConfig '{name}' is missing required keys: {missing}"
        )
    return PhaseConfig(
        train_source=raw["train_source"],
        validation_source=raw["validation_source"],
        epochs=int(raw["epochs"]),
        learning_rate=float(raw["learning_rate"]),
        load_from_phase1=bool(raw.get("load_from_phase1", name == "phase2")),
    )


def load_config(
    path: str | Path,
    overrides: Optional[Dict[str, Any]] = None,
) -> ExperimentConfig:
    """Load a YAML config, apply CLI overrides, and validate.

    Args:
        path: Path to the YAML config file.
        overrides: Optional flat dict of CLI overrides (``seed``, ``epochs``,
            ``batch_size``, ``learning_rate``, ``output_dir``).

    Returns:
        A validated :class:`ExperimentConfig` instance.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
        ConfigValidationError: If any validation rule fails.
    """
    yaml_path = Path(path)
    if not yaml_path.is_file():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with yaml_path.open("r", encoding="utf-8") as fp:
        raw = yaml.safe_load(fp) or {}

    if overrides:
        _apply_overrides(raw, overrides)

    config = _build_config(raw)
    config.config_path = str(yaml_path)
    _validate_config(config)
    return config


def _apply_overrides(raw: Dict[str, Any], overrides: Dict[str, Any]) -> None:
    """Mutate the raw YAML dict in-place with supported CLI overrides."""
    if "seed" in overrides and overrides["seed"] is not None:
        raw["seed"] = int(overrides["seed"])
    if "epochs" in overrides and overrides["epochs"] is not None:
        raw.setdefault("training", {})["epochs"] = int(overrides["epochs"])
    if "batch_size" in overrides and overrides["batch_size"] is not None:
        raw.setdefault("training", {})["batch_size"] = int(overrides["batch_size"])
    if "learning_rate" in overrides and overrides["learning_rate"] is not None:
        raw.setdefault("training", {})["learning_rate"] = float(overrides["learning_rate"])
    if "output_dir" in overrides and overrides["output_dir"] is not None:
        raw.setdefault("output", {})["base_dir"] = str(overrides["output_dir"])


def _build_config(raw: Dict[str, Any]) -> ExperimentConfig:
    """Materialize an :class:`ExperimentConfig` from a parsed YAML dict."""
    exp_raw = raw.get("experiment", {})
    model_raw = raw.get("model", {})
    data_raw = raw.get("data", {})
    train_raw = raw.get("training", {})
    eval_raw = raw.get("evaluation", {})
    out_raw = raw.get("output", {})
    runtime_raw = raw.get("runtime", {})

    train_sources = [_build_data_source(s, "train") for s in data_raw.get("train_sources", [])]
    validation_source = _build_data_source(data_raw.get("validation_source", {}), "validation")
    test_sources = [_build_data_source(s, "test") for s in data_raw.get("test_sources", [])]

    config = ExperimentConfig(
        experiment=ExperimentInfo(
            id=int(exp_raw["id"]),
            name=str(exp_raw["name"]),
            description=str(exp_raw.get("description", "")),
        ),
        model=ModelConfig(
            backbone=str(model_raw["backbone"]),
            num_labels=int(model_raw["num_labels"]),
            max_length=int(model_raw.get("max_length", 512)),
        ),
        data=DataConfig(
            label_space=str(data_raw["label_space"]),
            train_sources=train_sources,
            validation_source=validation_source,
            test_sources=test_sources,
        ),
        training=TrainingConfig(
            strategy=str(train_raw["strategy"]),
            epochs=int(train_raw.get("epochs", 5)),
            batch_size=int(train_raw.get("batch_size", 16)),
            gradient_accumulation_steps=int(train_raw.get("gradient_accumulation_steps", 1)),
            learning_rate=float(train_raw.get("learning_rate", 2.0e-5)),
            weight_decay=float(train_raw.get("weight_decay", 0.01)),
            warmup_ratio=float(train_raw.get("warmup_ratio", 0.1)),
            lr_scheduler_type=str(train_raw.get("lr_scheduler_type", "linear")),
            fp16=bool(train_raw.get("fp16", True)),
            gradient_checkpointing=bool(train_raw.get("gradient_checkpointing", False)),
            phase1=_build_phase(train_raw.get("phase1"), "phase1"),
            phase2=_build_phase(train_raw.get("phase2"), "phase2"),
            source_weights=train_raw.get("source_weights"),
        ),
        evaluation=EvaluationConfig(
            metrics=list(eval_raw.get("metrics", ["precision", "recall", "f1"])),
            per_entity=bool(eval_raw.get("per_entity", True)),
            test_sets_to_evaluate=list(eval_raw.get("test_sets_to_evaluate", [])),
        ),
        output=OutputConfig(
            base_dir=str(out_raw.get("base_dir", "outputs")),
            save_total_limit=int(out_raw.get("save_total_limit", 2)),
            logging_steps=int(out_raw.get("logging_steps", 50)),
            eval_steps=int(out_raw.get("eval_steps", 0)),
            save_strategy=str(out_raw.get("save_strategy", "best")),
        ),
        runtime=RuntimeConfig(
            num_workers=int(runtime_raw.get("num_workers", 4)),
            pin_memory=bool(runtime_raw.get("pin_memory", True)),
            device=str(runtime_raw.get("device", "auto")),
        ),
        seed=int(raw.get("seed", 42)),
    )
    return config


def _validate_config(config: ExperimentConfig) -> None:
    """Apply the validation rules from spec 03 section 5."""
    if config.experiment.id not in {1, 2, 3, 4, 5, 6}:
        raise ConfigValidationError(
            f"experiment.id must be in 1..6, got {config.experiment.id}."
        )

    if config.data.label_space not in LABEL_SPACES:
        raise ConfigValidationError(
            f"data.label_space must be one of {sorted(LABEL_SPACES)}, "
            f"got {config.data.label_space!r}."
        )
    expected_num_labels = len(LABEL_SPACES[config.data.label_space])
    if config.model.num_labels != expected_num_labels:
        raise ConfigValidationError(
            f"model.num_labels={config.model.num_labels} inconsistent with "
            f"label_space={config.data.label_space!r} (expected {expected_num_labels})."
        )

    if config.model.backbone not in VALID_BACKBONES:
        logger.warning(
            "Backbone %r is not in the known list %s. Proceeding, but verify the HF "
            "identifier carefully.",
            config.model.backbone,
            sorted(VALID_BACKBONES),
        )

    strategy = config.training.strategy
    if strategy not in VALID_STRATEGIES:
        raise ConfigValidationError(
            f"training.strategy must be one of {sorted(VALID_STRATEGIES)}, got {strategy!r}."
        )

    if strategy == "single" and len(config.data.train_sources) != 1:
        raise ConfigValidationError(
            f"strategy='single' requires exactly 1 train_source, got {len(config.data.train_sources)}."
        )
    if strategy == "sequential":
        if len(config.data.train_sources) != 2:
            raise ConfigValidationError(
                "strategy='sequential' requires exactly 2 train_sources."
            )
        if config.training.phase1 is None or config.training.phase2 is None:
            raise ConfigValidationError(
                "strategy='sequential' requires both training.phase1 and training.phase2."
            )
        names = {s.name for s in config.data.train_sources}
        for phase_name, phase in (("phase1", config.training.phase1), ("phase2", config.training.phase2)):
            if phase.train_source not in names:
                raise ConfigValidationError(
                    f"training.{phase_name}.train_source={phase.train_source!r} does not "
                    f"match any train_sources name."
                )
    if strategy.startswith("joint_"):
        if len(config.data.train_sources) != 2:
            raise ConfigValidationError(
                f"strategy={strategy!r} requires exactly 2 train_sources."
            )
        if not config.training.source_weights:
            raise ConfigValidationError(
                f"strategy={strategy!r} requires training.source_weights."
            )
        present_tags = {s.source_tag for s in config.data.train_sources}
        missing = present_tags - set(config.training.source_weights.keys())
        if missing:
            raise ConfigValidationError(
                f"source_weights is missing entries for source tags: {sorted(missing)}."
            )

    for src in [*config.data.train_sources, config.data.validation_source, *config.data.test_sources]:
        if not Path(src.path).is_file():
            raise ConfigValidationError(
                f"Data source path does not exist: {src.path} (source={src.name}). "
                f"Verify that the dataset has been downloaded and the YAML path is correct."
            )

    test_names = {s.name for s in config.data.test_sources}
    extra = [n for n in config.evaluation.test_sets_to_evaluate if n not in test_names]
    if extra:
        raise ConfigValidationError(
            f"evaluation.test_sets_to_evaluate references unknown test sources: {extra}. "
            f"Known test sources: {sorted(test_names)}."
        )

    for metric in config.evaluation.metrics:
        if metric not in {"precision", "recall", "f1"}:
            raise ConfigValidationError(
                f"Unknown metric {metric!r} in evaluation.metrics. Allowed: precision, recall, f1."
            )


def dump_config_snapshot(config: ExperimentConfig, target: str | Path) -> Path:
    """Write a YAML snapshot of the resolved config next to the run results.

    Args:
        config: The resolved configuration to snapshot.
        target: Destination file path; parent directory will be created.

    Returns:
        Absolute path of the written snapshot.
    """
    out_path = Path(target)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(config.to_dict(), fp, sort_keys=False)
    return out_path.resolve()
