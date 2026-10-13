"""Central Model Registry for Tamban Fish Quality Inspection System.

Provides discovery, metadata validation, path safety checks, workflow
compatibility validation, and selection persistence for all available models.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Any, Mapping

from src.preprocessing.audit_v7_exports import SOURCE_CLASSES
from src.preprocessing.dataset_utils import project_root


LOGGER = logging.getLogger(__name__)
DEFAULT_REGISTRY_PATH = "models/MODEL_REGISTRY.json"
SETTINGS_FILE_PATH = "configs/active_model.json"
DEFAULT_MODEL2_ID = "yolo26m_partdet"
DEFAULT_MODEL1_ID = "fish_parent_detector"


@dataclass(frozen=True)
class ModelEntry:
    """Registered model metadata definition."""

    id: str
    role: str
    display_name: str
    name: str
    path: str
    sha256: str | None
    size_bytes: int
    architecture: str
    framework: str
    task: str
    input_size: int
    classes: int
    class_names: Mapping[int | str, str]
    adapter: str
    is_default_model1: bool = False
    is_default_model2: bool = False
    is_compatible_grading: bool = True
    incompatibility_reason: str | None = None
    compatible_workflows: tuple[str, ...] = field(default_factory=tuple)
    availability: str = "available"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "role": self.role,
            "display_name": self.display_name,
            "name": self.name,
            "path": self.path,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
            "architecture": self.architecture,
            "framework": self.framework,
            "task": self.task,
            "input_size": self.input_size,
            "classes": self.classes,
            "class_names": dict(self.class_names),
            "adapter": self.adapter,
            "is_default_model1": self.is_default_model1,
            "is_default_model2": self.is_default_model2,
            "is_compatible_grading": self.is_compatible_grading,
            "incompatibility_reason": self.incompatibility_reason,
            "compatible_workflows": list(self.compatible_workflows),
            "availability": self.availability,
            "notes": self.notes,
        }


class ModelRegistryError(ValueError):
    """Raised when model registration or path resolution fails."""


class ModelRegistry:
    """Central registry holding metadata for all discovered models."""

    def __init__(self, repo_root: Path | None = None, registry_file: Path | str | None = None) -> None:
        self.repo_root = (repo_root or project_root()).resolve()
        self.models_dir = (self.repo_root / "models").resolve()
        self.registry_path = (self.repo_root / (registry_file or DEFAULT_REGISTRY_PATH)).resolve()
        self.settings_path = (self.repo_root / SETTINGS_FILE_PATH).resolve()
        self._entries: dict[str, ModelEntry] = {}
        self._default_model1_id = DEFAULT_MODEL1_ID
        self._default_model2_id = DEFAULT_MODEL2_ID
        self.load_registry()

    def load_registry(self) -> None:
        """Load and validate the JSON registry file."""
        if not self.registry_path.is_file():
            LOGGER.warning("Model registry file not found at %s", self.registry_path)
            self._entries.clear()
            return

        with self.registry_path.open("r", encoding="utf-8") as stream:
            payload = json.load(stream)

        self._default_model1_id = str(payload.get("default_model1", DEFAULT_MODEL1_ID))
        self._default_model2_id = str(payload.get("default_model2", DEFAULT_MODEL2_ID))

        models_list = payload.get("models", [])
        entries: dict[str, ModelEntry] = {}
        for item in models_list:
            model_id = str(item.get("id"))
            entry = ModelEntry(
                id=model_id,
                role=str(item.get("role", "")),
                display_name=str(item.get("display_name", model_id)),
                name=str(item.get("name", "")),
                path=str(item.get("path", "")),
                sha256=item.get("sha256"),
                size_bytes=int(item.get("size_bytes", 0)),
                architecture=str(item.get("architecture", "")),
                framework=str(item.get("framework", "ultralytics")),
                task=str(item.get("task", "detect")),
                input_size=int(item.get("input_size", 640)),
                classes=int(item.get("classes", len(SOURCE_CLASSES))),
                class_names=item.get("class_names", dict(SOURCE_CLASSES)),
                adapter=str(item.get("adapter", "yolo_detection")),
                is_default_model1=bool(item.get("is_default_model1", False)),
                is_default_model2=bool(item.get("is_default_model2", False)),
                is_compatible_grading=bool(item.get("is_compatible_grading", True)),
                incompatibility_reason=item.get("incompatibility_reason"),
                compatible_workflows=tuple(item.get("compatible_workflows", ())),
                availability=str(item.get("availability", "available")),
                notes=str(item.get("notes", "")),
            )
            entries[model_id] = entry

        self._entries = entries

    def get_model(self, model_id: str) -> ModelEntry | None:
        return self._entries.get(model_id)

    def list_models(self, role: str | None = None) -> list[ModelEntry]:
        if role is None:
            return list(self._entries.values())
        return [entry for entry in self._entries.values() if entry.role == role]

    def list_model2_candidates(self) -> list[ModelEntry]:
        """Return all models suitable for Model 2 selection."""
        return [
            entry for entry in self._entries.values()
            if entry.role in {"model2_part_quality", "part_preview"}
        ]

    def get_default_model2_id(self) -> str:
        return self._default_model2_id

    def get_default_model1_id(self) -> str:
        return self._default_model1_id

    def get_persisted_model2_id(self) -> str:
        """Retrieve persisted user selection or the default YOLO26."""
        if self.settings_path.is_file():
            try:
                data = json.loads(self.settings_path.read_text(encoding="utf-8"))
                selected = data.get("selected_model2")
                if selected and selected in self._entries:
                    return str(selected)
            except Exception as exc:
                LOGGER.warning("Could not read persisted model settings: %s", exc)
        return self.get_default_model2_id()

    def persist_model2_selection(self, model_id: str) -> None:
        """Persist the operator's Model 2 selection."""
        if model_id not in self._entries:
            raise ModelRegistryError(f"Cannot persist unknown model ID: {model_id}")
        try:
            self.settings_path.parent.mkdir(parents=True, exist_ok=True)
            self.settings_path.write_text(
                json.dumps({"selected_model2": model_id}, indent=2),
                encoding="utf-8",
            )
        except OSError as exc:
            LOGGER.warning("Could not persist model selection: %s", exc)

    def resolve_checkpoint_path(self, model_id_or_entry: str | ModelEntry) -> Path:
        """Resolve checkpoint path relative to project root with boundary check."""
        entry = (
            model_id_or_entry
            if isinstance(model_id_or_entry, ModelEntry)
            else self.get_model(model_id_or_entry)
        )
        if entry is None:
            raise ModelRegistryError(f"Model ID '{model_id_or_entry}' is not registered.")

        rel_path = Path(entry.path)
        resolved = (self.repo_root / rel_path).resolve()

        # Validate that path stays within the repo root / models directory
        try:
            resolved.relative_to(self.models_dir)
        except ValueError as exc:
            raise ModelRegistryError(
                f"Security check failed: model checkpoint path '{entry.path}' is outside models directory."
            ) from exc

        if not resolved.is_file():
            raise FileNotFoundError(f"Model checkpoint file does not exist: {resolved}")

        return resolved

    def validate_compatibility(self, model_id: str, workflow: str = "model2_grading") -> tuple[bool, str | None]:
        """Check whether a model can fulfill the requested workflow contract."""
        entry = self.get_model(model_id)
        if entry is None:
            return False, f"Model ID '{model_id}' is not in the registry."

        if workflow in {"model2_grading", "live_inspection", "analyze_image"}:
            if not entry.is_compatible_grading:
                reason = entry.incompatibility_reason or "Model is marked incompatible with the grading workflow."
                return False, reason
            if entry.role not in {"model2_part_quality", "part_preview"}:
                return False, f"Model role '{entry.role}' cannot act as Model 2 part detector."
            return True, None

        if workflow == "model1_whole_fish_detection":
            if entry.role != "model1_fish_detector":
                return False, f"Model role '{entry.role}' cannot act as Model 1 whole-fish detector."
            return True, None

        if workflow in entry.compatible_workflows:
            return True, None

        return False, f"Model '{entry.display_name}' does not support workflow '{workflow}'."


_REGISTRY_INSTANCE: ModelRegistry | None = None


def get_model_registry() -> ModelRegistry:
    """Return singleton ModelRegistry instance."""
    global _REGISTRY_INSTANCE
    if _REGISTRY_INSTANCE is None:
        _REGISTRY_INSTANCE = ModelRegistry()
    return _REGISTRY_INSTANCE
