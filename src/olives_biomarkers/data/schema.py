"""Canonical label schema: maps verbatim OLIVES parquet columns to stable names.

The parquet column names contain spaces, slashes and parentheses (for example
``"Atrophy / thinning of retinal layers"``). Renaming them ad hoc across the
codebase invites silent mismatches, so every rename goes through
:class:`LabelSchema`, which is loaded from ``configs/label_schema.yaml``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from olives_biomarkers.utils.io import YamlIO


@dataclass(frozen=True)
class LabelDefinition:
    """One biomarker label."""

    key: str
    source_column: str
    display_name: str
    vip_cup_code: str | None = None


class SchemaValidationError(RuntimeError):
    """Raised when the data on disk does not match the declared schema."""


class LabelSchema:
    """Resolved view of the label schema for one target set.

    Args:
        payload: Parsed contents of ``label_schema.yaml``.
        target_set: Either ``"six"`` or ``"sixteen"``.
        config_name: Which HF config the identifier/clinical columns come from.
    """

    def __init__(
        self,
        payload: dict[str, Any],
        target_set: str = "sixteen",
        config_name: str = "disease_classification",
    ) -> None:
        if target_set not in {"six", "sixteen"}:
            raise ValueError(f"target_set must be 'six' or 'sixteen', got {target_set!r}")
        self._payload = payload
        self.target_set = target_set
        self.config_name = config_name

        block = payload.get(target_set)
        if not block:
            raise SchemaValidationError(f"label schema has no '{target_set}' section")

        self.labels: list[LabelDefinition] = [
            LabelDefinition(
                key=key,
                source_column=spec["source"],
                display_name=spec.get("display", key),
                vip_cup_code=spec.get("vip_cup_code"),
            )
            for key, spec in block.items()
        ]

        source_block = payload.get("source", {}).get(config_name)
        if source_block is None:
            raise SchemaValidationError(f"label schema has no source block for {config_name!r}")
        self.identifier_columns: dict[str, str] = dict(source_block.get("identifier_columns", {}))
        self.clinical_columns: dict[str, str] = dict(source_block.get("clinical_columns", {}))
        self.disease_column: str | None = source_block.get("disease_column")
        self.image_column: str = source_block.get("image_column", "Image")
        self.disease_label_map: dict[int, str] = {
            int(k): v for k, v in payload.get("disease_label_map", {}).items()
        }
        self.unresolved_fields: dict[str, Any] = payload.get("unresolved_fields", {})

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    @classmethod
    def from_yaml(
        cls,
        path: str | Path,
        target_set: str = "sixteen",
        config_name: str = "disease_classification",
    ) -> LabelSchema:
        """Load the schema from a YAML file."""
        return cls(YamlIO.read(path), target_set=target_set, config_name=config_name)

    # ------------------------------------------------------------------
    # accessors
    # ------------------------------------------------------------------
    @property
    def label_keys(self) -> list[str]:
        """Canonical snake_case label keys, in a fixed order."""
        return [label.key for label in self.labels]

    @property
    def source_columns(self) -> list[str]:
        """Verbatim parquet column names for the target labels, same order."""
        return [label.source_column for label in self.labels]

    @property
    def display_names(self) -> list[str]:
        """Human-readable label names for plots and tables."""
        return [label.display_name for label in self.labels]

    @property
    def n_labels(self) -> int:
        """Number of labels in this target set."""
        return len(self.labels)

    @property
    def source_to_key(self) -> dict[str, str]:
        """Mapping from parquet column name to canonical key."""
        return {label.source_column: label.key for label in self.labels}

    @property
    def key_to_display(self) -> dict[str, str]:
        """Mapping from canonical key to display name."""
        return {label.key: label.display_name for label in self.labels}

    def rename_map(self) -> dict[str, str]:
        """Full source-to-canonical rename map (identifiers, clinical, disease, labels)."""
        mapping = {source: key for key, source in self.identifier_columns.items()}
        mapping.update({source: key for key, source in self.clinical_columns.items()})
        if self.disease_column:
            mapping[self.disease_column] = "disease_label"
        mapping.update(self.source_to_key)
        return mapping

    def required_source_columns(self, include_labels: bool = True) -> list[str]:
        """Columns that must be present in the parquet file."""
        columns = list(self.identifier_columns.values()) + list(self.clinical_columns.values())
        if self.disease_column:
            columns.append(self.disease_column)
        if include_labels:
            columns.extend(self.source_columns)
        return columns

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------
    def validate_columns(self, available: list[str], include_labels: bool = True) -> None:
        """Raise :class:`SchemaValidationError` when declared columns are missing.

        Args:
            available: Column names actually present in the parquet schema.
            include_labels: Whether label columns are expected in this file.
        """
        missing = [c for c in self.required_source_columns(include_labels) if c not in available]
        if missing:
            raise SchemaValidationError(
                "Columns declared in label_schema.yaml are missing from the data.\n"
                f"  target_set : {self.target_set}\n"
                f"  config     : {self.config_name}\n"
                f"  missing    : {missing}\n"
                f"  available  : {sorted(available)}"
            )

    def describe(self) -> str:
        """Multi-line summary for the audit report and notebook output."""
        lines = [
            f"Target set    : {self.target_set} ({self.n_labels} labels)",
            f"HF config     : {self.config_name}",
            f"Identifiers   : {self.identifier_columns}",
            f"Clinical      : {self.clinical_columns}",
            f"Disease column: {self.disease_column}",
            "Labels:",
        ]
        for i, label in enumerate(self.labels, start=1):
            code = f" [{label.vip_cup_code}]" if label.vip_cup_code else ""
            lines.append(f"  {i:>2}. {label.key:<20}{code:<7} <- {label.source_column!r}")
        if self.unresolved_fields:
            lines.append("Unresolved fields (absent from this data source):")
            for name, spec in self.unresolved_fields.items():
                lines.append(f"  - {name}: {spec.get('status', 'unknown')}")
        return "\n".join(lines)
