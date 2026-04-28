from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from typing import Any, Literal


JsonDict = dict[str, Any]
SCHEMA_VERSION = "1.0.0"


def json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)


def stable_hash(value: Any, *, length: int = 16) -> str:
    payload = value if isinstance(value, str) else json_dumps(value)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def stable_id(prefix: str, *parts: Any) -> str:
    safe_parts = [str(part).replace("/", "_").replace(":", "_") for part in parts if part is not None and str(part) != ""]
    return ":".join([prefix, *safe_parts])


def to_plain(value: Any) -> Any:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict)):
        value = value.tolist()
    if isinstance(value, list | tuple):
        return [to_plain(item) for item in value]
    if isinstance(value, dict):
        return {str(key): to_plain(item) for key, item in value.items()}
    return value


def require_non_empty(value: str | None, field_name: str) -> str:
    if not value:
        raise ValueError(f"{field_name} is required")
    return value


@dataclass(slots=True)
class CanonicalDocument:
    document_id: str
    source_name: str
    source_document_id: str
    document_path: str
    document_type: str
    num_pages: int
    language: str | None = None
    domain: str | None = None
    checksum: str | None = None
    metadata: JsonDict = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        require_non_empty(self.document_id, "document_id")
        require_non_empty(self.source_name, "source_name")
        require_non_empty(self.source_document_id, "source_document_id")
        if self.num_pages <= 0:
            raise ValueError("num_pages must be positive")

    def to_dict(self) -> JsonDict:
        self.validate()
        return asdict(self)


@dataclass(slots=True)
class CanonicalPage:
    page_id: str
    document_id: str
    source_name: str
    source_page_id: str
    page_index: int
    page_image_asset_id: str
    width: int
    height: int
    rotation: float = 0.0
    attributes: JsonDict = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        require_non_empty(self.page_id, "page_id")
        require_non_empty(self.document_id, "document_id")
        require_non_empty(self.source_name, "source_name")
        if self.page_index < 0:
            raise ValueError("page_index must be non-negative")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("page width and height must be positive")

    def to_dict(self) -> JsonDict:
        self.validate()
        return asdict(self)


@dataclass(slots=True)
class CanonicalRegion:
    region_id: str
    page_id: str
    document_id: str
    source_name: str
    source_annotation_id: str
    bbox: list[float]
    rotation: float
    element_type: str
    category: str | None = None
    subcategory: str | None = None
    reading_order: int | None = None
    crop_asset_id: str | None = None
    quality_flags: list[str] = field(default_factory=list)
    metadata: JsonDict = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        require_non_empty(self.region_id, "region_id")
        require_non_empty(self.page_id, "page_id")
        require_non_empty(self.document_id, "document_id")
        require_non_empty(self.source_name, "source_name")
        require_non_empty(self.element_type, "element_type")
        if len(self.bbox) != 4:
            raise ValueError("bbox must contain [x1, y1, x2, y2]")
        x1, y1, x2, y2 = [float(value) for value in self.bbox]
        if x1 >= x2 or y1 >= y2:
            raise ValueError("bbox must satisfy x1 < x2 and y1 < y2")

    def to_dict(self) -> JsonDict:
        self.validate()
        return asdict(self)


@dataclass(slots=True)
class CanonicalTaskRecord:
    record_id: str
    task: str
    source_name: str
    document_id: str
    page_id: str
    region_id: str | None
    image_asset_id: str
    target: JsonDict
    category: str | None = None
    subcategory: str | None = None
    language: str | None = None
    quality_flags: list[str] = field(default_factory=list)
    provenance: JsonDict = field(default_factory=dict)
    metadata: JsonDict = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        require_non_empty(self.record_id, "record_id")
        require_non_empty(self.task, "task")
        require_non_empty(self.source_name, "source_name")
        require_non_empty(self.document_id, "document_id")
        require_non_empty(self.page_id, "page_id")
        require_non_empty(self.image_asset_id, "image_asset_id")
        if not self.target:
            raise ValueError("target is required")

    def to_dict(self) -> JsonDict:
        self.validate()
        return asdict(self)


@dataclass(slots=True)
class AssetRecord:
    asset_id: str
    asset_type: str
    source_name: str
    document_id: str | None
    page_id: str | None
    region_id: str | None
    task: str | None
    path: str
    width: int
    height: int
    format: str
    checksum: str | None = None
    parent_asset_id: str | None = None
    transform_spec_hash: str | None = None
    coordinate_space: str = "canonical_page_pixel_xyxy"
    transform: JsonDict = field(default_factory=dict)
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        require_non_empty(self.asset_id, "asset_id")
        require_non_empty(self.asset_type, "asset_type")
        require_non_empty(self.source_name, "source_name")
        require_non_empty(self.path, "path")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("asset width and height must be positive")

    def to_dict(self) -> JsonDict:
        self.validate()
        return asdict(self)


@dataclass(slots=True)
class ViewRecord:
    id: str
    stage: Literal["sft", "rlvr", "eval"]
    task: str
    image_path: str
    prompt: list | str
    label: str
    source_name: str
    document_id: str
    page_id: str
    region_id: str | None
    canonical_record_id: str
    canonical_image_asset_id: str
    target_format: str
    prompt_template_id: str
    split: Literal["train", "val", "test"]
    view_image_asset_id: str | None = None
    image_bytes: bytes | None = None
    images: list | None = None
    system_prompt: str | None = None
    messages: str | None = None
    data_source: str | None = None
    extra_info: JsonDict = field(default_factory=dict)
    metadata: JsonDict = field(default_factory=dict)
    reward_model: JsonDict | None = None
    reward_profile_id: str | None = None
    reward_payload: JsonDict | None = None
    reward_payload_id: str | None = None
    reward_payload_path: str | None = None
    answer_key: str | None = None
    verifier_metadata: JsonDict | None = None
    schema_version: str = SCHEMA_VERSION

    def validate(self) -> None:
        for field_name in (
            "id",
            "stage",
            "task",
            "image_path",
            "prompt",
            "label",
            "source_name",
            "document_id",
            "page_id",
            "canonical_record_id",
            "canonical_image_asset_id",
            "target_format",
            "prompt_template_id",
            "split",
        ):
            require_non_empty(getattr(self, field_name), field_name)
        if self.stage == "rlvr":
            require_non_empty(self.reward_profile_id, "reward_profile_id")
            require_non_empty(self.data_source, "data_source")
            if self.reward_payload is None and not self.reward_payload_id and not self.reward_payload_path:
                raise ValueError("rlvr view records require reward payload information")

    def to_dict(self) -> JsonDict:
        self.validate()
        return asdict(self)


@dataclass(slots=True)
class RewardResult:
    score: float
    normalized_score: float
    passed: bool | None
    details: JsonDict
    error: str | None = None

    def to_dict(self) -> JsonDict:
        return asdict(self)
