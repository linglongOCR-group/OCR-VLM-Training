from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal

JsonDict = dict[str, Any]


def _to_plain(value: Any) -> Any:
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes, dict)):
        value = value.tolist()
    if isinstance(value, list | tuple):
        return [_to_plain(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    return value


def _get(raw: JsonDict, key: str, default: Any = None) -> Any:
    value = _to_plain(raw.get(key, default))
    return default if value is None else value


@dataclass(slots=True)
class BaseRecord:
    sample_id: str
    image_uri: str
    width_px: int
    height_px: int
    task_type: str
    source_type: str | None = None
    provenance: JsonDict = field(default_factory=dict)
    difficulty: str | None = None
    is_synthetic: bool = False
    metadata: JsonDict = field(default_factory=dict)

    @property
    def record_type(self) -> str:
        raise NotImplementedError

    def to_dict(self) -> JsonDict:
        data = asdict(self)
        data["record_type"] = self.record_type
        return data

    def validate_common(self) -> None:
        if not self.sample_id:
            raise ValueError("sample_id is required")
        if not self.image_uri:
            raise ValueError("image_uri is required")
        if self.width_px <= 0:
            raise ValueError("width_px must be positive")
        if self.height_px <= 0:
            raise ValueError("height_px must be positive")
        if not self.task_type:
            raise ValueError("task_type is required")


@dataclass(slots=True)
class DocumentRecord(BaseRecord):
    doc_id: str | None = None
    num_pages: int = 0
    language: str | None = None
    domain: str | None = None
    checksum: str | None = None
    schema_version: str = "1.0.0"

    @property
    def record_type(self) -> Literal["document"]:
        return "document"

    def validate(self) -> None:
        self.validate_common()


@dataclass(slots=True)
class PageRecord(BaseRecord):
    blocks: list[JsonDict] = field(default_factory=list)
    reading_order: list[str] = field(default_factory=list)
    merged_markdown: str = ""
    doc_id: str | None = None
    page_index: int | None = None

    @property
    def record_type(self) -> Literal["page"]:
        return "page"

    def validate(self) -> None:
        self.validate_common()


@dataclass(slots=True)
class RegionRecord(BaseRecord):
    region_type: str = ""
    targets: JsonDict = field(default_factory=dict)
    source_page_ref: JsonDict | None = None
    bbox_px: list[float] | None = None

    @property
    def record_type(self) -> Literal["region"]:
        return "region"

    def validate(self) -> None:
        self.validate_common()
        if not self.region_type:
            raise ValueError("region_type is required for region records")
        if not self.targets:
            raise ValueError("targets is required for region records")


@dataclass(slots=True)
class DetectionRecord(BaseRecord):
    instances: list[JsonDict] = field(default_factory=list)
    source_page_ref: JsonDict | None = None

    @property
    def record_type(self) -> Literal["detection"]:
        return "detection"

    def validate(self) -> None:
        self.validate_common()
        if self.instances is None:
            raise ValueError("instances must be a list for detection records")
        for instance in self.instances:
            if "category" not in instance:
                raise ValueError("each detection instance requires category")
            if "bbox_px" not in instance and "polygon_px" not in instance:
                raise ValueError("each detection instance requires bbox_px or polygon_px")


def validate_record(raw: JsonDict | BaseRecord) -> BaseRecord:
    if isinstance(raw, BaseRecord):
        raw.validate()
        return raw

    raw = _to_plain(raw)
    record_type = raw.get("record_type")
    common = {
        "sample_id": _get(raw, "sample_id", ""),
        "image_uri": _get(raw, "image_uri", ""),
        "width_px": int(_get(raw, "width_px", 0)),
        "height_px": int(_get(raw, "height_px", 0)),
        "task_type": _get(raw, "task_type", ""),
        "source_type": _get(raw, "source_type"),
        "provenance": _get(raw, "provenance", {}) or {},
        "difficulty": _get(raw, "difficulty"),
        "is_synthetic": bool(_get(raw, "is_synthetic", False) or False),
        "metadata": _get(raw, "metadata", {}) or {},
    }

    if record_type == "document":
        record: BaseRecord = DocumentRecord(
            **common,
            doc_id=_get(raw, "doc_id"),
            num_pages=int(_get(raw, "num_pages", 0)),
            language=_get(raw, "language"),
            domain=_get(raw, "domain"),
            checksum=_get(raw, "checksum"),
        )
    elif record_type == "page":
        record = PageRecord(
            **common,
            blocks=list(_get(raw, "blocks", []) or []),
            reading_order=list(_get(raw, "reading_order", []) or []),
            merged_markdown=_get(raw, "merged_markdown", "") or "",
            doc_id=_get(raw, "doc_id"),
            page_index=_get(raw, "page_index"),
        )
    elif record_type == "region":
        record = RegionRecord(
            **common,
            region_type=_get(raw, "region_type", "") or "",
            targets=_get(raw, "targets", {}) or {},
            source_page_ref=_get(raw, "source_page_ref"),
            bbox_px=_get(raw, "bbox_px"),
        )
    elif record_type == "detection":
        record = DetectionRecord(
            **common,
            instances=list(_get(raw, "instances", []) or []),
            source_page_ref=_get(raw, "source_page_ref"),
        )
    else:
        raise ValueError(f"unsupported record_type: {record_type!r}")

    record.validate()
    return record
