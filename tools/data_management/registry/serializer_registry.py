from __future__ import annotations

from tools.data_management.registry.base import Registry
from tools.data_management.serializers.base import TargetSerializer


def default_serializer_registry() -> Registry[TargetSerializer]:
    from tools.data_management.serializers.formula_latex import LatexPlainSerializer
    from tools.data_management.serializers.layout_mineru import MinerULayoutSerializer
    from tools.data_management.serializers.table_otsl import EnhancedOTSLSerializer
    from tools.data_management.serializers.text_plain import PlainTextSerializer

    registry: Registry[TargetSerializer] = Registry()
    for serializer in (
        MinerULayoutSerializer(),
        EnhancedOTSLSerializer(),
        LatexPlainSerializer(),
        PlainTextSerializer(),
    ):
        registry.register(serializer.name, serializer)
    return registry
