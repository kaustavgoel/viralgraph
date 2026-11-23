"""Machine learning virality prediction and reach regression engine.

Includes strict temporal boundary guards, grouped dataset generation,
and interpretable classifiers with zero target leakage.
"""

from viralgraph.models.dataset_builder import (
    extract_strict_early_features,
    generate_grouped_cascade_dataset,
)
from viralgraph.models.virality_classifiers import (
    ModelEvaluationReport,
    evaluate_models_with_leakage_guards,
)

__all__ = [
    "extract_strict_early_features",
    "generate_grouped_cascade_dataset",
    "ModelEvaluationReport",
    "evaluate_models_with_leakage_guards",
]
