from .trainer import (
    FlaxTrainerForCausalLM,
    FlaxTrainerForMaskedLM,
    FlaxTrainerForSequenceClassification,
    FlaxTrainerForTokenClassification
)

__version__ = "0.0.0.8"

__all__ = [
    "FlaxTrainerForCausalLM", "FlaxTrainerForMaskedLM",
    "FlaxTrainerForSequenceClassification", "FlaxTrainerForTokenClassification"
]
