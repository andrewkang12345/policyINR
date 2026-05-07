from .cvae import CVAE
from .cvae_rnn import CVAERNN
from .vqvae import VQVAE
from .state_action_bem import StateActionBEM
from .inr_transformer import (
    INRTransformer,
    INRTransformerHistoryConditioned,
    INRTransformerFittedLatent,
    INRTransformerInferLatent,
)
from .inr_diffusion import INRDiffusion, INRDiffusionHistoryConditioned
from .base import RepresentationModel
from utils.registry import MODELS


def build_model(name: str, **kwargs) -> RepresentationModel:
    return MODELS.get(name)(**kwargs)


__all__ = [
    "CVAE",
    "CVAERNN",
    "VQVAE",
    "StateActionBEM",
    "INRTransformer",
    "INRTransformerHistoryConditioned",
    "INRTransformerFittedLatent",
    "INRTransformerInferLatent",
    "INRDiffusion",
    "INRDiffusionHistoryConditioned",
    "RepresentationModel",
    "build_model",
    "MODELS",
]
