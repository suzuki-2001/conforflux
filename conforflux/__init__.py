from conforflux.config import ConforFluxConfig
from conforflux.hooks import ConforFluxHooks
from conforflux.state import ConforFluxGuidanceState, build_state_from_trunk

__all__ = [
    "ConforFluxConfig",
    "ConforFluxHooks",
    "ConforFluxGuidanceState",
    "build_state_from_trunk",
    "ConforFluxCallback",
]

__version__ = "0.1.0"


def __getattr__(name):
    if name == "ConforFluxCallback":
        from conforflux.callback import ConforFluxCallback
        return ConforFluxCallback
    raise AttributeError(name)
