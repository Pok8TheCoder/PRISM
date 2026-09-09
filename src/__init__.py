# PRISM - Predictive Recurrent Infiltration State Model
try:
    from src.model_comparison import run_model_comparison
    __all__ = ["run_model_comparison"]
except ImportError:
    __all__ = []
