from moviasai.forecasting.models import VehicleForecastingBaseModel, VehicleForecastingMultiHeadModel

_LAZY_IMPORTS = {
    "LinearTrendForecaster": "moviasai.forecasting.baseline_model",
    "ForecastingPredictor": "moviasai.forecasting.predictor",
    "TrainingPipeline": "moviasai.forecasting.training_pipeline",
    "MultiHeadTrainingPipeline": "moviasai.forecasting.training_pipeline",
    "MoETrainingPipeline": "moviasai.forecasting.training_pipeline",
    "VehicleForecastingSafeMoEModel": "moviasai.forecasting.models",
}


def __getattr__(name: str):
    if name in _LAZY_IMPORTS:
        import importlib
        mod = importlib.import_module(_LAZY_IMPORTS[name])
        return getattr(mod, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
