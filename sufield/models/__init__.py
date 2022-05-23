from . import res16unet, resunet

MODELS = []

_ZOO_FACTORY = lambda module: {
    name: getattr(module, name) for name in dir(module) if hasattr(getattr(module, name), 'TRANSFORM')
}
MODEL_ZOO = {}
MODEL_ZOO = {**MODEL_ZOO, **_ZOO_FACTORY(resunet)}
MODEL_ZOO = {**MODEL_ZOO, **_ZOO_FACTORY(res16unet)}

