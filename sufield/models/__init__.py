from . import res16unet

MODELS = []

_ZOO_FACTORY = lambda module: {
    name: getattr(module, name) for name in dir(module)
}
MODEL_ZOO = {}
MODEL_ZOO = {**MODEL_ZOO, **_ZOO_FACTORY(res16unet)}

