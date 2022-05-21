from . import transforms
from .transforms import cf_collate_fn_factory

_ZOO_FACTORY = lambda module: {
    name: getattr(module, name) for name in dir(module) if hasattr(getattr(module, name), 'TRANSFORM')
}
TRANSFORM_ZOO = _ZOO_FACTORY(transforms)

import numpy as np

get_transform = lambda input_args: transforms.Compose(
    [TRANSFORM_ZOO[name](*eval(args)) for name, args in input_args.items()])
