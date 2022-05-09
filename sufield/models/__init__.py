# from sufield.models.trilateral_crf import TrilateralCRF
from . import res16unet, resunet

MODELS = []


def add_models(module):
    MODELS.extend([getattr(module, a) for a in dir(module) if 'Net' in a])


add_models(resunet)
add_models(res16unet)


def get_models():
    '''Returns a tuple of sample models.'''
    return MODELS


def load_model(name):
    '''
    Creates and returns an instance of the model given its class name.
    '''
    # Find the model class from its name
    all_models = get_models()
    mdict = {model.__name__: model for model in all_models}
    if name not in mdict:
        print('Invalid model index. Options are:')
        # Display a list of valid model names
        for model in all_models:
            print(f'\t* {model.__name__}')
        return None
    net_class = mdict[name]

    return net_class
