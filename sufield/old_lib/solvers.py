import logging

from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import LambdaLR, StepLR
from configparser import SectionProxy as Sec


class LambdaStepLR(LambdaLR):

    def __init__(self, optimizer, lr_lambda, last_step=-1):
        super(LambdaStepLR, self).__init__(optimizer, lr_lambda, last_step)

    @property
    def last_step(self):
        """Use last_epoch for the step counter"""
        return self.last_epoch

    @last_step.setter
    def last_step(self, v):
        self.last_epoch = v


class PolyLR(LambdaStepLR):
    """DeepLab learning rate policy"""

    def __init__(self, optimizer, max_iter, power=0.9, last_step=-1):
        super(PolyLR, self).__init__(optimizer, lambda s: (1 - s / (max_iter + 1))**power, last_step)


class SquaredLR(LambdaStepLR):
    """ Used for SGD Lars"""

    def __init__(self, optimizer, max_iter, last_step=-1):
        super(SquaredLR, self).__init__(optimizer, lambda s: (1 - s / (max_iter + 1))**2, last_step)


class ExpLR(LambdaStepLR):

    def __init__(self, optimizer, step_size, gamma=0.9, last_step=-1):
        # (0.9 ** 21.854) = 0.1, (0.95 ** 44.8906) = 0.1
        # To get 0.1 every N using gamma 0.9, N * log(0.9)/log(0.1) = 0.04575749 N
        # To get 0.1 every N using gamma g, g ** N = 0.1 -> N * log(g) = log(0.1) -> g = np.exp(log(0.1) / N)
        super(ExpLR, self).__init__(optimizer, lambda s: gamma**(s / step_size), last_step)


def initialize_optimizer(params, conf: Sec):
    if conf['Optimizer'] == 'SGD':
        return SGD(
            params,
            lr=conf.getfloat('LearningRate'),
            momentum=conf.getfloat('SGDMomentum'),
            dampening=conf.getfloat('SGDDampening'),
            weight_decay=conf.getfloat('SGDWeightDecay'),
        )
    elif conf['optimizer'] == 'Adam':
        return Adam(
            params,
            lr=conf.getfloat('LearningRate'),
            betas=(conf.getfloat('AdamBeta1'), conf.getfloat('AdamBeta2')),
            weight_decay=conf.getfloat('AdamWeightDecay'),
        )
    else:
        logging.error('Optimizer type not supported')
        raise ValueError('Optimizer type not supported')


def initialize_scheduler(optimizer, conf: Sec, last_step=-1):
    if conf['Scheduler'] == 'StepLR':
        return StepLR(optimizer, step_size=conf.getint('StepLRSize'), gamma=conf.getfloat('StepLRGamma'), last_epoch=last_step)
    elif conf['Scheduler'] == 'PolyLR':
        return PolyLR(optimizer, max_iter=conf.getint('PolyLRMaxIter'), power=conf.getfloat('PolyLRPolyPower'), last_step=last_step)
    elif conf['Scheduler'] == 'SquaredLR':
        return SquaredLR(optimizer, max_iter=conf.getint('SquaredLRMaxIter'), last_step=last_step)
    elif conf['Scheduler'] == 'ExpLR':
        return ExpLR(optimizer, step_size=conf.getint('ExpLRStepSize'), gamma=conf.getfloat('ExpLRGamma'), last_step=last_step)
    else:
        logging.error('Scheduler not supported')
        raise ValueError('Scheduler not supported')

# TODO criterion
# def initialize_criterion(conf:Sec):
#     if conf['Criterion'] == 'CrossEntropyLoss':
        