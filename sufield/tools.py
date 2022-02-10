# %%
from random import randint
import random
import matplotlib.pyplot as plt
import numpy as np
import torch

import matplotlib.colors as mcolors

COLORS = list(mcolors.TABLEAU_COLORS.values())

@plt.rc_context({'font.family': ['serif']})
def render_fit(statistics, fn_list, bins=100, fn_steps=100, dpi=400, save=None, stat_range=None):

    fig = plt.figure(dpi=dpi)
    if isinstance(statistics, torch.Tensor):
        stat = np.asarray(statistics.cpu())
    else:
        stat = np.asarray(statistics)

    if stat_range is None:
        stat_range = (stat.min(), stat.max())
    stat_min, stat_max = stat_range
    ######## Histogram part ########
    a, bin_edges = np.histogram(stat, density=True, bins=bins, range=(stat_min, stat_max))
    ax = fig.add_axes((0, 0, 1, 1))

    ax.bar(bin_edges[:-1], a, alpha=0.5, width=np.diff(bin_edges))

    ax.ticklabel_format(axis='y', style='sci', scilimits=(-1, 2))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ######## Curve part ########
    ax1 = ax.twinx()
    ax1.ticklabel_format(axis='y', style='sci', scilimits=(-1, 2))

    for idx, fn in enumerate(fn_list):
        x = list(stat_min + (stat_max - stat_min) / fn_steps * i for i in range(fn_steps))
        y = [fn(i) for i in x]
        ax1.plot(x, y, color=COLORS[idx])
        ax1.spines['top'].set_visible(False)
        ax1.spines['left'].set_visible(False)

    plt.draw()
    if save is not None:
        plt.axis('on')
        plt.savefig(save, transparent=True,bbox_inches='tight')
    else:
        plt.show()
    plt.close(fig)




if __name__ == '__main__':
    render_fit([(50+random.random()*100) for i in range(1000000)], [lambda x: x * 1, lambda x: x * 2], bins=2000)
# %%

# TODO extract a double mixture model fitter


from scipy.special import polygamma
from scipy import e, optimize
import torch

def generate_A(A, B):

    def func(x):
        polys = polygamma(0, [x[0] + x[1], x[0], x[1]])
        return [polys[0] - polys[1] - A, polys[0] - polys[2] - B]

    def jac(x):
        polys = polygamma(1, [x[0] + x[1], x[0], x[1]])
        return [[polys[0] - polys[1], polys[0]], [polys[0], polys[0] - polys[2]]]

    return func, jac

def gamma(x):
    return torch.lgamma(x).exp()

def pdf_spec(t, param):
    a, b = param
    return gamma(a + b) / gamma(a) / gamma(b) * t**(a - 1) * (1 - t)**(b - 1)

def fit_double_beta(stat, init=[(1.5, 20), (4, 10)]):
    init_spec=init
    weights_spec = torch.as_tensor(0.5).cuda()
    params_spec = torch.as_tensor(init_spec, dtype=torch.float64).cuda()
    for i in range(100):
        divisor = weights_spec * pdf_spec(stat, params_spec[0]) + (1 - weights_spec) * pdf_spec(
            stat, params_spec[1])

        r1 = weights_spec * pdf_spec(stat, params_spec[0]) / divisor
        r2 = (1 - weights_spec) * pdf_spec(stat, params_spec[1]) / divisor

        weights_spec = r1.sum() / len(stat)

        A1 = -(r1 * torch.log(stat)).sum() / r1.sum()
        B1 = -(r1 * torch.log(1 - stat)).sum() / r1.sum()
        A2 = -(r2 * torch.log(stat)).sum() / r2.sum()
        B2 = -(r2 * torch.log(1 - stat)).sum() / r2.sum()

        f, j = generate_A(A1.cpu(), B1.cpu())
        result = optimize.root(f, params_spec[0].cpu(), jac=j)
        params_spec[0][0], params_spec[0][1] = result.x
        f, j = generate_A(A2.cpu(), B2.cpu())
        result = optimize.root(f, params_spec[0].cpu(), jac=j)
        print(result)
        params_spec[1][0], params_spec[1][1] = result.x

    return params_spec
# %%
import numpy as np
PA = '/home/aidrive/tb5zhh/RL_min_dist.npy'
a = torch.as_tensor(np.load(PA)).cuda()
# %%
