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
