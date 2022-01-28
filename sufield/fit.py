# %%
import os
import numpy as np
import scipy.special as sp
import json
import torch
from IPython import embed
from scipy import e, optimize
from scipy.special import polygamma
from tqdm import tqdm
import matplotlib.pyplot as plt
from plyfile import PlyData

try:
    from .config import (CLASS_LABELS, CONF_FILE, SCANNET_COLOR_MAP, TRAIN_IDS, VALID_CLASS_IDS)
except:
    from config import (CLASS_LABELS, CONF_FILE, SCANNET_COLOR_MAP, TRAIN_IDS, VALID_CLASS_IDS)

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
# Move these lines to config files
UNC_DATA_PATH = '/home/aidrive/tb5zhh/3d_scene_understand/3DScanSeg/results'
ORIGIN_PATH = '/home/aidrive/tb5zhh/3d_scene_understand/data/full_mesh/train'
gen_base_path = lambda s: f'/home/aidrive/tb5zhh/3d_scene_understand/data/{s}/train'
SPEC_DATA_PATH = 'results/spec_predictions'
FIT_RESULT_PATH = 'results/fitting'
GENERATE_PATH = 'results/generate_datasets'
NUM_CLS = 20


def generate_A(A, B):

    def func(x):
        polys = polygamma(0, [x[0] + x[1], x[0], x[1]])
        return [polys[0] - polys[1] - A, polys[0] - polys[2] - B]

    def jac(x):
        polys = polygamma(1, [x[0] + x[1], x[0], x[1]])
        return [[polys[0] - polys[1], polys[0]], [polys[0], polys[0] - polys[2]]]

    return func, jac


def generate_C(C):

    def func(x):
        return [-polygamma(0, x[0]) + np.log(x[0]) - C]

    def jac(x):
        return [-polygamma(1, x[0]) + 1 / x[0]]

    return func, jac


def gamma(x):
    return torch.lgamma(x).exp()


def pdf(t, param_spec, param_unc):
    a, b = param_spec
    c, d = param_unc
    return gamma(a + b) / gamma(a) / gamma(b) * d**c / gamma(c) * t[0]**(a - 1) * (1 - t[0])**(b - 1) * t[1]**(c - 1) * e**(-d * t[1])


def pdf_spec(t, param):
    a, b = param
    return gamma(a + b) / gamma(a) / gamma(b) * t**(a - 1) * (1 - t)**(b - 1)


def pdf_unc(t, param):
    c, d = param
    return d**c / (gamma(c)) * t**(c - 1) * e**(-d * t)


class FitRunner():
    CACHE_VER = 1

    def __init__(self, num_point, debug=False) -> None:
        self.num_point = num_point
        self.fit_params = [None for _ in range(NUM_CLS)]
        self.debug = debug

    def load(self, cache=True):
        # records = [[np.ndarray((0)), np.ndarray((0))] for _ in range(NUM_CLS)]  # don't ever use [] * n again !!
        if cache and os.path.isfile(f'load_cache_{self.num_point}_v{self.CACHE_VER}.obj'):
            cache = torch.load(f'load_cache_{self.num_point}_v{self.CACHE_VER}.obj')
            self.spec_stat = [i.cuda() for i in cache['spec_stat']]
            self.unc_stat = [i.cuda() for i in cache['unc_stat']]
        else:
            spec_distances = [[] for _ in range(NUM_CLS)]
            unc_distances = [[] for _ in range(NUM_CLS)]
            for ply_name in tqdm(TRAIN_IDS, desc='load'):
                # spec_labels = np.asarray(spec_obj['labels'])[spec_mapping]
                spec_obj = torch.load(f"{SPEC_DATA_PATH}/{ply_name}_{self.num_point}.obj")
                spec_confidence = np.asarray(spec_obj['confidence'])

                unc_obj = np.asarray(torch.load(f'{UNC_DATA_PATH}/{self.num_point}/{ply_name}_unc.obj'))
                unc_mapping = np.asarray(torch.load(f'{UNC_DATA_PATH}/mappings/{ply_name}_mapping.obj')['inverse'])
                unc_labels = np.asarray(torch.load(f"{UNC_DATA_PATH}/{self.num_point}/{ply_name}_predicted.obj")[unc_mapping])
                unc_confidence = unc_obj[unc_mapping]

                for idx, cls_id in enumerate(VALID_CLASS_IDS):
                    # spec_selector = np.where(spec_labels == cls_id)[0]
                    unc_selector = np.where(unc_labels == idx)[0]  # ! use idx on purpose
                    spec_conf = spec_confidence[unc_selector].flatten()
                    unc_conf = unc_confidence[unc_selector][:, idx].flatten()

                    spec_distances[idx].append(spec_conf)
                    unc_distances[idx].append(unc_conf)

                    # records[idx][0] = np.hstack((records[idx][0], spec_conf))
                    # records[idx][1] = np.hstack((records[idx][1], unc_conf))
            self.spec_stat = [torch.as_tensor(np.hstack(spec_distances[i])).cuda() for i in range(NUM_CLS)]
            self.unc_stat = [torch.as_tensor(np.hstack(unc_distances[i])).cuda() for i in range(NUM_CLS)]

            for i in range(NUM_CLS):
                self.spec_stat[i] = (self.spec_stat[i] + 1e-10) / (self.spec_stat[i].max() + 1e-9)
                self.unc_stat[i] = (self.unc_stat[i] + 1e-10) / (self.unc_stat[i].max() + 1e-9)

            if cache:
                torch.save({
                    'spec_stat': [i.cpu() for i in self.spec_stat],
                    'unc_stat': [i.cpu() for i in self.unc_stat],
                }, f'load_cache_{self.num_point}_v{self.CACHE_VER}.obj')
        return self

    def fit(self,
            fitted_cls=[],
            fit_unc=True,
            fit_spec=True,
            cache=True,
            update=False,
            render=False,
            init_unc=[(2, 2), (1, 1)],
            init_spec=[(2, 1), (1.5, 2)],
            part=False):
        if part:
            it = fitted_cls
        else:
            it = range(NUM_CLS)
        for cls_id in it:
            if cache and os.path.isfile(f'{FIT_RESULT_PATH}/{self.num_point}/{cls_id}.json') and cls_id not in fitted_cls:
                print(f'{cls_id} use cache')
                with open(f'{FIT_RESULT_PATH}/{self.num_point}/{cls_id}.json') as f:
                    self.fit_params[cls_id] = json.load(f)
                if render:
                    self.render(cls_id, with_bar=True)
                continue
            print(f'fitting {cls_id}')
            # Init parameters, TODO check
            params_spec = torch.as_tensor(init_spec, dtype=torch.float64).cuda()
            weights_spec = torch.as_tensor(0.5).cuda()
            params_unc = torch.as_tensor(init_unc, dtype=torch.float64).cuda()
            weights_unc = torch.as_tensor(0.5).cuda()

            for _ in range(100):
                if fit_spec:
                    divisor = weights_spec * pdf_spec(self.spec_stat[cls_id], params_spec[0]) + (1 - weights_spec) * pdf_spec(
                        self.spec_stat[cls_id], params_spec[1])

                    r1 = weights_spec * pdf_spec(self.spec_stat[cls_id], params_spec[0]) / divisor
                    r2 = (1 - weights_spec) * pdf_spec(self.spec_stat[cls_id], params_spec[1]) / divisor

                    weights_spec = r1.sum() / len(self.spec_stat[cls_id])

                    A1 = -(r1 * torch.log(self.spec_stat[cls_id])).sum() / r1.sum()
                    B1 = -(r1 * torch.log(1 - self.spec_stat[cls_id])).sum() / r1.sum()
                    A2 = -(r2 * torch.log(self.spec_stat[cls_id])).sum() / r2.sum()
                    B2 = -(r2 * torch.log(1 - self.spec_stat[cls_id])).sum() / r2.sum()

                    f, j = generate_A(A1.cpu(), B1.cpu())
                    result = optimize.root(f, params_spec[0].cpu(), jac=j)
                    params_spec[0][0], params_spec[0][1] = result.x
                    if self.debug:
                        print(result.message)
                        print(result.x)
                        print(result)

                    f, j = generate_A(A2.cpu(), B2.cpu())
                    result = optimize.root(f, params_spec[0].cpu(), jac=j)
                    params_spec[1][0], params_spec[1][1] = result.x

                if fit_unc:
                    divisor = weights_unc * pdf_unc(self.unc_stat[cls_id], params_unc[0]) + (1 - weights_unc) * pdf_unc(self.unc_stat[cls_id], params_unc[1])
                    r1 = weights_unc * pdf_unc(self.unc_stat[cls_id], params_unc[0]) / divisor
                    r2 = (1 - weights_unc) * pdf_unc(self.unc_stat[cls_id], params_unc[1]) / divisor

                    weights_unc = r1.sum() / len(self.unc_stat[cls_id])

                    # %
                    C1 = torch.log((r1 * self.unc_stat[cls_id]).sum() / r1.sum()) - (r1 * torch.log(self.unc_stat[cls_id])).sum() / r1.sum()
                    C2 = torch.log((r2 * self.unc_stat[cls_id]).sum() / r2.sum()) - (r2 * torch.log(self.unc_stat[cls_id])).sum() / r2.sum()
                    D1 = r1.sum() / (r1 * self.unc_stat[cls_id]).sum()
                    D2 = r2.sum() / (r2 * self.unc_stat[cls_id]).sum()
                    # %
                    func, jac = generate_C(C1.cpu())
                    result = optimize.root(func, params_unc[0][0].cpu(), jac=jac)
                    params_unc[0][0] = result.x[0]
                    params_unc[0][1] = params_unc[0][0].detach() * D1
                    func, jac = generate_C(C2.cpu())
                    result = optimize.root(func, params_unc[1][0].cpu(), jac=jac)
                    params_unc[1][0] = result.x[0]
                    params_unc[1][1] = params_unc[1][0].detach() * D2

            os.makedirs(FIT_RESULT_PATH, exist_ok=True)
            if os.path.isfile(f'{FIT_RESULT_PATH}/{self.num_point}/{cls_id}.json'):
                with open(f'{FIT_RESULT_PATH}/{self.num_point}/{cls_id}.json') as f:
                    fit_result = json.load(f)
            else:
                fit_result = {'spec': [], 'unc': [], 'spec_weight': 0.5, 'unc_weight': 0.5}
            if fit_spec:
                fit_result['spec'] = [params_spec[0][0].item(), params_spec[0][1].item(), params_spec[1][0].item(), params_spec[1][1].item()]
                fit_result['spec_weight'] = weights_spec.item()
            if fit_unc:
                fit_result['unc'] = [params_unc[0][0].item(), params_unc[0][1].item(), params_unc[1][0].item(), params_unc[1][1].item()]
                fit_result['unc_weight'] = weights_unc.item()
            self.fit_params[cls_id] = fit_result
            if render:
                self.render(cls_id, with_bar=True)
            print(fit_result)
            if update:
                os.makedirs(f'{FIT_RESULT_PATH}/{self.num_point}/', exist_ok=True)
                with open(f'{FIT_RESULT_PATH}/{self.num_point}/{cls_id}.json', 'w') as f:
                    json.dump(fit_result, f, indent='    ')
        return self

    def render(self, cls_id, with_bar=False, save=False, n_bins=500, xlim=0.01, dpi=150):
        '''
        a: spectral
        b: unc
        '''
        fit_result = self.fit_params[cls_id]
        params_spec = (fit_result['spec'][0:2], fit_result['spec'][2:4])
        weight_spec = fit_result['spec_weight']
        params_unc = (fit_result['unc'][0:2], fit_result['unc'][2:4])
        weight_unc = fit_result['unc_weight']

        def calc_spec(t, param, weight):
            a, b = param
            return (sp.gamma(a + b) / sp.gamma(a) / sp.gamma(b) * t**(a - 1) * (1 - t)**(b - 1)) * weight

        def calc_unc(t, param, weight):
            c, d = param
            return (d**c / sp.gamma(c) * t**(c - 1) * e**(-d * t)) * weight

        # record_cpu = record.cpu()
        width_spec = (self.spec_stat[cls_id].max() - self.spec_stat[cls_id].min()) / n_bins
        width_unc = (self.unc_stat[cls_id].max() - self.unc_stat[cls_id].min()) / n_bins
        # width_unc = (xlim - stat_unc.min()) / n_bins

        width_spec = width_spec.cpu()
        width_unc = width_unc.cpu()

        xa = [self.spec_stat[cls_id].min().cpu() + i * width_spec for i in range(n_bins)]
        xb = [self.unc_stat[cls_id].min().cpu() + i * width_unc for i in range(n_bins)]

        fig = plt.figure(dpi=dpi)

        # spec
        a = torch.histc(self.spec_stat[cls_id], bins=n_bins, min=xa[0], max=xa[-1]).cpu()
        a1 = a / (a.sum() * width_spec)

        ax = fig.add_subplot(2, 4, 1)
        ax.bar(xa, a1, width=width_spec)
        # ax.set_xticks([])
        # ax.set_yticks([])
        # plt.show()
        # plt.cla()

        # raise Exception
        ax = fig.add_subplot(2, 4, 2)
        if with_bar:
            ax.bar(xa, a1, width=width_spec)
        ax.plot(xa, [calc_spec(i + width_spec / 2, params_spec[0], weight_spec) for i in xa], color='red')

        ax = fig.add_subplot(2, 4, 3)
        if with_bar:
            ax.bar(xa, a1, width=width_spec)
        ax.plot(xa, [calc_spec(i + width_spec / 2, params_spec[1], 1 - weight_spec) for i in xa], color='orange')

        ax = fig.add_subplot(2, 4, 4)
        if with_bar:
            ax.bar(xa, a1, width=width_spec)
        ax.plot(xa, [calc_spec(i + width_spec / 2, params_spec[0], weight_spec) for i in xa], color='red')
        ax.plot(xa, [calc_spec(i + width_spec / 2, params_spec[1], 1 - weight_spec) for i in xa], color='orange')
        ax.plot(xa, [calc_spec(i + width_spec / 2, params_spec[0], weight_spec) + calc_spec(i + width_spec / 2, params_spec[1], 1 - weight_spec) for i in xa],
                color='black')

        # unc
        b = torch.histc(self.unc_stat[cls_id], bins=n_bins, min=xb[0], max=xb[-1]).cpu()
        b1 = b / (b.sum() * width_unc)
        # print(b1[0])
        # print(b1[1])
        # print(b1[2])

        ax = fig.add_subplot(2, 4, 5)
        # ax.set_xlim([0, xlim])
        # ax.set_ylim([0, 2])
        ax.bar(xb, b1, width=width_unc)

        ax = fig.add_subplot(2, 4, 6)
        # ax.set_xlim([0, xlim])
        # ax.set_ylim([0, 2])
        if with_bar:
            ax.bar(xb, b1, width=width_unc)
        ax.plot(xb, [calc_unc(i + width_unc / 2, params_unc[0], weight_unc) for i in xb], color='red')

        ax = fig.add_subplot(2, 4, 7)
        # ax.set_xlim([0, xlim])
        # ax.set_ylim([0, 2])
        if with_bar:
            ax.bar(xb, b1, width=width_unc)
        ax.plot(xb, [calc_unc(i + width_unc / 2, params_unc[1], 1 - weight_unc) for i in xb], color='orange')

        ax = fig.add_subplot(2, 4, 8)
        # ax.set_xlim([0, xlim])
        # ax.set_ylim([0, 2])
        if with_bar:
            ax.bar(xb, b1, width=width_unc)
        ax.plot(xb, [calc_unc(i + width_unc / 2, params_unc[0], weight_unc) for i in xb], color='red')
        ax.plot(xb, [calc_unc(i + width_unc / 2, params_unc[1], 1 - weight_unc) for i in xb], color='orange')
        ax.plot(xb, [calc_unc(i + width_unc / 2, params_unc[0], weight_unc) + calc_unc(i + width_unc / 2, params_unc[1], 1 - weight_unc) for i in xb],
                color='black')

        # ax = fig.add_subplot(2, 2, 3,projection='3d',proj_type='ortho')
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])
        # X, Y = torch.meshgrid(torch.as_tensor(xa), torch.as_tensor(xb))
        # # X += 1e-7
        # # Y += 1e-7
        # Z = calc_a(X + 1e-7, params[0]) * calc_b(Y + 1e-7, params[0])

        # ax.plot_surface(X, Y, Z.numpy(), rstride=1, cstride=1, cmap='rainbow')
        # ax.view_init(elev=90., azim=-90.)

        # ax = fig.add_subplot(2, 2, 4,projection='3d',proj_type='ortho')
        # ax.set_xticks([])
        # ax.set_yticks([])
        # ax.set_zticks([])
        # X, Y = torch.meshgrid(torch.as_tensor(xa), torch.as_tensor(xb))
        # # X += 1e-7
        # # Y += 1e-7
        # Z = calc_a(X + 1e-7, params[1]) * calc_b(Y + 1e-7, params[1])

        # ax.plot_surface(X, Y, Z.numpy(), rstride=1, cstride=1, cmap='rainbow')
        # ax.view_init(elev=90., azim=-90.)
        if 'JPY_PARENT_PID' in os.environ:
            plt.show()
        else:
            plt.savefig(f'tmp_{cls_id}.png')

    def judge(self, spec_d, unc_d, params_spec, params_unc, weight_spec, weight_unc):
        # params[:, 0, 0], params[:, 0, 1], params[:, 0, 2], params[:, 0, 3] = params[:, 0]
        # params[:, 1, 0], params[:, 1, 1], params[:, 1, 2], params[:, 1, 3] = params[1]

        spec1 = gamma(params_spec[:, 0] + params_spec[:, 1]) / gamma(params_spec[:, 0]) / gamma(
            params_spec[:, 1]) * spec_d**(params_spec[:, 0] - 1) * (1 - spec_d)**(params_spec[:, 1] - 1)
        spec2 = gamma(params_spec[:, 2] + params_spec[:, 3]) / gamma(params_spec[:, 2]) / gamma(
            params_spec[:, 3]) * spec_d**(params_spec[:, 2] - 1) * (1 - spec_d)**(params_spec[:, 3] - 1)
        unc1 = (params_unc[:, 1]**params_unc[:, 0]) / gamma(params_unc[:, 0]) * unc_d.pow(params_unc[:, 0] - 1) * e**(-params_unc[:, 1] * unc_d)
        unc2 = (params_unc[:, 3]**params_unc[:, 2]) / gamma(params_unc[:, 2]) * unc_d.pow(params_unc[:, 2] - 1) * e**(-params_unc[:, 3] * unc_d)

        selector_sdouble = torch.bitwise_and(weight_spec * spec1 <= (1 - weight_spec) * spec2, weight_unc * unc1 <= (1 - weight_unc) * unc2)
        selector_double = weight_spec * spec1 * weight_unc * unc1 <= (1 - weight_spec) * spec2 * (1 - weight_unc) * unc2
        selector_spec = weight_spec * spec1 <= (1 - weight_spec) * spec2
        selector_unc = weight_unc * unc1 <= (1 - weight_unc) * unc2

        # raise Exception
        return selector_sdouble, selector_double, selector_spec, selector_unc

    def unc_label_bins(self):
        correct_ds = []
        incorrect_ds = []
        for ply_name in tqdm(TRAIN_IDS, desc='uncxlabel'):
            gt_data = PlyData.read(f'/home/aidrive/tb5zhh/3d_scene_understand/data/full/train/{ply_name}.ply')
            gt_labels = np.asarray(gt_data['vertex']['label'])
            unc_obj = torch.load(f'{UNC_DATA_PATH}/{self.num_point}/{ply_name}_unc.obj')
            unc_mapping = torch.load(f'{UNC_DATA_PATH}/mappings/{ply_name}_mapping.obj')['inverse']
            unc_labels = torch.load(f"{UNC_DATA_PATH}/{self.num_point}/{ply_name}_predicted.obj")[unc_mapping]
            unc_confidence = unc_obj[unc_mapping][torch.as_tensor(list(range(len(unc_labels)))), unc_labels.flatten()]
            correct_d = unc_confidence[torch.as_tensor(VALID_CLASS_IDS)[unc_labels] == torch.as_tensor(gt_labels)]
            correct_ds.append(correct_d)
            incorrect_d = unc_confidence[torch.as_tensor(VALID_CLASS_IDS)[unc_labels] != torch.as_tensor(gt_labels)]
            incorrect_ds.append(incorrect_d)
        
        self.correct_ds = np.hstack(correct_ds)
        self.incorrect_ds = np.hstack(incorrect_ds)

        self.correct_ds = (self.correct_ds + 1e-10) / (self.correct_ds.max() + 1e-9)
        self.incorrect_ds = (self.incorrect_ds + 1e-10) / (self.incorrect_ds.max() + 1e-9)
        return self

    def generate(self):
        base_path = gen_base_path(self.num_point)
        spec_params, unc_params, spec_weights, unc_weights = [], [], [], []

        for cls_id in range(NUM_CLS):
            spec_params.append(self.fit_params[cls_id]['spec'])
            unc_params.append(self.fit_params[cls_id]['unc'])
            spec_weights.append(self.fit_params[cls_id]['spec_weight'])
            unc_weights.append(self.fit_params[cls_id]['unc_weight'])

        spec_params = torch.tensor(spec_params, dtype=torch.float64).cuda()
        unc_params = torch.tensor(unc_params, dtype=torch.float64).cuda()
        spec_weights = torch.tensor(spec_weights, dtype=torch.float64).cuda()
        unc_weights = torch.tensor(unc_weights, dtype=torch.float64).cuda()

        for ply_name in tqdm(TRAIN_IDS, desc='generate'):
            spec_obj = torch.load(f"{SPEC_DATA_PATH}/{ply_name}_{self.num_point}.obj")
            spec_confidence = torch.as_tensor(spec_obj['confidence'])

            unc_obj = torch.load(f'{UNC_DATA_PATH}/{self.num_point}/{ply_name}_unc.obj')
            unc_mapping = torch.load(f'{UNC_DATA_PATH}/mappings/{ply_name}_mapping.obj')['inverse']
            unc_labels = torch.load(f"{UNC_DATA_PATH}/{self.num_point}/{ply_name}_predicted.obj")[unc_mapping]
            unc_confidence = unc_obj[unc_mapping]

            unc_labels = unc_labels.cuda()
            spec_confidence = spec_confidence.to(torch.float64).cuda()
            unc_confidence = unc_confidence.to(torch.float64).cuda()

            unc_confidence = torch.gather(unc_confidence, 1, unc_labels.reshape(1, -1))
            selected_params_spec = spec_params[unc_labels]
            selected_params_unc = unc_params[unc_labels]
            selected_weights_spec = spec_weights[unc_labels]
            selected_weights_unc = unc_weights[unc_labels]

            class_ids = np.array(VALID_CLASS_IDS + (255,))
            selector_sdouble, selector_double, selector_spec, selector_unc = self.judge(
                spec_confidence,
                unc_confidence,
                selected_params_spec,
                selected_params_unc,
                selected_weights_spec,
                selected_weights_unc,
            )

            ply_data = PlyData.read(f'{base_path}/{ply_name}.ply')
            gt_labels = np.copy(ply_data['vertex']['label'])

            SAVE_PATH = f'{GENERATE_PATH}/{self.num_point}_fit_sdouble/train'
            os.makedirs(SAVE_PATH, exist_ok=True)
            new_labels = torch.where(selector_sdouble.cuda(), unc_labels, NUM_CLS).flatten()
            generated_labels = np.array(class_ids[new_labels.cpu()])
            ply_data['vertex']['label'] = np.where(gt_labels != 255, gt_labels, generated_labels)
            ply_data.write(f'{SAVE_PATH}/{ply_name}.ply')

            SAVE_PATH = f'{GENERATE_PATH}/{self.num_point}_fit_double/train'
            os.makedirs(SAVE_PATH, exist_ok=True)
            new_labels = torch.where(selector_double.cuda(), unc_labels, NUM_CLS).flatten()
            generated_labels = np.array(class_ids[new_labels.cpu()])
            ply_data['vertex']['label'] = np.where(gt_labels != 255, gt_labels, generated_labels)
            ply_data.write(f'{SAVE_PATH}/{ply_name}.ply')

            SAVE_PATH = f'{GENERATE_PATH}/{self.num_point}_fit_spec/train'
            os.makedirs(SAVE_PATH, exist_ok=True)
            new_labels = torch.where(selector_spec.cuda(), unc_labels, NUM_CLS).flatten()
            generated_labels = np.array(class_ids[new_labels.cpu()])
            ply_data['vertex']['label'] = np.where(gt_labels != 255, gt_labels, generated_labels)
            ply_data.write(f'{SAVE_PATH}/{ply_name}.ply')

            SAVE_PATH = f'{GENERATE_PATH}/{self.num_point}_fit_unc/train'
            os.makedirs(SAVE_PATH, exist_ok=True)
            new_labels = torch.where(selector_unc.cuda(), unc_labels, NUM_CLS).flatten()
            generated_labels = np.array(class_ids[new_labels.cpu()])
            ply_data['vertex']['label'] = np.where(gt_labels != 255, gt_labels, generated_labels)
            ply_data.write(f'{SAVE_PATH}/{ply_name}.ply')
        return self

    def validate_render(self, ply_name='scene0049_00', target_path='.'):
        for variant in ('double', 'sdouble', 'spec', 'unc'):
            origin_ply = PlyData.read(f"{ORIGIN_PATH}/{ply_name}.ply")
            generated_ply = PlyData.read(f"{GENERATE_PATH}/{self.num_point}_fit_{variant}/train/{ply_name}.ply")

            colors = torch.tensor(list(SCANNET_COLOR_MAP.values()) + [(255., 255., 255.)])
            t = torch.tensor(generated_ply['vertex']['label'], dtype=int)
            labels = t.where(t != 255, torch.tensor(41))

            origin_ply['vertex']['red'] = colors[labels][:, 0]
            origin_ply['vertex']['green'] = colors[labels][:, 1]
            origin_ply['vertex']['blue'] = colors[labels][:, 2]
            origin_ply.write(f"{target_path}/{ply_name}_{variant}_{self.num_point}_render.ply")
        return self

    def validate_miou(self):
        for variant in ('double', 'sdouble', 'spec', 'unc'):
            raise NotImplementedError
        return self




# %%
# if __name__ == '__main__':
# a = FitRunner(100, debug=False).validate_render()
# a = FitRunner(20, debug=False).load().fit(update=True, render=True, init_spec=[(10, 10), (2, 2)], fitted_cls=list(range(NUM_CLS)))
# a = FitRunner(50, debug=False).load().fit(update=False, render=True, init_spec=[(10, 8), (10, 10)], fitted_cls=(10, 11, 16, 17), part=True)
# a = FitRunner(50, debug=False).load().fit().generate()

# a.generate().validate_render()
# a.load().fit()

# %%
def draw(t):
    n_bins = 400
    width = 1 / n_bins
    fig = plt.figure(dpi=400)
    x = [i * 1 / n_bins for i in range(n_bins)]
    y = torch.histc(torch.as_tensor(t), bins=n_bins, min=0,max=1)
    ax = fig.add_subplot(1,1,1)
    ax.bar(x,y,width=width)
    plt.show()
a = FitRunner(200, debug=False).unc_label_bins()
draw(a.correct_ds)
draw(a.incorrect_ds)
a = FitRunner(100, debug=False).unc_label_bins()
draw(a.correct_ds)
draw(a.incorrect_ds)
a = FitRunner(50, debug=False).unc_label_bins()
draw(a.correct_ds)
draw(a.incorrect_ds)
a = FitRunner(20, debug=False).unc_label_bins()
draw(a.correct_ds)
draw(a.incorrect_ds)

# %%
