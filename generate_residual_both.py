# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Generate random images using the techniques described in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

import os
import re
import click
import tqdm
import pickle
import numpy as np
import torch
import PIL.Image
import dnnlib
from torch_utils import distributed as dist
from training.pos_embedding import Pos_Embedding


# ----------------------------------------------------------------------------
# Proposed EDM sampler (Algorithm 2).

def edm_sampler(
        net, latents, imagecondition, class_labels=None, randn_like=torch.randn_like,
        num_steps=18, sigma_min=0.002, sigma_max=80, rho=7,
        S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    # img_channel = latents.shape[1]
    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Time step discretization.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    t_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho
    t_steps = torch.cat([net.round_sigma(t_steps), torch.zeros_like(t_steps[:1])])  # t_N = 0

    # Main sampling loop.
    x_next = latents.to(torch.float64) * t_steps[0]

    with torch.no_grad():
        for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
            x_cur = x_next

            # Increase noise temporarily.
            gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= t_cur <= S_max else 0
            t_hat = net.round_sigma(t_cur + gamma * t_cur)
            x_hat = x_cur + (t_hat ** 2 - t_cur ** 2).sqrt() * S_noise * randn_like(x_cur)

            #dist.print0("Denoise line 80")
            denoised = net(x=x_hat, sigma=t_hat, imagecondition=imagecondition,
                           class_labels=class_labels).to(torch.float64)

            d_cur = (x_hat - denoised) / t_hat
            x_next = x_hat + (t_next - t_hat) * d_cur

            # Apply 2nd order correction.
            if i < num_steps - 1:
                #dist.print0(f"Denoise line 93 : {i} step")
                denoised = net(x_next, t_next, imagecondition, class_labels).to(torch.float64)
                # denoised = denoised[:, :img_channel]
                d_prime = (x_next - denoised) / t_next
                x_next = x_hat + (t_next - t_hat) * (0.5 * d_cur + 0.5 * d_prime)

    return x_next


# ----------------------------------------------------------------------------
# Generalized ablation sampler, representing the superset of all sampling
# methods discussed in the paper.

def ablation_sampler(
        net, latents, class_labels=None, randn_like=torch.randn_like,
        num_steps=18, sigma_min=None, sigma_max=None, rho=7,
        solver='heun', discretization='edm', schedule='linear', scaling='none',
        epsilon_s=1e-3, C_1=0.001, C_2=0.008, M=1000, alpha=1,
        S_churn=0, S_min=0, S_max=float('inf'), S_noise=1,
):
    assert solver in ['euler', 'heun']
    assert discretization in ['vp', 've', 'iddpm', 'edm']
    assert schedule in ['vp', 've', 'linear']
    assert scaling in ['vp', 'none']

    # Helper functions for VP & VE noise level schedules.
    vp_sigma = lambda beta_d, beta_min: lambda t: (np.e ** (0.5 * beta_d * (t ** 2) + beta_min * t) - 1) ** 0.5
    vp_sigma_deriv = lambda beta_d, beta_min: lambda t: 0.5 * (beta_min + beta_d * t) * (sigma(t) + 1 / sigma(t))
    vp_sigma_inv = lambda beta_d, beta_min: lambda sigma: ((beta_min ** 2 + 2 * beta_d * (
                sigma ** 2 + 1).log()).sqrt() - beta_min) / beta_d
    ve_sigma = lambda t: t.sqrt()
    ve_sigma_deriv = lambda t: 0.5 / t.sqrt()
    ve_sigma_inv = lambda sigma: sigma ** 2

    # Select default noise level range based on the specified time step discretization.
    if sigma_min is None:
        vp_def = vp_sigma(beta_d=19.1, beta_min=0.1)(t=epsilon_s)
        sigma_min = {'vp': vp_def, 've': 0.02, 'iddpm': 0.002, 'edm': 0.002}[discretization]
    if sigma_max is None:
        vp_def = vp_sigma(beta_d=19.1, beta_min=0.1)(t=1)
        sigma_max = {'vp': vp_def, 've': 100, 'iddpm': 81, 'edm': 80}[discretization]

    # Adjust noise levels based on what's supported by the network.
    sigma_min = max(sigma_min, net.sigma_min)
    sigma_max = min(sigma_max, net.sigma_max)

    # Compute corresponding betas for VP.
    vp_beta_d = 2 * (np.log(sigma_min ** 2 + 1) / epsilon_s - np.log(sigma_max ** 2 + 1)) / (epsilon_s - 1)
    vp_beta_min = np.log(sigma_max ** 2 + 1) - 0.5 * vp_beta_d

    # Define time steps in terms of noise level.
    step_indices = torch.arange(num_steps, dtype=torch.float64, device=latents.device)
    if discretization == 'vp':
        orig_t_steps = 1 + step_indices / (num_steps - 1) * (epsilon_s - 1)
        sigma_steps = vp_sigma(vp_beta_d, vp_beta_min)(orig_t_steps)
    elif discretization == 've':
        orig_t_steps = (sigma_max ** 2) * ((sigma_min ** 2 / sigma_max ** 2) ** (step_indices / (num_steps - 1)))
        sigma_steps = ve_sigma(orig_t_steps)
    elif discretization == 'iddpm':
        u = torch.zeros(M + 1, dtype=torch.float64, device=latents.device)
        alpha_bar = lambda j: (0.5 * np.pi * j / M / (C_2 + 1)).sin() ** 2
        for j in torch.arange(M, 0, -1, device=latents.device):  # M, ..., 1
            u[j - 1] = ((u[j] ** 2 + 1) / (alpha_bar(j - 1) / alpha_bar(j)).clip(min=C_1) - 1).sqrt()
        u_filtered = u[torch.logical_and(u >= sigma_min, u <= sigma_max)]
        sigma_steps = u_filtered[((len(u_filtered) - 1) / (num_steps - 1) * step_indices).round().to(torch.int64)]
    else:
        assert discretization == 'edm'
        sigma_steps = (sigma_max ** (1 / rho) + step_indices / (num_steps - 1) * (
                    sigma_min ** (1 / rho) - sigma_max ** (1 / rho))) ** rho

    # Define noise level schedule.
    if schedule == 'vp':
        sigma = vp_sigma(vp_beta_d, vp_beta_min)
        sigma_deriv = vp_sigma_deriv(vp_beta_d, vp_beta_min)
        sigma_inv = vp_sigma_inv(vp_beta_d, vp_beta_min)
    elif schedule == 've':
        sigma = ve_sigma
        sigma_deriv = ve_sigma_deriv
        sigma_inv = ve_sigma_inv
    else:
        assert schedule == 'linear'
        sigma = lambda t: t
        sigma_deriv = lambda t: 1
        sigma_inv = lambda sigma: sigma

    # Define scaling schedule.
    if scaling == 'vp':
        s = lambda t: 1 / (1 + sigma(t) ** 2).sqrt()
        s_deriv = lambda t: -sigma(t) * sigma_deriv(t) * (s(t) ** 3)
    else:
        assert scaling == 'none'
        s = lambda t: 1
        s_deriv = lambda t: 0

    # Compute final time steps based on the corresponding noise levels.
    t_steps = sigma_inv(net.round_sigma(sigma_steps))
    t_steps = torch.cat([t_steps, torch.zeros_like(t_steps[:1])])  # t_N = 0

    # Main sampling loop.
    t_next = t_steps[0]
    x_next = latents.to(torch.float64) * (sigma(t_next) * s(t_next))
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):  # 0, ..., N-1
        x_cur = x_next

        # Increase noise temporarily.
        gamma = min(S_churn / num_steps, np.sqrt(2) - 1) if S_min <= sigma(t_cur) <= S_max else 0
        t_hat = sigma_inv(net.round_sigma(sigma(t_cur) + gamma * sigma(t_cur)))
        x_hat = s(t_hat) / s(t_cur) * x_cur + (sigma(t_hat) ** 2 - sigma(t_cur) ** 2).clip(min=0).sqrt() * s(
            t_hat) * S_noise * randn_like(x_cur)

        # Euler step.
        h = t_next - t_hat
        denoised = net(x_hat / s(t_hat), sigma(t_hat), class_labels).to(torch.float64)
        d_cur = (sigma_deriv(t_hat) / sigma(t_hat) + s_deriv(t_hat) / s(t_hat)) * x_hat - sigma_deriv(t_hat) * s(
            t_hat) / sigma(t_hat) * denoised
        x_prime = x_hat + alpha * h * d_cur
        t_prime = t_hat + alpha * h

        # Apply 2nd order correction.
        if solver == 'euler' or i == num_steps - 1:
            x_next = x_hat + h * d_cur
        else:
            assert solver == 'heun'
            denoised = net(x_prime / s(t_prime), sigma(t_prime), class_labels).to(torch.float64)
            d_prime = (sigma_deriv(t_prime) / sigma(t_prime) + s_deriv(t_prime) / s(t_prime)) * x_prime - sigma_deriv(
                t_prime) * s(t_prime) / sigma(t_prime) * denoised
            x_next = x_hat + h * ((1 - 1 / (2 * alpha)) * d_cur + 1 / (2 * alpha) * d_prime)

    return x_next


# ----------------------------------------------------------------------------
# Wrapper for torch.Generator that allows specifying a different random seed
# for each sample in a minibatch.

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])


# ----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2)) + 1))
        else:
            ranges.append(int(p))
    return ranges


# ----------------------------------------------------------------------------

def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


# ----------------------------------------------------------------------------
rootpath = "/external/syhome"
#rootpath = "H:"
itersname = "008192"
network_dir = f"{rootpath}/Random/time_diffusion/residualmodel_both/trained_model/results_residual/both_noscale_network-snapshot-{itersname}.pkl"
data_dir =f"{rootpath}/2_Datasets/11_TimeGraph/brain/cortical_thickness.txt"
out_dir = f"{rootpath}/Random/time_diffusion/residualmodel_both/{itersname}/"
if not os.path.exists(out_dir):
    os.makedirs(out_dir)


nsteps=1000

@click.command()
@click.option('--network', 'network_pkl', help='Network pickle filename', metavar='PATH|URL', type=str,
              default=network_dir, required=True)
@click.option('--resolution', help='Sample resolution', metavar='INT', type=int, default=512)
@click.option('--embed_fq', help='Positional embedding frequency', metavar='INT', type=int, default=0)
@click.option('--mask_pos', help='Mask out pos channels', metavar='BOOL', type=bool, default=False, show_default=True)
@click.option('--on_latents', help='Generate with latent vae', metavar='BOOL', type=bool, default=False,
              show_default=True)
@click.option('--outdir', help='Where to save the output images', metavar='DIR', type=str, default="./results/",
              required=True)
# patch options
@click.option('--x_start', help='Sample resolution', metavar='INT', type=int, default=0)
@click.option('--y_start', help='Sample resolution', metavar='INT', type=int, default=0)
@click.option('--image_size', help='Sample resolution', metavar='INT', type=int, default=None)
@click.option('--seeds', help='Random seeds (e.g. 1,2,5-10)', metavar='LIST', type=parse_int_list, default='1',
              show_default=True)
@click.option('--subdirs', help='Create subdirectory for every 1000 seeds', is_flag=True)
@click.option('--class', 'class_idx', help='Class label  [default: random]', metavar='INT', type=click.IntRange(min=0),
              default=None)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT', type=click.IntRange(min=1),
              default=1, show_default=True)
@click.option('--steps', 'num_steps', help='Number of sampling steps', metavar='INT', type=click.IntRange(min=1),
              default=nsteps, show_default=True)
@click.option('--sigma_min', help='Lowest noise level  [default: varies]', metavar='FLOAT',
              type=click.FloatRange(min=0, min_open=True))
@click.option('--sigma_max', help='Highest noise level  [default: varies]', metavar='FLOAT',
              type=click.FloatRange(min=0, min_open=True))
@click.option('--rho', help='Time step exponent', metavar='FLOAT', type=click.FloatRange(min=0, min_open=True),
              default=7, show_default=True)
@click.option('--S_churn', 'S_churn', help='Stochasticity strength', metavar='FLOAT', type=click.FloatRange(min=0),
              default=0, show_default=True)
@click.option('--S_min', 'S_min', help='Stoch. min noise level', metavar='FLOAT', type=click.FloatRange(min=0),
              default=0, show_default=True)
@click.option('--S_max', 'S_max', help='Stoch. max noise level', metavar='FLOAT', type=click.FloatRange(min=0),
              default='inf', show_default=True)
@click.option('--S_noise', 'S_noise', help='Stoch. noise inflation', metavar='FLOAT', type=float, default=1,
              show_default=True)
@click.option('--solver', help='Ablate ODE solver', metavar='euler|heun', type=click.Choice(['euler', 'heun']))
@click.option('--disc', 'discretization', help='Ablate time step discretization {t_i}', metavar='vp|ve|iddpm|edm',
              type=click.Choice(['vp', 've', 'iddpm', 'edm']))
@click.option('--schedule', help='Ablate noise schedule sigma(t)', metavar='vp|ve|linear',
              type=click.Choice(['vp', 've', 'linear']))
@click.option('--scaling', help='Ablate signal scaling s(t)', metavar='vp|none', type=click.Choice(['vp', 'none']))
def main(network_pkl, resolution, on_latents, embed_fq, mask_pos, x_start, y_start, image_size, outdir, subdirs, seeds,
         class_idx, max_batch_size, device=torch.device('cuda'), **sampler_kwargs):

    # Load network.
    dist.print0(f'Loading network from "{network_pkl}"...')
    with dnnlib.util.open_url(network_pkl, verbose=(dist.get_rank() == 0)) as f:
        net = pickle.load(f)['ema'].to(device)

    c = dnnlib.EasyDict()

    c.dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.CustomDataset_both_time_test', path=data_dir,
                                       use_normalizer=True, cache=True)
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, num_workers=1, prefetch_factor=2)

    dataset_obj = dnnlib.util.construct_class_by_name(**c.dataset_kwargs)  # subclass of training.dataset.Dataset
    dataset_iterator = iter(torch.utils.data.DataLoader(dataset=dataset_obj, batch_size=1))

    for batch_seeds in range(len(dataset_iterator)):

        dist.print0(f"{batch_seeds} data")
        batch_size = 1

        # batch_mul = batch_mul_dict[patch_size]
        source, target, timegaps, sicknames, ages, sexs, pids, src_times,trg_times, digs= [], [], [], [], [], [],[],[],[],[]
        for _ in range(batch_size):  # batch size per gpu
            source_, target_, timegap_, sickname_, age_, sex_,pids_,src_times_,trg_times_,digs_ = next(dataset_iterator)
            source.append(source_)
            target.append(target_)
            timegaps.append(timegap_)
            sicknames.append(sickname_)
            ages.append(age_)
            sexs.append(sex_)
            pids.append(pids_)
            src_times.append(src_times_)
            trg_times.append(trg_times_)
            digs.append(digs_)


        del  source_, target_, timegap_, sickname_, age_, sex_,pids_,src_times_,trg_times_,digs_


        source = torch.cat(source, dim=0)
        target = torch.cat(target, dim=0)
        timegaps = torch.cat(timegaps, dim=0)
        sicknames = torch.cat(sicknames, dim=0)
        ages = torch.cat(ages, dim=0)
        sexs = torch.cat(sexs, dim=0)
        source_in = torch.cat([source, timegaps, sicknames, ages, sexs], dim=1)



        pid = np.array(pids[0][0])
        srcdate = np.array(src_times[0][0])
        trgdate = np.array(trg_times[0][0])
        dig= np.array(digs[0][0])

        if dig == 0 :
            sicktype = 'CN'
        elif dig == -1.0:
            sicktype ='AD'
        else:
            sicktype = 'MCI'

        source = source.to(device)
        source_in = source_in.to(device)
        target = target.to(device)


        source = source.to(torch.float32)
        source_in = source_in.to(torch.float32)
        target = target.to(torch.float32)

        repeats=1
        ave_error=0.
        for kid in range (repeats):

            latents = torch.randn([batch_size, 1, source_in.shape[-1]], device=device)

            # Generate images.
            sampler_kwargs = {key: value for key, value in sampler_kwargs.items() if value is not None}
            backprojects = edm_sampler(net, latents, source_in, class_labels=None,
                                       randn_like=torch.randn_like, **sampler_kwargs)
            backprojects = torch.squeeze(backprojects)

            # Save images.
            #
            sourcea = torch.squeeze(source)
            sourcea = sourcea.cpu().numpy()

            # Save images.
            images_np = backprojects.cpu().numpy()
            estimated=images_np [2:2+68]+sourcea[2:2+68]

            #
            targeta = torch.squeeze(target)
            targeta = targeta.cpu().numpy()
            targeta=targeta [2:2+68]

            ave_error = ave_error + np.sum(abs(targeta - estimated))

            srca=sourcea[2:2+68]


            diffs=targeta - srca
            diffe=targeta - estimated
            # Convert each element to string and join with commas
            targeta = ",".join(map(str, targeta))
            estimated = ",".join(map(str, estimated))
            srca = ",".join(map(str, srca))
            diff_ts = ",".join(map(str, diffs))
            diff_te = ",".join(map(str, diffe))
            f_full = open(f"{out_dir}Full_{itersname}_{nsteps}steps.txt", "a")
            f_full.write(f"{pid},{srcdate},{trgdate},{sicktype},{ave_error / 68.0 / float(repeats)}, {srca}, {targeta},{estimated},{diff_ts},{diff_te} \n")
            f_full.close()
    # Done.
    dist.print0('Done.')


# ----------------------------------------------------------------------------

if __name__ == "__main__":
    main()

# ----------------------------------------------------------------------------
