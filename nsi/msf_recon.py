"""Multi-slice fusion reconstruction of an NSI cone-beam scan.

Runs MACE with the data-fit proximal map as the forward agent and three
DRUNet denoiser agents, one per slice orientation, at consensus weights
(1/2, 1/6, 1/6, 1/6) -- the multi-slice fusion construction validated on
demo problems in mbirtorch/experiments/drunet (see
mbirtorch_plans/plans/nn_priors/multi_slice_fusion_findings.md).  The MACE
loop and the agents are inlined from that directory so this script is
self-contained; the preprocessing duplicates the opening of Lilly_recon.py.

The reconstruction is initialized at the EXISTING standard MBIR recon (the
--standard_recon h5 produced by Lilly_recon.py with matching settings), so
the standard recon is not recomputed here.  The denoiser strength
sigma_scaled is the one regularization knob: DRUNet is a Gaussian denoiser
conditioned on the noise standard deviation of images valued in [0, 1]; the
volume is mapped into that range by a fixed intensity scale (the standard
recon's 99.9th-percentile value goes to 0.9), the network runs at
sigma_scaled, and the result is mapped back.  0.075 was the best MACE value
on the demo problems; the useful range is roughly 0.02 (light) to 0.2
(heavy).  Since there is no ground truth, the reported metric is the
weighted sinogram residual rms_w(y - Ax) (data consistency), plus the saved
volumes for visual comparison.

Outputs under ./output/msf/: the fusion recon (and optionally the
three-orientation DRUNet postprocessing of the standard recon) as h5 in the
same conventions as Lilly_recon.py, plus a small npz with the metrics and
per-iteration traces.

Run from this directory in the mbirtorch conda env on a GPU node, e.g.:
    python msf_recon.py --data_path /depot/bouman/data/Lilly/Autoinjector_HighRes_Horizontal/
"""

import argparse
import os
import time

import numpy as np
import torch

import mbirtorch as mt
import mbirtorch.preprocess as mtp


# --------------------------------------------------------------- MACE loop --

def mace(agents, x0, mu, rho=0.5, num_iterations=30, callback=None):
    """Weighted Mann iteration for consensus equilibrium over agents that
    map a volume tensor to a volume tensor on one fixed device.  At a fixed
    point every agent output equals the consensus, so the per-iteration
    spread max_i ||X_i - x_bar|| / ||x_bar|| is the convergence trace."""

    def norm(x):
        return float(torch.sqrt(torch.sum(x * x)))

    W = [x0.clone() for _ in agents]
    info = {'consensus_spread': [], 'consensus_change': []}
    previous_x_bar = None
    with torch.no_grad():
        for iteration in range(num_iterations):
            X = [agent(w) for agent, w in zip(agents, W)]
            x_bar = sum(m * x for m, x in zip(mu, X))
            z = sum(m * (2.0 * x - w) for m, x, w in zip(mu, X, W))
            W = [w + 2.0 * rho * (z - x) for w, x in zip(W, X)]

            x_bar_norm = norm(x_bar)
            info['consensus_spread'].append(
                max(norm(x - x_bar) for x in X) / x_bar_norm)
            info['consensus_change'].append(
                norm(x_bar - previous_x_bar) / x_bar_norm
                if previous_x_bar is not None else 0.0)
            previous_x_bar = x_bar
            if callback is not None:
                callback(iteration, x_bar, info)
    return x_bar, info


# ------------------------------------------------------------------ agents --

class ForwardProxAgent:
    """Proximal map of the data-fit term via TomographyModel.prox_map, with
    warm starts from its own previous output and the cumulative iteration
    count passed as first_iteration, so the model's partition sequence walks
    coarse to fine across MACE iterations."""

    def __init__(self, model, sinogram, weights=None, sigma_prox=None,
                 inner_iterations=3, init_recon=None, logfile_path=None):
        self.model = model
        self.sinogram = sinogram
        self.weights = weights
        self.sigma_prox = sigma_prox
        self.inner_iterations = inner_iterations
        self.logfile_path = logfile_path
        self._previous_output = init_recon
        self._iterations_done = 0

    def __call__(self, v):
        first = self._iterations_done
        output, _ = self.model.prox_map(
            v, self.sinogram, sigma_prox=self.sigma_prox,
            weights=self.weights, init_recon=self._previous_output,
            do_initialization=(first == 0),
            max_iterations=first + self.inner_iterations,
            first_iteration=first,
            stop_threshold_change_pct=0.0,
            logfile_path=self.logfile_path,
            print_logs=False, output_sharded=True)
        self._previous_output = output
        self._iterations_done = first + self.inner_iterations
        return output


def load_drunet(device):
    """The pretrained grayscale DRUNet (weights from the local cache, or
    downloaded on first use)."""
    from deepinv.models import DRUNet
    net = DRUNet(in_channels=1, out_channels=1, pretrained='download',
                 device=device)
    net.eval()
    return net


class DRUNetAgent:
    """DRUNet applied to the volume's slices along ``slice_axis``, with a
    fixed intensity scale mapping recon values into the network's [0, 1]
    range.  The scale and the region-of-reconstruction mask are applied to
    the VOLUME, so all three orientation agents share them."""

    def __init__(self, net, sigma_noise, intensity_scale, ror_mask=None,
                 slice_batch=8, slice_axis=2):
        self.net = net
        self.sigma_noise = float(sigma_noise)
        self.intensity_scale = float(intensity_scale)
        self.ror_mask = ror_mask
        self.slice_batch = slice_batch
        self.slice_axis = slice_axis

    def __call__(self, v):
        import torch.nn.functional as functional
        x = torch.moveaxis(self.intensity_scale * v,
                           self.slice_axis, 0).unsqueeze(1)
        height, width = x.shape[-2:]
        pad_rows = (-height) % 8
        pad_cols = (-width) % 8
        if pad_rows or pad_cols:
            x = functional.pad(x, (0, pad_cols, 0, pad_rows), mode='reflect')
        sigma_scaled = self.intensity_scale * self.sigma_noise
        with torch.no_grad():
            y = torch.cat([self.net(batch, sigma_scaled)
                           for batch in x.split(self.slice_batch)])
        y = torch.moveaxis(y[..., :height, :width].squeeze(1),
                           0, self.slice_axis)
        y = y / self.intensity_scale
        if self.ror_mask is not None:
            y = v + self.ror_mask * (y - v)
        return y


# -------------------------------------------------------------------- main --

if __name__ == "__main__":
    sharpness = 1.0
    verbose = 1

    parser = argparse.ArgumentParser(
        description="Multi-slice fusion recon of an NSI scan")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to existing data directory.")
    parser.add_argument("--downsampling", type=int, default=3,
                        help="Subsampling factor for detector rows and channels.")
    parser.add_argument("--subsample_view_factor", type=int, default=5,
                        help="Subsampling factor for projection views.")
    parser.add_argument("--sino_cropping", type=int, default=1,
                        help="Flag for applying sinogram cropping")
    parser.add_argument("--standard_recon", type=str,
                        default="./output/lilly/recon_Autoinjector_HighRes_"
                                "Horizontal_nummetal_0_voxel_pitch_102.94um.h5",
                        help="Existing standard MBIR recon with matching "
                             "settings (initialization and intensity scale).")
    parser.add_argument("--sigma_scaled", type=float, default=0.075,
                        help="Denoiser strength in the network scale; see "
                             "the header comment.")
    parser.add_argument("--iterations", type=int, default=30,
                        help="MACE outer iterations.")
    parser.add_argument("--rho", type=float, default=0.5,
                        help="Mann averaging parameter in (0, 1).")
    parser.add_argument("--inner_prox", type=int, default=3,
                        help="VCD iterations per data-prox call.")
    parser.add_argument("--save_postproc", type=int, default=1,
                        help="Also save the three-orientation DRUNet "
                             "postprocessing of the standard recon.")
    args = parser.parse_args()

    # Job preflight: fail fast off-GPU, and name the library under test.
    assert torch.cuda.is_available(), 'NOT ON GPU: torch.cuda unavailable'
    print('library under test:', mt.__file__, flush=True)

    output_path = './output/msf/'
    os.makedirs(output_path, exist_ok=True)
    logfile_dir = './logs/'
    os.makedirs(logfile_dir, exist_ok=True)

    if args.data_path is not None and not os.path.isdir(args.data_path):
        raise FileNotFoundError(
            f"--data_path does not exist or is not a directory: {args.data_path}")
    dataset_dir = args.data_path
    dataset_tag = os.path.basename(dataset_dir.rstrip("/"))
    downsample_rate = [args.downsampling, args.downsampling]

    # === NSI preprocessing, duplicated from the opening of Lilly_recon.py ===
    if verbose > 0:
        print("\n************** NSI dataset preprocessing **************")
    cropping = bool(args.sino_cropping)
    sino, ct_model = mtp.nsi.get_sino_and_model(
        dataset_dir, downsample_factor=downsample_rate,
        subsample_view_factor=args.subsample_view_factor, auto_crop=cropping)
    sino = np.maximum(sino, 0.0)

    if verbose > 0:
        print("\n***************** Set up MBIRTORCH model ****************")
    ct_model.set_params(sharpness=sharpness, verbose=verbose,
                        positivity_flag=True)
    weights_trans = mt.gen_weights(sino, weight_type='transmission_root')
    # One device: the MACE loop exchanges plain tensors between the agents.
    ct_model.configure_devices(devices=['cuda:0'])
    device = 'cuda:0'

    # === The standard recon: initialization and intensity scale ===
    recon_shape = tuple(ct_model.get_params('recon_shape'))
    standard_np, _ = mt.import_recon_hdf5(args.standard_recon)
    if tuple(standard_np.shape) != recon_shape:
        raise ValueError(
            f"The standard recon {args.standard_recon} has shape "
            f"{standard_np.shape} but the model expects {recon_shape}; "
            f"regenerate it with matching settings (Lilly_recon.py).")
    standard = torch.as_tensor(np.ascontiguousarray(standard_np),
                               dtype=torch.float32, device=device)
    print(f"Standard recon loaded: shape {recon_shape}", flush=True)

    # The fixed intensity scale: a strided subsample keeps the quantile
    # under torch.quantile's element limit.
    subsample = standard[::4, ::4, ::4].flatten()
    robust_max = float(torch.quantile(subsample, 0.999))
    intensity_scale = 0.9 / robust_max

    if ct_model.get_params('use_ror_mask'):
        mask = mt.get_2d_ror_mask(recon_shape)
        ror_mask = torch.as_tensor(mask.astype(np.float32),
                                   device=device).unsqueeze(-1)
    else:
        ror_mask = None

    sinogram_t = torch.as_tensor(sino, device=device)
    weights_t = torch.as_tensor(np.asarray(weights_trans), device=device)

    def data_consistency(volume):
        """Weighted rms sinogram residual, the phantom-free data metric."""
        residual = sinogram_t - torch.as_tensor(
            ct_model.forward_project(volume), device=device)
        return float(torch.sqrt(torch.sum(weights_t * residual ** 2)
                                / torch.sum(weights_t)))

    net = load_drunet(device)
    print(f"sigma_scaled {args.sigma_scaled:g} "
          f"({args.sigma_scaled / intensity_scale:.5f} in recon units), "
          f"intensity scale {intensity_scale:.2f} "
          f"(robust max {robust_max:.5f})", flush=True)

    def make_denoiser(axis):
        return DRUNetAgent(net, args.sigma_scaled / intensity_scale,
                           intensity_scale, ror_mask=ror_mask,
                           slice_axis=axis)

    delta_voxel_mm = (ct_model.get_params('delta_voxel')
                      * ct_model.get_params('alu_value'))
    delta_voxel_um = delta_voxel_mm * 1000
    file_tag = (f"{dataset_tag}_sigma_{args.sigma_scaled:g}"
                f"_voxel_pitch_{delta_voxel_um:.2f}um")

    # === Optional: the postprocessing comparison volume ===
    if args.save_postproc:
        time_start = time.time()
        with torch.no_grad():
            postproc = sum(make_denoiser(axis)(standard)
                           for axis in (0, 1, 2)) / 3.0
        print(f"Postprocessing (3-orientation average) took "
              f"{time.time() - time_start:.1f} s", flush=True)

    # === MACE multi-slice fusion ===
    print("\n*************** Multi-slice fusion (MACE) ***************",
          flush=True)
    time_start = time.time()
    np.random.seed(0)
    forward = ForwardProxAgent(
        ct_model, sino, weights=weights_trans, sigma_prox=None,
        inner_iterations=args.inner_prox, init_recon=standard.clone(),
        logfile_path=os.path.join(logfile_dir, f"msf_{dataset_tag}.log"))
    agents = [forward] + [make_denoiser(axis) for axis in (0, 1, 2)]
    mu = [0.5, 1 / 6, 1 / 6, 1 / 6]

    def progress(iteration, x_bar, info):
        print(f"  iteration {iteration + 1:3d}/{args.iterations}: "
              f"consensus spread {info['consensus_spread'][-1]:.2e}, "
              f"change {info['consensus_change'][-1]:.2e}, "
              f"elapsed {time.time() - time_start:.0f} s", flush=True)

    fusion, info = mace(agents, standard.clone(), mu=mu, rho=args.rho,
                        num_iterations=args.iterations, callback=progress)
    print(f"MACE took {time.time() - time_start:.1f} s "
          f"(sigma_prox auto value "
          f"{float(ct_model.get_params('sigma_prox')):.5f})", flush=True)

    # === Metrics: data consistency (no ground truth exists) ===
    metrics = {'standard': data_consistency(standard),
               'fusion': data_consistency(fusion)}
    if args.save_postproc:
        metrics['postproc'] = data_consistency(postproc)
    print("Weighted sinogram residual rms_w(y - Ax):")
    for name, value in metrics.items():
        print(f"  {name:10s} {value:.5f}")

    # === Save ===
    fusion_path = os.path.join(output_path, f"msf_recon_{file_tag}.h5")
    mt.export_recon_hdf5(fusion_path, fusion, recon_dict=None,
                         remove_flash=True)
    print(f"Fusion recon saved to {os.path.abspath(fusion_path)}", flush=True)
    if args.save_postproc:
        postproc_path = os.path.join(output_path,
                                     f"postproc_recon_{file_tag}.h5")
        mt.export_recon_hdf5(postproc_path, postproc, recon_dict=None,
                             remove_flash=True)
        print(f"Postprocessing recon saved to "
              f"{os.path.abspath(postproc_path)}", flush=True)

    trace_path = os.path.join(output_path, f"msf_traces_{file_tag}.npz")
    np.savez(trace_path,
             consensus_spread=np.asarray(info['consensus_spread']),
             consensus_change=np.asarray(info['consensus_change']),
             sigma_scaled=args.sigma_scaled, rho=args.rho,
             iterations=args.iterations, inner_prox=args.inner_prox,
             intensity_scale=intensity_scale,
             sigma_prox=float(ct_model.get_params('sigma_prox')),
             **{f'rms_w_{name}': value for name, value in metrics.items()})
    print(f"Traces and metrics saved to {os.path.abspath(trace_path)}",
          flush=True)
