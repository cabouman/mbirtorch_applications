"""
Command-line driver for 4D MACE CT reconstruction.

Typical usage is through the shell script:
    bash test_script_4d.sh

The script can also be run directly:
    python Lilly_recon_4d.py --data_path /path/to/nsi/dataset

Every parameter is a command-line flag.  The defaults are the validated values for the
4DCT phantom dataset.

This is the mbirtorch version of the mbirjax driver of the same name.  The flags of that
driver are unchanged, so a command line written for it runs here, and the calls differ
only in the package.  Two groups of flags are new, because the parameters behind them are
new to the mbirtorch class: the strength of the three denoisers, and the device pool.
Their defaults are the class defaults, so a run that sets none of them behaves as the
class does on its own.
"""

import argparse
import os
import time
import warnings

import numpy as np

import mbirtorch as mt
import mbirtorch.preprocess as mtp


def parse_args():
    """Parse the command-line arguments.  The defaults are the validated values for the 4DCT phantom."""
    parser = argparse.ArgumentParser(
        description="4D MACE CT Reconstruction (mbirtorch)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    g = parser.add_argument_group("data and output")
    g.add_argument("--data_path", type=str, required=True,
                   help="NSI dataset directory, or a .tgz path/URL to download and extract.")
    g.add_argument("--download_dir", type=str, default="./data",
                   help="Extraction directory used when --data_path is a .tgz.")
    g.add_argument("--output_path", type=str, default="./output/lilly",
                   help="Directory for the recon (.npy), GIF and init cache. Logs go to ./logs/. "
                        "Files are named recon_4d_<dataset>_voxel_pitch_<um>um, as in nsi/Lilly_recon.py.")

    g = parser.add_argument_group("preprocessing")
    g.add_argument("--downsampling", type=int, default=1,
                   help="Subsampling factor for detector rows and channels.")
    g.add_argument("--sharpness", type=float, default=1.0,
                   help="Sharpness of the per-frame models, which scales sigma_x and "
                        "sigma_prox: one unit doubles both. It therefore sets the prior of "
                        "the per-frame reconstruction that initialises the run, and, while "
                        "--sigma_prox is left automatic, the strength of the data-fit agent "
                        "for the whole run. It does NOT reach the three denoisers; those "
                        "take --denoiser_sharpness. snr_db, set on ct_model, fixes sigma_y.")

    g = parser.add_argument_group("time frames")
    g.add_argument("--frames_per_rotation", type=int, default=6,
                   help="Time frames per full 360-degree rotation.")
    g.add_argument("--frame_overlap_factor", type=float, default=2.0,
                   help="Number of frames that share any given view. Each frame "
                        "spans frame_overlap_factor * (360 / frames_per_rotation) degrees.")
    g.add_argument("--num_frames", type=int, default=None,
                   help="Reconstruct only the first N time frames. Omit (default) to use all frames. "
                        "If N exceeds the total frame count, all frames are used.")

    g = parser.add_argument_group("MACE algorithm")
    g.add_argument("--max_mace_iterations", type=int, default=10,
                   help="Maximum number of outer MACE iterations.")
    g.add_argument("--stop_threshold_change_pct", type=float, default=0.2,
                   help="Stop when the consensus image changes by less than this percent between "
                        "iterations. Set to 0 to always run max_mace_iterations.")
    g.add_argument("--mace_prior_weight", type=float, default=0.5, help="Total weight of the three prior agents.")
    g.add_argument("--rho_mann", type=float, default=0.5, help="Mann iteration step size (ADMM rho).")
    g.add_argument("--prox_num_iterations", type=int, default=3, help="Max prox_map iterations per MACE step.")
    g.add_argument("--prox_stop_threshold", type=float, default=0.02, help="Prox_map convergence threshold.")
    g.add_argument("--sigma_prox", type=float, default=None, help="Proximal map sigma. Omit for automatic selection.")
    g.add_argument("--weight_type", type=str, default="transmission_root",
                   choices=["unweighted", "transmission", "transmission_root", "emission"],
                   help="Sinogram weighting, as in mt.gen_weights. transmission_root is the "
                        "validated setting for 4D transmission data. 'unweighted' passes "
                        "weights=None, which means unit weights.")
    g.add_argument("--no_dejitter", action="store_true", help="Disable the DCT-I temporal dejitter.")

    g = parser.add_argument_group("denoiser strength (new in mbirtorch)")
    g.add_argument("--denoiser_sharpness", type=float, default=0.0,
                   help="Sharpness of the three qGGMRF denoisers, which scales their automatic "
                        "sigma_x. Higher is sharper: one unit doubles sigma_x. The default of 0 "
                        "is the denoiser class default.")
    g.add_argument("--sigma_noise", type=float, default=None,
                   help="Noise level of the three denoisers. Omit to estimate it from the "
                        "initial image.")
    g.add_argument("--denoiser_sigma_x", type=float, default=None,
                   help="Prior strength of the three denoisers, shared by all of them. Omit to "
                        "estimate it from the initial image. Setting it disables that estimate, "
                        "and --denoiser_sharpness then has no effect.")
    g.add_argument("--nbr_weight_time", type=float, default=1.0,
                   help="Weight of a frame neighbour against a spatial one in the denoisers' "
                        "priors. The frame direction appears in all three hyperplane volumes and "
                        "each spatial direction in two, so the default of 1.0 gives a frame "
                        "neighbour 1.5 times the weight of a spatial one, and 2/3 weights them "
                        "equally. Larger values smooth harder along time, at the cost of motion.")

    g = parser.add_argument_group("execution")
    g.add_argument("--serial", action="store_true",
                   help="Run all tasks on one device. By default all visible GPUs are "
                        "used. Restrict them with CUDA_VISIBLE_DEVICES.")
    g.add_argument("--num_devices", type=int, default=None,
                   help="Use the first N devices instead of all of them. A GPU may not be "
                        "named twice, so N is capped by the number of visible GPUs and by "
                        "MBIRTORCH_NUM_DEVICES when that is set.")
    g.add_argument("--verbose", type=int, default=1, help="0 = silent, 1 = progress, 2 = debug.")
    g.add_argument("--gif_vmax", type=float, default=0.06,
                   help="Upper display bound for the output GIF, in units of attenuation.")
    g.add_argument("--gif_slice_axis", type=int, default=None, choices=[0, 1, 2, 3],
                   help="Write one GIF holding this axis fixed: 0=time, 1=x, 2=y, 3=z. Omit "
                        "(default) to write all three spatial planes. Fixing time steps through "
                        "the slices of a single time frame instead of playing over time.")
    g.add_argument("--gif_slice_index", type=int, default=None,
                   help="Index along --gif_slice_axis, which must be given as well. Omit for the "
                        "middle of that axis.")

    args = parser.parse_args()
    if args.gif_slice_index is not None and args.gif_slice_axis is None:
        parser.error("--gif_slice_index needs --gif_slice_axis, to say which axis it indexes.")
    if args.serial and args.num_devices is not None:
        parser.error("--serial and --num_devices both set the device pool; give one of them.")
    return args


def resolve_dataset(args):
    """Return the dataset directory.  A data_path that is not a directory is downloaded and extracted."""
    if os.path.isdir(args.data_path):
        return args.data_path
    os.makedirs(args.download_dir, exist_ok=True)
    return mt.download_and_extract(args.data_path, args.download_dir)


def set_denoiser_strength(mace_model, args):
    """Apply the denoiser strength flags, and report what the run will use.

    Setting sigma_x directly turns the automatic estimate off, which every model
    warns about.  The warning is expected here, since the flag exists to do exactly
    that, so it is replaced by a line that says so.
    """
    mace_model.set_params(sharpness=args.denoiser_sharpness,
                          sigma_noise=args.sigma_noise,
                          nbr_weight_time=args.nbr_weight_time)
    if args.denoiser_sigma_x is not None:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*auto-regularization.*")
            mace_model.set_params(sigma_x=args.denoiser_sigma_x)
        print(f"Denoiser sigma_x pinned to {args.denoiser_sigma_x}; the automatic estimate "
              f"and --denoiser_sharpness are not used.")
    else:
        print(f"Denoiser sigma_x estimated from the initial image, scaled by "
              f"sharpness {args.denoiser_sharpness}.")
    if args.nbr_weight_time != 1.0:
        print(f"Frame-neighbour weight {args.nbr_weight_time} (1.0 weights a frame neighbour "
              f"1.5 times a spatial one; 2/3 weights them equally).")


def append_run_info(log_dir, args, dataset_dir, num_frames, devices, run_time_h, out_path):
    """Append the script settings to the run_info.txt started by MACE4DModel.recon()."""
    with open(os.path.join(log_dir, "run_info.txt"), "a") as f:
        f.write("\n# Script settings (Lilly_recon_4d.py)\n")
        f.write(f"dataset              = {dataset_dir}\n")
        f.write(f"downsampling         = {args.downsampling}\n")
        f.write(f"frames_per_rotation  = {args.frames_per_rotation}\n")
        f.write(f"frame_overlap_factor = {args.frame_overlap_factor}\n")
        f.write(f"num_frames           = {num_frames}\n")
        f.write(f"weight_type          = {args.weight_type}\n")
        f.write(f"sharpness (frames)   = {args.sharpness}\n")
        f.write(f"denoiser_sharpness   = {args.denoiser_sharpness}\n")
        f.write(f"sigma_noise          = {args.sigma_noise}\n")
        f.write(f"denoiser_sigma_x     = {args.denoiser_sigma_x}\n")
        f.write(f"nbr_weight_time      = {args.nbr_weight_time}\n")
        f.write(f"devices              = {devices}\n")
        f.write(f"total wall time      = {run_time_h:.2f} h\n")
        f.write(f"recon saved to       = {out_path}\n")


def main():
    args = parse_args()
    dataset_dir = resolve_dataset(args)

    output_path = args.output_path
    os.makedirs(output_path, exist_ok=True)

    print("\n************** NSI dataset preprocessing **************")
    sino, ct_model = mtp.nsi.get_sino_and_model(
        dataset_dir,
        downsample_factor=[args.downsampling, args.downsampling],
        auto_crop=True,
    )
    ct_model.set_params(sharpness=args.sharpness, positivity_flag=True, verbose=args.verbose)

    print("\n************** Build 4D MACE model **************")
    # The frame structure follows from the model's angles alone, so it is fixed here.
    # The sinogram enters at recon() below.
    mace_model = mt.MACE4DModel(
        ct_model,
        frames_per_rotation=args.frames_per_rotation,
        frame_overlap_factor=args.frame_overlap_factor,
        num_frames=args.num_frames,
    )
    mace_model.set_params(
        mace_prior_weight=args.mace_prior_weight,
        rho_mann=args.rho_mann,
        prox_num_iterations=args.prox_num_iterations,
        prox_stop_threshold=args.prox_stop_threshold,
        sigma_prox=args.sigma_prox,
        dejitter=not args.no_dejitter,
        verbose=args.verbose,
    )
    set_denoiser_strength(mace_model, args)
    if args.serial:
        mace_model.set_device_pool(1)
    elif args.num_devices is not None:
        mace_model.set_device_pool(args.num_devices)
    devices = [str(d) for d in mace_model.devices]
    print(f"Time frames: {mace_model.num_frames} "
          f"({mace_model.view_slices[0].stop - mace_model.view_slices[0].start} views each)")
    print(f"Devices: {len(devices)}: {', '.join(devices)}")

    # The output naming follows nsi/Lilly_recon.py and uses the dataset tag plus the voxel
    # pitch.  The frame count is appended so that a partial --num_frames run never
    # overwrites a full one.  The init cache uses the same stem because its loader checks
    # only the array shape.
    dataset_tag = os.path.basename(dataset_dir.rstrip("/"))
    delta_voxel_um = ct_model.get_params('delta_voxel') * ct_model.get_params('alu_value') * 1000
    frames_tag = f"_frames_{mace_model.num_frames}"
    stem = f"recon_4d_{dataset_tag}_voxel_pitch_{delta_voxel_um:.2f}um{frames_tag}"
    init_dir = os.path.join(output_path, "init", stem)
    log_dir = os.path.join("./logs", stem)

    # transmission_root is the validated weighting for 4D data.  The model itself defaults
    # to unit weights, so the choice is made explicitly here.
    weights = (None if args.weight_type == "unweighted"
               else mt.gen_weights(sino, weight_type=args.weight_type))

    print("\n************** Run 4D MACE reconstruction **************")
    time0 = time.time()
    recon_4d, recon_dict = mace_model.recon(
        sino,
        weights=weights,
        max_iterations=args.max_mace_iterations,
        stop_threshold_change_pct=args.stop_threshold_change_pct,
        init_dir=init_dir,
        log_dir=log_dir,
    )
    run_time_h = (time.time() - time0) / 3600

    out_path = os.path.abspath(os.path.join(output_path, f"{stem}.npy"))
    np.save(out_path, recon_4d)
    print(f"\n[INFO] Total wall time: {run_time_h:.2f} hours.")
    print(f"[INFO] Iterations run: {recon_dict['recon_params']['iterations completed']} "
          f"of {args.max_mace_iterations}.")
    print(f"[INFO] Denoiser sigma_x: {recon_dict['recon_params']['denoiser sigma_x']:.6g} "
          f"({recon_dict['recon_params']['denoiser sigma_x source']}).")
    print(f"[INFO] Temporal/spatial spread: {recon_dict['recon_params']['temporal over spatial spread']}.")
    print(f"[INFO] Recon saved to: {out_path}")
    print(f"[INFO] Logs:           {os.path.abspath(log_dir)}")

    append_run_info(log_dir, args, dataset_dir, mace_model.num_frames, devices, run_time_h, out_path)

    # By default, one GIF is written per spatial plane, each playing over time.  The recon
    # takes hours and the GIFs take seconds, so writing all three avoids having to pick the
    # interesting plane before the run.  --gif_slice_axis narrows the output to one axis,
    # and fixing time instead steps through the slices of a single frame.  The slice index
    # is resolved here so that it can go into the file name.  Then runs that differ only
    # in the plane shown do not overwrite each other.
    gif_axes = (1, 2, 3) if args.gif_slice_axis is None else (args.gif_slice_axis,)
    for axis in gif_axes:
        index = (recon_4d.shape[axis] // 2 if args.gif_slice_index is None
                 else args.gif_slice_index)
        gif_path = os.path.join(output_path, f"{stem}_{'txyz'[axis]}{index}.gif")
        mt.save_volume_as_gif(recon_4d, gif_path, slice_axis=axis, slice_index=index,
                              vmin=0, vmax=args.gif_vmax)
        print(f"[INFO] GIF saved to:   {os.path.abspath(gif_path)}")


if __name__ == "__main__":
    main()
