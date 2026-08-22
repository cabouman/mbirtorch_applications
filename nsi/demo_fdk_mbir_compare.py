import os
import time

import mbirtorch as mt
import mbirtorch.preprocess as mtp
import pprint

pp = pprint.PrettyPrinter(indent=4)

if __name__ == "__main__":
    print('This script is demonstrates the preprocessing and reconstruction of NSI an dataset\
    \n\t using both FDK and MBIR reconstruction.\n')

    # recon parameters
    sharpness = 1.0
    snr_db = 30.0
    downsample = 4
    verbose = 1              # Print, but do not display plots

    # User defined paths
    output_path = './output/nsi_demo_mar/'   # path to store output recon images
    os.makedirs(output_path, exist_ok=True)  # mkdir if directory does not exist
    download_dir = './demo_data/'            # Directory to store downloaded data

    # Prompt the user for dataset choice
    choice = input("Download dataset with metal? (Y/n): ").strip().lower()
    if choice == 'n':
        # URL to test phantom without metal
        dataset_url = 'https://www.datadepot.rcac.purdue.edu/bouman/data/demo_nsi_vert_no_metal_all_views.tgz'
        metal = False
    else:
        # URL to test phantom with metal
        dataset_url = 'https://www.datadepot.rcac.purdue.edu/bouman/data/demo_nsi_vert_metal_all_views.tgz'
        metal = True
    print(f"Selected dataset URL: {dataset_url}")

    # Download and extract data. Then set path to NSI scan directory.
    dataset_dir = mt.download_and_extract(dataset_url, download_dir)

    # preprocessing parameters
    downsample_factor = [downsample, downsample]  # downsample factor of scan view images along detector rows and detector columns.
    subsample_view_factor = 2*downsample  # view subsample factor.

    print("\n************** NSI dataset preprocessing **************")
    sino, ct_model = \
        mtp.nsi.get_sino_and_model(dataset_dir,
                                   downsample_factor=downsample_factor,
                                   subsample_view_factor=subsample_view_factor)

    print("\n************** Set up MBIRTORCH model **************")
    # Set user determined parameter values
    ct_model.set_params(sharpness=sharpness, snr_db=snr_db, verbose=1)

    # Print out model parameters
    ct_model.print_params()

    print("\n************** Calculate sinogram weights **************")
    weights = mt.gen_weights(sino, weight_type='transmission_root')

    print("\n************** Perform FDK reconstruction **************")
    # ##########################
    # Perform FDK reconstruction
    fdk_recon = ct_model.recon_direct(sino)
    #mt.slice_viewer(fdk_recon)

    print("\n************** Perform MBIR reconstruction **************")
    # #### Perform MBIR reconstruction
    time0 = time.time()
    mbir_recon, mbir_recon_dict = ct_model.recon(sino, weights=weights)
    elapsed = time.time() - time0
    print('Elapsed time for recon is {:.3f} seconds'.format(elapsed))

    # #### Print out parameters used in recon
    pprint.pprint(mbir_recon_dict['recon_params'])

    # #### Save MBIR reconstruction to HDF5 file output
    mt.export_recon_hdf5(os.path.join(output_path, "recon.h5"), mbir_recon, recon_dict=mbir_recon_dict, remove_flash=True)

    if verbose >1:
        # Display FDK versus MBIR
        vmin = 0
        vmax = downsample_factor[0] * 0.025
        mt.slice_viewer(fdk_recon, mbir_recon, data_dicts=[None, mbir_recon_dict], vmin=0, vmax=vmax, slice_label= ["FDK Recon", "MBIR Recon"], title='Axial Slice')
