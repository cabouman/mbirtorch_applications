# Repository for MBIRTORCH demonstrations
## Overview
This repository contains scripts that demonstrate the usage of [MBIRTORCH](https://github.com/cabouman/mbirtorch) in selected CT applications.
## Quick start guide
1. *Install the conda environment and package for MBIRTORCH.* Instructions available [here](https://github.com/cabouman/mbirtorch).
2. *Clone this repository:*
   ```
   git clone git@github.com:cabouman/mbirtorch_applications.git
   ```
3. Run demo scripts for the application of your choice.

Available applications include:
   * Cone-beam CT reconstruction using NorthStar Instrument (NSI) data:
     ```
     cd mbirtorch_applications/nsi
     python demo_fdk_mbir_compare.py
     ```
   * View Selection using VCL as published in ICCP/PAMI 2025:
     ```
     cd mbirtorch_applications/vcls
     python demo_vcls.py
     ```
   * Parallel-beam CT reconstruction using NERSC/Advanced Light Source (ALS) data:
     ```
     cd mbirtorch_applications/nersc
     python demo_nersc.py
     ```
