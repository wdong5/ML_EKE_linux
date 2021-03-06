

[![Input Dataset DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4682270.svg)](https://doi.org/10.5281/zenodo.4682270)

# Setup MOM6 environment on Linux system

This is a record to help people setup the environment with their on system and sucessfully reproduce the experiments of ML_EKE_linux (https://github.com/CrayLabs/NCAR_ML_EKE).

# Reproduction

We strove to make this project as reproducible as possible. If you find any places where
reproducibility is lacking or could be improved, please file an issue to let us know.

## Installation

Below are the steps to install the various components needed to reproduce the
results in the paper for the ML-EKE parameterization with SmartSim

## My Environment
GPUs: Nvidia V100
python:3.7.5

### Clone

First, clone the repository with the MOM6 submodule

```bash
git clone --recursive https://github.com/CrayLabs/NCAR_ML_EKE.git
```
Next you need to install SmartSim 0.3.0 and SmartRedis 0.1.0 with
dependencies. We provide the capability to run this use case on
CPU or GPU-enabled nodes. Please be sure to build SmartSim accordingly.

Note: There is a ``env`` file in ``MOM6/build/gnu`` that specifies
the programming environment we built with. Specifically, we used
the GNU toolchain with gcc 8.3.1. Source this before building 
anything (if you are on a Cray or HPC system with modules)

Follow the [instructions for the full installation](https://www.craylabs.org/build/html/installation.html#full-installation) of
both libraries and be sure to build for the architecture you
want to run the tests on (e.g. CPU or GPU). These results were
run off of the releases, downloadable on their respective github pages.

In addition, when installing SmartRedis, make the static library
that will be compiled into MOM6. 

```bash
cd smartredis-0.1.0
source setup_env.sh
make lib
```

Do not exit the terminal used to build SmartRedis, environment
variables are set that are used in the compilation of MOM6 to
ease the build process.

### Build MOM6

Assuming you checked out the repository, built and installed
SmartSim and SmartRedis, you now need to build MOM6. A script
is included for the compilation.

```bash
cd MOM6
./compile_ice_ocean_SIS2.sh
```
Go grab a coffee and wait for that to complete. If there are issues
with the build, try sourcing the ``setup_env.sh`` script in SmartRedis
and try the script again.

After modification, I build the executable files of both the ocean_sis2 and the ocean_only.


### Download the MOM6 input data

We host and include the input data we used to run MOM6 along with
pre-trained models and scripts we used for the paper.

To download the data, either at the DOI link at the top
of the repo or [here](https://doi.org/10.5281/zenodo.4682270)

Download the data into ``MOM6/INPUT``. The MOM6 input dataset
pretrained models and scripts for the SmartSim workload are all
included.

### checke the dataset link of MOM6 SIS2 input
```bash
cd MOM6/INPUT
ll .datasets
du -s "WOA05_pottemp_salt.nc" 
```
please check if this links to a right position.

### download special MOM6_SIS6 dataset
```bash
cd ../../; mkdir MOM6_SIS6_input; cd MOM6_SIS6_input;
wget ftp://ftp.gfdl.noaa.gov/perm/Alistair.Adcroft/MOM6-testing/CORE.tgz;
wgetftp://ftp.gfdl.noaa.gov/perm/Alistair.Adcroft/MOM6-testing/obs.tgz;
wget ftp://ftp.gfdl.noaa.gov/perm/Alistair.Adcroft/MOM6-testing/obs.woa13.tgz;
wget ftp://ftp.gfdl.noaa.gov/perm/Alistair.Adcroft/MOM6-testing/OM4_025.tgz;
tar xvzf obs.woa13.tgz;
tar xvzf OM4_025.tgz;
tar xvzf obs.woa13.tgz;
tar xvzf obs.tgz;
cd ./MOM6/INPUT/
rm .datasets
ln -s ../../../MOM5_INPUT .datasets
ll .datasets
```


## Run

Before running the SmartSim driver script, be sure that
the computational setup described by the script suits your
system. 

Once configured, the entire workload can be executed with

```bash
cd driver
python driver.py
```
See the updated driver.py file, and we use the "local" launcher instead.
To add account or other information please consult the [SmartSim API Docs](https://www.craylabs.org/build/html/api/smartsim_api.html#smartsim-api)

If you can not run "driver.py" with smartredis directly, you're good to try below (make sure block= Flase in experiment.start):

```bash
python driver.py
cd ./driver/AI-EKE-MOM6/MOM/MOM_0/
mpirun -n 16 /home/dongw/org_NCAR_ML_EKE/MOM6/build/gnu/ice_ocean_SIS2/repro/MOM6
```


## Check
some useful commands to test the status of Smartsim and Smartredis
```bash
ps aux | grep redis-server
export SSDB="localhost:6704"
```

A few important files in MOM6 simulation
```bash
MOM_parameter_doc.short
MOM_override
```

## Results

When the workload is run successfully, there will be a 
``AI-EKE-MOM6`` directory with all of the output from each
ensemble member. Included are timings in each ``MOM_<ensemble_number>.out``
which look like

```text
(SmartRedis put tensor)               0.003483      0.019676      0.010159      0.003618  0.000    41     0   909
(SmartRedis run model)                0.946908      2.807891      1.943015      0.462866  0.008    41     0   909
(SmartRedis unpack tensor )           0.001292      0.011080      0.001756      0.000605  0.000    41     0   909
```
You can use these to examine the overall timings of each operation that uses 
SmartRedis inside MOM6

## Variants

Below are some variants that can be run for examining different
configurations or for different systems

### Reference Simulation

The compare the SmartSim approach vs the MEKE paramterization, 
change the line in ``MOM6/OM4_025/MOM_override`` from

``EKE_SOURCE='sr'``
to 
``EKE_SOURCE='prog'``

Then comment out the parts of the driver script that create
and launch the database. Once commented out, you can run the
driver script as normal and the MOM6 simulations will be
executed with the MEKE paramterization instead of the
Smartsim approach.

### CPU-only Machines

If you don't have GPU nodes, don't worry, we have you covered.
Follow the instructions to install SmartSim for CPU. We include
a pre-trained model for CPU inference as well. To use the CPU
model, change the line in ``MOM6/OM4_025/MOM_override`` from

``SMARTREDIS_MODEL='ncar_ml_eke.gpu.pt'``
to
``SMARTREDIS_MODEL='ncar_ml_eke.pt'``

both models are included in the input data directory you downloaded
earlier.


## Contributors

The collaboration was a joint effort between Hewlett Packard Enterprise (HPE),
National Center for Atmosheric Research (NCAR), and the University of Victoria (U Vic)

Contributors in no particular order

 - Andrew Shao (U Vic)
 - Sam Partee (HPE)
 - Alessandro Rigazzi (HPE)
 - Scott Bachman (NCAR)
 - Gustavo Marques (NCAR)
 - Matthew Ellis (HPE)
