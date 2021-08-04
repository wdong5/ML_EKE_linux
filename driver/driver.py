"""
Driver script for the execution of SmartSim enabled
MOM6 simulations with the OM4_05 configuration.

Note: certain parameters and calls will need to
be updated depending on the resources of the
system.

This script assumes launching on a slurm cluster
with at least
   - 228 CPU nodes with 96 cpus (including hyperthreads)
   - 16 nodes with P100 GPUs and 36 cpu cores
This can be changed to suit your system with the parameters
listed below

To run the exact same experiment as our paper, increase
the time in both batch jobs and the number of days
to 10 years.
"""

from glob import glob
from smartsim import Experiment
from smartsim.settings import SbatchSettings, SrunSettings
from smartsim.database import SlurmOrchestrator
from smartsim.utils.log import log_to_file

#for local launcher
from smartsim.settings import RunSettings
from smartsim.database import Orchestrator


# logging output saved to file
log_to_file("./driver.log")

# experiment parameters
# DB_NODES = 1
ENSEMBLE_NODES = 1
ENSEMBLE_MEMBERS = 1
# MOM6_EXE = "/home/dongw/NCAR_ML_EKE/MOM6/build/gnu/ice_ocean_SIS2/repro/MOM6"
MOM6_EXE = "/home/dongw/NCAR_OCEAN_only/MOM6/build/gnu/ocean_only/repro/MOM6"

# name of experiment where output will be placed
experiment = Experiment("AI-EKE-MOM6", launcher="local")

# define slurm execution settings for a single member
# of the ensemble
# mom_opts= {
#     "nodes": 1,
#     "ntasks": 1,
#     "exclusive": None,
# }
#mom_settings = SrunSettings(MOM6_EXE, run_args=mom_opts)
mom_settings = RunSettings(MOM6_EXE) #, run_args={"-np": "2"})


# define batch parameters for entire ensemble
# batch_opts = {
#     "mincpus": 96,
#     "ntasks-per-node": 48,
#     "exclusive": None
# }
#ensemble_batch = SbatchSettings(nodes=ENSEMBLE_NODES, time="10:00:00", batch_args=batch_opts)

# create reference to MOM6 ensemble
mom_ensemble = experiment.create_ensemble("MOM",
                                          #batch_settings=ensemble_batch,
                                          run_settings=mom_settings,
                                          replicas=ENSEMBLE_MEMBERS)

# Attach input files and configuration files to each
# MOM6 simulation
mom_ensemble.attach_generator_files(
    to_configure=glob("../MOM6/MOM6_config/*"),
    to_copy="../MOM6/OM4_025",
    to_symlink="../MOM6/INPUT",
)

# configs to write into 'to_configure' files listed
# above. If you change the number of processors for
# each MOM6 simulation, you will need to change this.
MOM6_config = {
    "SIM_DAYS": 1, # length of simlations
    "DOMAIN_LAYOUT": "32,36",
    "MASKTABLE": "mask_table.242.32x36"
    }
for model in mom_ensemble:
    model.params = MOM6_config

# register models so keys don't overwrite each other
# in the database
for model in mom_ensemble:
    model.register_incoming_entity(model)

# creation of ML database specific to Slurm.
# there are also PBS, Cobalt, and local variants
# db = SlurmOrchestrator(db_nodes=DB_NODES, time="10:00:00", threads_per_queue=4)
# db.set_cpus(36)
# db.set_batch_arg("constraint", "P100")
# db.set_batch_arg("exclusive", None)

REDIS_PORT=6815
db = Orchestrator(port=REDIS_PORT, threads_per_queue=4)

# generate run directories and write configurations
experiment.generate(mom_ensemble, db, overwrite=True)

# start the database and ensemble batch jobs.
experiment.start(mom_ensemble, db,  block=False, summary=True)

# print a summary of the run.
print(experiment.summary())
