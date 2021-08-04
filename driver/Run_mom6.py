
import os
from smartsim import Experiment
from smartsim.settings import RunSettings

from smartredis import Client
from smartsim.database import Orchestrator
import numpy as np

import torch
import torch.nn as nn

REDIS_PORT=6810

# Create the SmartSim Experiment.
#experiment = Experiment("AI-EKE-MOM6", launcher="slurm")
experiment = Experiment("AI-EKE-MOM6", launcher="local")

# create and start a database
orc = Orchestrator(port=REDIS_PORT)
experiment.generate(orc)
experiment.start(orc,  block=False)
client = Client(address='127.0.0.1:'+str(REDIS_PORT), cluster=False)

# send_tensor = np.ones((4,3,3))
#
# client.put_tensor("tutorial_tensor_1", send_tensor)
#
# receive_tensor = client.get_tensor("tutorial_tensor_1")
#
# print('Receive tensor:\n\n', receive_tensor)

#
# Set the model in the Redis database from the file
client.set_model_from_file("EKEResnet", "/home/dongw/NCAR_ML_EKE/ml_eke/nn/trained_models/ResNet_4_custom.cpu.pt", "TORCH", "CPU")

# Put a tensor in the database as a test input
data = torch.rand(1,4).numpy()
print(data)
client.put_tensor("EKEResnet_input", data)

# Run model and retrieve the output
client.run_model("EKEResnet", inputs=["EKEResnet_input"], outputs=["EKEResnet_output"])
out_data = client.get_tensor("EKEResnet_output")
print(out_data)
experiment.stop(orc)
