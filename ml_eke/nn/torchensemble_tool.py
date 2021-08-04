import argparse
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torch.utils.data.distributed
from torch.utils.data import TensorDataset
import horovod.torch as hvd
import numpy as np
from scipy.stats import norm
import time
from my_args import args

from nn_models import EKEResnet, EKETriNet, EKEWideTriNet, EKEResnetSmall, EKEResnetExtraSmall, EKECNN

from torchensemble import FusionRegressor, VotingRegressor, BaggingRegressor, GradientBoostingRegressor, SnapshotEnsembleRegressor, AdversarialTrainingRegressor, FastGeometricRegressor

def ensemble_model(train_features):
    # model ensemble type
    if args.ensemblemodel.lower() == 'fusion':
        # Define the ensemble
        ensemble = FusionRegressor(
            estimator=EKEResnet(train_features),               # here is your deep learning model
            n_estimators=args.estimatornum,                        # number of base estimators
        )
    elif args.ensemblemodel.lower() == 'voting':
        # Define the ensemble
        ensemble = VotingRegressor(
            estimator=EKEResnet(train_features),               # here is your deep learning model
            n_estimators=args.estimatornum,                        # number of base estimators
        )
    elif args.ensemblemodel.lower() == 'bagging':
        # Define the ensemble
        ensemble = BaggingRegressor(
            estimator=EKEResnet(train_features),               # here is your deep learning model
            n_estimators=args.estimatornum,                        # number of base estimators
        )
    elif args.ensemblemodel.lower() == 'gradientboosting':
        # Define the ensemble
        ensemble = GradientBoostingRegressor(
            estimator=EKEResnet(train_features),               # here is your deep learning model
            n_estimators=args.estimatornum,                        # number of base estimators
        )
    elif args.ensemblemodel.lower() == 'snapshot':
        # Define the ensemble
        ensemble = SnapshotEnsembleRegressor(
            estimator=EKEResnet(train_features),               # here is your deep learning model
            n_estimators=args.estimatornum,                        # number of base estimators
        )
    elif args.ensemblemodel.lower() == 'adversarial':
        # Define the ensemble
        ensemble = AdversarialTrainingRegressor(
            estimator=EKEResnet(train_features),               # here is your deep learning model
            n_estimators=args.estimatornum,                        # number of base estimators
        )
    elif args.ensemblemodel.lower() == 'fastgeometic':
        # Define the ensemble
        ensemble = FastGeometricRegressor(
            estimator=EKEResnet(train_features),               # here is your deep learning model
            n_estimators=args.estimatornum,                        # number of base estimators
        )

    # Set the optimizer
    ensemble.set_optimizer(
        "Adam",                                 # type of parameter optimizer
         lr=1e-3,            # learning rate of the optimizer
         weight_decay=5e-4   # weight decay of the optimizer
    )

    # Set the learning rate scheduler
    ensemble.set_scheduler(
        "CosineAnnealingLR",                    # type of learning rate scheduler
         T_max=50,                           # additional arguments on the scheduler
    )

    # Train the ensemble
    ensemble.fit(
        train_loader,
        epochs=50,                          # number of training epochs
        save_model=True,        #Specify whether to save the model parameters
        save_dir="./voting_5/",
    )

    # Evaluate the ensemble
    # for data, target in test_loader:
    #     if args.cuda:
    #         data, target = data.cuda(), target.cuda()
        #acc = ensemble.predict(data)         # testing accuracy

    acc = ensemble.evaluate(test_loader)

    print(acc)


if __name__ == '__main__':
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    # Horovod: initialize library.
    hvd.init()
    rank = hvd.rank()
    torch.manual_seed(args.seed)

    if args.cuda:
        # Horovod: pin GPU to local rank.
        torch.cuda.set_device(hvd.local_rank())
        torch.cuda.manual_seed(args.seed)

    # Horovod: limit # of CPU threads to be used per worker.
    torch.set_num_threads(1)

    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    # When supported, use 'forkserver' to spawn dataloader workers instead of 'fork' to prevent
    # issues with Infiniband implementations that are not fork-safe
    if (kwargs.get('num_workers', 0) > 0 and hasattr(mp, '_supports_context') and
            mp._supports_context and 'forkserver' in mp.get_all_start_methods()):
        kwargs['multiprocessing_context'] = 'forkserver'

    X_train = np.load('./data/X_train_cf_all_4.npy')
    X_test = np.load('./data/X_test_cf_all_4.npy')

    y_train = np.load('./data/y_train_cf_all_4.npy')
    y_test = np.load('./data/y_test_cf_all_4.npy')

    train_samples = X_train.shape[0]
    train_features = X_train.shape[1]
    test_samples = X_test.shape[0]

    # For fast model research: typically, ~1.0M samples per node is feasible.
    max_samples = 1000000
    if args.weighted_sampling:
        max_samples *= 10  # allow to select 1/10 of samples according to weights
    train_size = min(max_samples*hvd.size(), train_samples)//hvd.size()*hvd.size()

    X_train = torch.tensor(X_train[:train_size, :])
    y_train = torch.tensor(y_train[:train_size])

    test_downsample = 100
    test_samples = (test_samples//test_downsample)
    X_test = torch.tensor(X_test[:test_samples, :])
    y_test = torch.tensor(y_test[:test_samples])
    print(train_size, test_samples)

    if args.weighted_sampling: #wenqian:not going into this condition
        train_chunk_size = train_size // hvd.size()
        X_train = X_train[rank*train_chunk_size:(rank+1)*train_chunk_size,:]
        y_train = y_train[rank*train_chunk_size:(rank+1)*train_chunk_size]
        weights = compute_weights(y_train)
        train_dataset = TensorDataset(X_train, y_train)
        train_sampler = torch.utils.data.WeightedRandomSampler(weights, train_chunk_size//10,
                replacement=False)
        if rank==0:
          print("Training on {} out of {} training samples"
                .format(len(y_train)*hvd.size(), train_samples))
    else:
        train_dataset = TensorDataset(X_train, y_train)
        # Horovod: use DistributedSampler to partition the training data.
        train_sampler = torch.utils.data.distributed.DistributedSampler(
            train_dataset, num_replicas=hvd.size(), rank=hvd.rank())

        if rank==0:
          print("Training on {} out of {} training samples"
                .format(train_size, train_samples))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=train_sampler, **kwargs)

    test_dataset = TensorDataset(X_test, y_test)
    # Horovod: use DistributedSampler to partition the test data.
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                              sampler=test_sampler, **kwargs)
    train_features = X_train.shape[1]

    # train ensemble models
    #ensemble_model(train_features)

    # load ensemble nn_models
    from torchensemble.utils import io
    save_dir = "/home/dongw/NCAR_ML_EKE/ml_eke/nn/saved_model/VotingRegressor_EKEResnet_5_ckpt.pth"
    io.load(new_ensemble, save_dir)  # reload
    accuracy = ensemble.evaluate(test_loader)
