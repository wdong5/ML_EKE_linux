
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

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rc


import neatplot


from nn_models import EKEResnet, EKETriNet, EKEWideTriNet, EKEResnetSmall, EKEResnetExtraSmall, EKECNN

def metric_average(val, name):
    tensor = torch.tensor(val)
    avg_tensor = hvd.allreduce(tensor, name=name)
    return avg_tensor.item()

def infer():
    model.eval()
    test_loss = 0.
    loss_fn = nn.MSELoss(reduction = 'sum')
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        output = model(data)
        # sum up batch loss
        test_loss += loss_fn(output.squeeze(), target.squeeze()).item()

    # Horovod: use test_sampler to determine the number of examples in
    # this worker's partition.
    test_loss /= len(test_sampler)
    # Horovod: average metric values across workers.
    test_loss = metric_average(test_loss, 'avg_loss')

    # Horovod: print output only on first rank.
    if hvd.rank() == 0:
        print('\n{} - {} test samples: Average loss: {:.4f}\n'.format(
            model.name, test_samples, test_loss))

def plot():
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
    samples = [model(data).cpu().data.numpy() for _ in range(100)]

    m = np.mean(samples, axis=0).flatten()
    v = np.var(samples, axis=0).flatten()
    x_test = data.cpu().data.numpy()
    x_index = x_test[:,0]
    x_index, m = zip(*sorted(zip(x_index, m)))
    x_index, v = zip(*sorted(zip(x_index, v)))
    #x_index = np.sum(x_test, axis=1)
    x_index = np.array(x_index)
    m = np.array(m)
    v = np.array(v)
    print(m.shape, v.shape, x_index.shape)

    # x_test = data.cpu().data.numpy()[:100,0].flatten()
    # y_predict = samples.cpu().data.numpy()[:100].flatten()
    # m = np.mean(y_predict, axis=0)
    # v = np.var(y_predict, axis=0)
    # data.cpu().data.numpy().
    # print(x_test.shape, y_predict.shape)
    #plot mean and uncertainty
    fig = plt.figure(figsize=(6, 3), dpi=150)
    plt.plot(x_index, m)
    plt.fill_between(x_index, m-2*v**0.5, m+2*v**0.5, alpha=0.1) #plot two std (95% confidence)
    fig.savefig('UQ.png', dpi=200)


def plot_epistemic_uncertainty(ax, title, T=500):
    predictions = []
    # for _ in range(T):
    #   predictions += [model.predict(x_test,verbose=0)]
    for data, target in test_loader:
        if args.cuda:
            data, target = data.cuda(), target.cuda()
    predictions = [model(data).cpu().data.numpy() for _ in range(T)]

    mean, std = np.mean(np.array(predictions), axis=0), np.std(np.array(predictions), axis=0)
    x_test = data.cpu().data.numpy()
    y_test = target.cpu().data.numpy()
    fig = plt.figure(figsize=(6, 3), dpi=150)
    ax.plot(x_test, y_test, ls='--', color='green', label='test data')
    # ax.scatter(x_train, y_train, color='blue', label='train data')
    #ax.errorbar(x_test, mean, yerr=std, fmt='.', color='orange', label='uncertainty')
    #ax.set_title('{} - R2 {:.2f}'.format(title, r2_score(y_test, mean)))
    ax.set_title('{} - R2 {:.2f}'.format(title, np.square(np.subtract(y_test, mean)).mean()))
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend();
    fig.savefig('UQ.png', dpi=200)


from torchensemble import VotingClassifier



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

    X_test = np.load('./data/X_test_cf_all_4.npy')
    y_test = np.load('./data/y_test_cf_all_4.npy')

    train_features = X_test.shape[1]
    test_samples = X_test.shape[0]
    test_downsample = 100
    test_samples = (test_samples//test_downsample)
    X_test = torch.tensor(X_test[test_samples:test_samples+20, :])
    y_test = torch.tensor(y_test[test_samples:test_samples+20])
    print(len(X_test), len(y_test))

    test_dataset = TensorDataset(X_test, y_test)
    # Horovod: use DistributedSampler to partition the test data.
    test_sampler = torch.utils.data.distributed.DistributedSampler(
        test_dataset, num_replicas=hvd.size(), rank=hvd.rank())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.test_batch_size,
                                              sampler=test_sampler, **kwargs)
    weight_decay = 2e-4
    if args.model.lower() == 'trinet':
        model = EKETriNet(train_features, 3)
    elif args.model.lower() == 'resnet':
        model = EKEResnet(train_features)
        weight_decay = 2e-3
    elif args.model.lower() == 'widetrinet':
        model = EKEWideTriNet(train_features, depth=3, width=4) # best 3x4
    elif args.model.lower() == 'resnet_small':
        model = EKEResnetSmall(train_features)
        weight_decay = 2e-3
    elif args.model.lower() == 'resnet_extrasmall':
        model = EKEResnetExtraSmall(train_features)
        weight_decay = 2e-4
    elif args.model.lower() == 'cnn':
        model = EKECNN(train_features, groups=2, width_per_group=8)
        weight_decay = 2e-5

    if rank==0:
        print(model.name)

    if args.cuda:
        # Move model to GPU.
        if rank==0:
            print("CUDA found.")
        model.cuda()
        # If using GPU Adasum allreduce, scale learning rate by local_size.
        if args.use_adasum and hvd.nccl_built():
            lr_scaler = hvd.local_size()

    loss_str = 'custom' if args.weighted_sampling else 'mse'
    #args.SAVED_MODEL = './saved_model/' + args.model + '-' + args.epochs + '_' + loss_str + '_cf_all_4.pkl'
    args.SAVED_MODEL =  f'./saved_model/{model.name}-{args.epochs}_{loss_str}_cf_all_4.pkl'
    model = torch.load(args.SAVED_MODEL)

    infer()
    plot()
    fig, ax = plt.subplots(1,2,figsize=(20,5))
    plot_epistemic_uncertainty(ax=ax[0],title='CNN UQ figure')
