import os
import argparse
import numpy
import  torch

# Training settings
parser = argparse.ArgumentParser(description='PyTorch EKE Training with HVD')
parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--model', type=str, default='widetrinet', metavar='N',
                    help='model type (default: resnet)', choices=['cnn', 'resnet', 'resnet_small', 'resnet_extrasmall', 'trinet', 'widetrinet'])
parser.add_argument('--ensemblemodel', type=str, default='voting', metavar='N',
                    help='ensemble model type (default: voting)', choices=['fusion', 'voting', 'bagging', 'gradientboosting', 'snapshot', 'adversarial', 'fastgeometic'])
parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                    help='input batch size for testing (default: 1000)')
parser.add_argument('--estimatornum', type=int, default=10, metavar='N',
                    help='number of ensemble models to train (default: 10)')
parser.add_argument('--epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=42, metavar='S',
                    help='random seed (default: 42)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')
parser.add_argument('--weighted-sampling', action='store_true', default=False,
                    help='use weighted sampling for training data')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')
parser.add_argument('--gradient-predivide-factor', type=float, default=1.0,
                    help='apply gradient predivide factor in optimizer (default: 1.0)')
parser.add_argument('--pretrained', dest='SAVED_MODEL', default=None,
                    help ='path to the pretrained model weights')
args = parser.parse_args()
