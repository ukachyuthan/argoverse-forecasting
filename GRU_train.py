import os
import shutil
import tempfile
import time
from typing import Any, Dict, List, Tuple, Union

import argparse
import joblib
from joblib import Parallel, delayed
import numpy as np
import pickle as pkl
from termcolor import cprint
import torch
import torch.nn as nn
import torch.nn.functional as F

from logger import Logger
import utils.baseline_config as config
import utils.baseline_utils as baseline_utils
from utils.lstm_utils import ModelUtils, LSTMDataset

use_cuda = torch.cuda.is_available()
if use_cuda:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
global_step = 0
best = float("inf")
np.random.seed(100)

ROLLOUT_LENS = [1, 10, 30]


def parse_arguments() -> Any:
    """Arguments for running the baseline.

    Returns:
        parsed arguments

    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_batch_size",
                        type=int,
                        default=512,
                        help="Test batch size")
    parser.add_argument("--model_path",
                        required=False,
                        type=str,
                        help="path to the saved model")
    parser.add_argument("--obs_len",
                        default=20,
                        type=int,
                        help="Observed length of the trajectory")
    parser.add_argument("--pred_len",
                        default=30,
                        type=int,
                        help="Prediction Horizon")
    parser.add_argument(
        "--normalize",
        action="store_true",
        help="Normalize the trajectories if non-map baseline is used",
    )
    parser.add_argument(
        "--use_delta",
        action="store_true",
        help="Train on the change in position, instead of absolute position",
    )
    parser.add_argument(
        "--train_features",
        default="",
        type=str,
        help="path to the file which has train features.",
    )
    parser.add_argument(
        "--val_features",
        default="",
        type=str,
        help="path to the file which has val features.",
    )
    parser.add_argument(
        "--test_features",
        default="",
        type=str,
        help="path to the file which has test features.",
    )
    parser.add_argument(
        "--joblib_batch_size",
        default=100,
        type=int,
        help="Batch size for parallel computation",
    )
    parser.add_argument("--use_map",
                        action="store_true",
                        help="Use the map based features")
    parser.add_argument("--use_social",
                        action="store_true",
                        help="Use social features")
    parser.add_argument("--test",
                        action="store_true",
                        help="If true, only run the inference")
    parser.add_argument("--train_batch_size",
                        type=int,
                        default=512,
                        help="Training batch size")
    parser.add_argument("--val_batch_size",
                        type=int,
                        default=512,
                        help="Val batch size")
    parser.add_argument("--end_epoch",
                        type=int,
                        default=5000,
                        help="Last epoch")
    parser.add_argument("--lr",
                        type=float,
                        default=0.001,
                        help="Learning rate")
    parser.add_argument(
        "--traj_save_path",
        required=False,
        type=str,
        help=
        "path to the pickle file where forecasted trajectories will be saved.",
    )
    return parser.parse_args()


class EncoderRNN(nn.Module):
    def __init__(self, ins=2, es=8, hs=16):
        super(EncoderRNN, self).__init__()
        self.hs = hs
        self.linear1 = nn.Linear(ins, es)
        self.lstm1 = nn.LSTMCell(es, hs)
        self.gru1 = nn.GRUCell(es, hs)


    def forward(self, x: torch.FloatTensor, hidden: Any) -> Any:

        embedded = F.relu(self.linear1(x))
        hidden = self.gru1(embedded, hidden)
        return hidden


class DecoderRNN(nn.Module):
    def __init__(self, ins=8, hs=16, ops=2):
        super(DecoderRNN, self).__init__()
        self.hs = hs

        self.linear1 = nn.Linear(ops, es)
        self.lstm1 = nn.LSTMCell(es, hs)
        self.linear2 = nn.Linear(hs, ops)
        self.gru1 = nn.GRUCell(es, hs)

    def forward(self, x, hidden):
        embedded = F.relu(self.linear3(F.relu(self.linear2(F.relu(self.linear1(x))))))
        hidden = self.lstm1(embedded, hidden)
        output = self.linear4(hidden)
        return output, hidden


def train_loop(train_loader, encoder, decoder, model_utils= ModelUtils, encoder_optim, decoder_optim, rollout_len, criterion):

    loss_array = []

    for i, data, gt, helpers in enumerate(train_loader):

        data = data.to(device)
        gt = gt.to(device)
        encoder.train()
        decoder.train()
        encoder_optim.zero_grad()
        decoder_optim.zero_grad()
        bs = data.shape[0]
        feature_size = data.shape[1]
        encoder_hidden = model_utils.init_hidden(bs, encoder.module.hs if use_cuda else encoder.hs)
        loss = 0
        for j in range(feature_size):

            ins = data[:,i,:]
            enc_hs = encoder(ins, encoder_hidden)

        decoder_ins = ins[:,:2]
        decoder_hs = encoder_hs
        decoder_op = torch.zeros(gt.shape).to(device)

        for k in range(rollout_len):

            decoder_op, decoder_hs = decoder(decoder_ins, decoder_hs)

            loss += criterion(decoder_op[:,:2], target[:,di,:2])

            decoder_ins = decoder_op

        loss = loss/rollout_len
        loss_array.append(loss)
        loss.backward()
        encoder_optim.step()
        decoder_optim.step()

    return loss, loss_array

def train(train_loader, epoch, criterion, logger, encoder, decoder, encoder_optim, decoder_optim, model_utils = ModelUtils, rollout_len):

    global global_step

    loss,_ = train_loop(train_loader, encoder, decoder, model_utils, encoder_optim, decoder_optim, rollout_len, criterion)

    if global_step % 1000 == 0:

    print(f"Train -- Epoch:{epoch}, loss:{loss}, Rollout:{rollout_len}")

    logger.scalar_summary(tag="Train/loss", value=loss.item(), step=epoch)
    global_step += 1


def validate(train_loader, epoch, criterion, logger, encoder, decoder, encoder_optim, decoder_optim, model_utils = ModelUtils, rollout_len):
    args = parse_arguments()
    global best

    loss, loss_array = train_loop(train_loader, encoder, decoder, model_utils, encoder_optim, decoder_optim, rollout_len, criterion)

    val_loss = sum(loss_array) / len(loss_array)

    if val_loss <= best:
        best = val_loss
        if args.use_map:
            save_dir = "saved_models/lstm_map"
        elif args.use_social:
            save_dir = "saved_models/lstm_social"
        else:
            save_dir = "saved_models/lstm"

        os.makedirs(save_dir, exist_ok=True)
        model_utils.save_checkpoint(
            save_dir,
            {
                "epoch": epoch + 1,
                "rollout_len": rollout_len,
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "best_loss": val_loss,
                "encoder_optimizer": encoder_optimizer.state_dict(),
                "decoder_optimizer": decoder_optimizer.state_dict(),
            },
        )

    logger.scalar_summary(tag="Val/loss", value=val_loss.item(), step=epoch)

    return val_loss




def main():

    args = parse_arguments()

    model_utils = ModelUtils()

    if args.use_map and args.use_social:
        baseline_key = "map_social"
    elif args.use_map:
        baseline_key = "map"
    elif args.use_social:
        baseline_key = "social"
    else:
        baseline_key = "none"

    data_dict = baseline_utils.get_data(args, baseline_key)

    criterion = nn.MSELoss()
    encoder = EncoderRNN(ins=len(baseline_utils.BASELINE_INPUT_FEATURES[baseline_key]))
    decoder = DecoderRNN(ops=2)
    if use_cuda:
        encoder = nn.DataParallel(encoder)
        decoder = nn.DataParallel(decoder)
    encoder.to(device)
    decoder.to(device)
    encoder_optim = torch.optim.Adam(encoder.parameters(), lr=args.lr)
    decoder_optim = torch.optim.Adam(decoder.parameters(), lr=args.lr)

    rollout_id = 0

    log_dir = os.path.join(os.getcwd(), "lstm_logs", baseline_key)

    train_dataset = LSTMDataset(data_dict, args, "train")
    val_dataset = LSTMDataset(data_dict, args, "val")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True, drop_last=False, collate_fn=model_utils.my_collate_fn)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.val_batch_size, drop_last=False, shuffle=True, collate_fn=model_utils.my_collate_fn,)

    epoch = 0
    for i in range(rollout_id, len(ROLLOUT_LENS)):
        rollout_len = ROLLOUT_LENS[i]
        logger = Logger(log_dir, name="{}".format(rollout_len))
        while epoch < args.end_epoch:
            start = time.time()
            train(train_loader, epoch, criterion, logger, encoder, decoder, encoder_optim, decoder_optim, model_utils, rollout_len)
            epoch += 1
            if epoch % 5 == 0:
                val_loss = validate(val_loader, epoch, criterion, logger, encoder, decoder, encoder_optim, decoder_optim, model_utils, rollout_len)
                print("Validation loss =", val_loss, "after", epoch, "epochs")


if __name__ == "__main__":
    main()
