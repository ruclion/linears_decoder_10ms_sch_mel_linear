"""
Synthesis waveform from trained model.

usage: synthesis.py [options]

options:
    --result_dir=<s>         File directory to save result [default: ./result].
    -h, --help               Show help message.
"""

from torch.utils import data as data_utils
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch import optim
from torch import nn
import torch
from tensorboardX import SummaryWriter
# from tensorboard_logger import log_value
# import tensorboard_logger

# from nnmnkwii.datasets import FileSourceDataset, FileDataSource
from os.path import join, dirname, expanduser
from matplotlib import pyplot as plt
from tqdm import tqdm, trange
from scipy.io import wavfile
from docopt import docopt
import librosa.display
import numpy as np
import librosa
import scipy
import json
# import nltk
import sys
import os

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib

from hparams_dblstm import hparams as hparams_dblstm
# from text import text_to_sequence, symbols
# from util.plot import plot_alignment
from model_dblstm import DBLSTM

from hparams_timit import hparams as timit_hparams
from utils import spectrogram2wav

use_cuda = torch.cuda.is_available()


def tts(dblstm_model, ppg_path):
    """Convert text to speech waveform given a Tacotron model.
    """
    if use_cuda:
        dblstm_model = dblstm_model.cuda()

    dblstm_model.eval()

    ppg = np.load(ppg_path)

    print(ppg)

    ppg = Variable(torch.from_numpy(ppg)).unsqueeze(0).float()
    
    if use_cuda:
        ppg = ppg.cuda()

    print("INNER PPG:", ppg.shape)

    _, spec_pred  = dblstm_model(ppg)

    print("INNER SPEC 1:", spec_pred)

    spec_pred = spec_pred[0].cpu().data.numpy()

    print("INNER SPEC 2:", spec_pred)

    return spec_pred


if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    result_dir = args["--result_dir"]

    dblstm_model = DBLSTM(hparams_dblstm.batch_size)
    dblstm_checkpoint_path = "./checkpoints-dblstm/5-happy-218tolpc/checkpoint_step000178500.pth"

    dblstm_checkpoint = torch.load(dblstm_checkpoint_path)
    dblstm_model.load_state_dict(dblstm_checkpoint["state_dict"])

    ppg_paths = [pos_json for pos_json in os.listdir(result_dir) if pos_json.endswith('.npy')]
    print("check ppg_paths:", ppg_paths)

    # train_set_spec_path = "/home/zhangwq01/ChangheSong/data/LJSpeech-ppg-train-smaller/ljspeech-spec-00083.npy"
    # train_set_spec = np.load(train_set_spec_path)
    # train_set_spec = np.exp(train_set_spec)
    # audio = spectrogram2wav(train_set_spec.T, timit_hparams.n_fft, timit_hparams.win_length, timit_hparams.hop_length, timit_hparams.n_iter)
    # audio = inv_preemphasis(audio, coeff=timit_hparams.preemphasis)

    # train_set_wav_path = join(result_dir, "train_set_sample.wav")
    # waveform = audio * 32767 / max(0.01, np.max(np.abs(audio)))
    # wavfile.write(train_set_wav_path, timit_hparams.sample_rate, waveform.astype(np.int16))


    for idx, ppg_path in enumerate(ppg_paths):
      base_name = ppg_path.split('.')[0]
      print("Reading ppg from ppg_path:", join(result_dir, ppg_path))
      spectrogram = tts(dblstm_model, join(result_dir, ppg_path))
      result_spec_path = join(result_dir, "{}_sample_from_{}-step-178500.npy".format(idx, base_name))
      np.save(result_spec_path, spectrogram)

    print("Finished! Check out {} for generated audio samples.".format(result_dir))
    sys.exit(0)
