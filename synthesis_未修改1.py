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

from nnmnkwii.datasets import FileSourceDataset, FileDataSource
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
import nltk
import sys
import os

import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib

from hparams_dblstm import hparams as hparams_dblstm
from text import text_to_sequence, symbols
from util.plot import plot_alignment
from model_dblstm import DBLSTM

from hparams_timit import hparams as timit_hparams
from utils import spectrogram2wav

use_cuda = torch.cuda.is_available()


def _stft(y):
  return librosa.stft(y=y, n_fft=timit_hparams.n_fft, hop_length=timit_hparams.hop_length, win_length=timit_hparams.win_length)

def _istft(y):
  return librosa.istft(y, hop_length=timit_hparams.hop_length, win_length=timit_hparams.win_length)

def _db_to_amp(x):
  return np.power(10.0, x * 0.05)

def _denormalize(S):
  return (np.clip(S, 0, 1) * - timit_hparams.min_level_db) + timit_hparams.min_level_db

def _griffin_lim(S):
  '''librosa implementation of Griffin-Lim
  Based on https://github.com/librosa/librosa/issues/434
  '''
  angles = np.exp(2j * np.pi * np.random.rand(*S.shape))
  S_complex = np.abs(S).astype(np.complex)
  y = _istft(S_complex * angles)
  for i in range(timit_hparams.griffin_lim_iters):
    angles = np.exp(1j * np.angle(_stft(y)))
    y = _istft(S_complex * angles)
  return y

def inv_preemphasis(x):
  return scipy.signal.lfilter([1], [1, -timit_hparams.preemphasis], x)


def tts(dblstm_model, ppg_path):
    """Convert text to speech waveform given a Tacotron model.
    """
    if use_cuda:
        dblstm_model = dblstm_model.cuda()

    dblstm_model.eval()

    with open(ppg_path,'r') as load_f:
        ppg = json.load(load_f)
    ppg = np.array(ppg)

    print(ppg)

    ppg = Variable(torch.from_numpy(ppg)).unsqueeze(0).float()
    
    if use_cuda:
        ppg = ppg.cuda()

    print("INNER PPG:", ppg.shape)

    _, spec_pred  = dblstm_model(ppg)

    print("INNER SPEC 1:", spec_pred)

    spec_pred = spec_pred[0].cpu().data.numpy()

    print("INNER SPEC 2:", spec_pred)

    S = _db_to_amp(_denormalize(spec_pred.T) + timit_hparams.ref_level_db)
    audio = inv_preemphasis(_griffin_lim(S ** timit_hparams.power))
    # result_wav_path = join(checkpoint_dir, "checkpoint_step_{}_original.wav".format(global_step))
    audio = audio * 32767 / max(0.01, np.max(np.abs(audio)))
    # wavfile.write(result_wav_path, timit_hparams.sample_rate, original_audio.astype(np.int16))

    # audio = spectrogram2wav(spec_pred.T, timit_hparams.n_fft, timit_hparams.win_length, timit_hparams.hop_length, timit_hparams.n_iter)
    # audio = inv_preemphasis(audio, coeff=timit_hparams.preemphasis)

    return audio, spec_pred


if __name__ == "__main__":
    args = docopt(__doc__)
    print("Command line args:\n", args)
    result_dir = args["--result_dir"]

    dblstm_model = DBLSTM(hparams_dblstm.batch_size)
    dblstm_checkpoint_path = "./checkpoints-dblstm/with_l1_loss_500/checkpoint_step000010050.pth"
    # dblstm_checkpoint_path = "./checkpoints-dblstm/smaller_long_train/checkpoint_step000010200.pth"
    # dblstm_checkpoint_path = "./checkpoints-dblstm/without_exp_smaller/checkpoint_step000002250.pth"
    # dblstm_checkpoint_path = "./checkpoints-dblstm/without_exp/checkpoint_step000009000.pth"
    # dblstm_checkpoint_path = "./checkpoints-dblstm/retry/checkpoint_step000010000.pth"
    dblstm_checkpoint = torch.load(dblstm_checkpoint_path)
    dblstm_model.load_state_dict(dblstm_checkpoint["state_dict"])

    text_paths = [pos_json for pos_json in os.listdir(result_dir) if pos_json.endswith('.json')]
    print("check text_paths:", text_paths)

    # train_set_spec_path = "/home/zhangwq01/ChangheSong/data/LJSpeech-ppg-train-smaller/ljspeech-spec-00083.npy"
    # train_set_spec = np.load(train_set_spec_path)
    # train_set_spec = np.exp(train_set_spec)
    # audio = spectrogram2wav(train_set_spec.T, timit_hparams.n_fft, timit_hparams.win_length, timit_hparams.hop_length, timit_hparams.n_iter)
    # audio = inv_preemphasis(audio, coeff=timit_hparams.preemphasis)

    # train_set_wav_path = join(result_dir, "train_set_sample.wav")
    # waveform = audio * 32767 / max(0.01, np.max(np.abs(audio)))
    # wavfile.write(train_set_wav_path, timit_hparams.sample_rate, waveform.astype(np.int16))


    for idx, text_path in enumerate(text_paths):
      print("Reading ppg from text_path:", join(result_dir, text_path))
      waveform, spectrogram = tts(dblstm_model, join(result_dir, text_path))
      result_wav_path = join(result_dir, "{}_sample.wav".format(idx))
    #   waveform = waveform * 32767 / max(0.01, np.max(np.abs(waveform)))
      wavfile.write(result_wav_path, timit_hparams.sample_rate, waveform.astype(np.int16))

      result_spec_path = join(result_dir, "{}_sample_spec.png".format(idx))
      plt.imshow(spectrogram.T[:1000, :], cmap='hot', interpolation='nearest')
      plt.xlabel('fram nums')
      plt.ylabel('spec')
      plt.tight_layout()
      plt.savefig(result_spec_path, format='png')

    print("Finished! Check out {} for generated audio samples.".format(result_dir))
    sys.exit(0)
