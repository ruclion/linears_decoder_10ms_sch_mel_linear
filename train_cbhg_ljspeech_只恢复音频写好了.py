import os
import time
import numpy as np
from tqdm import tqdm
from os.path import join

from dataload_dblstm import DBlstmDtaset
from model_dblstm import DBLSTM


from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch import optim
import torch.nn as nn
import torch


from audio import spectrogram2wav_for_ppg_cbhg, write_wav


# 平时均用gpu训练，在外面指定命令行用哪一块gpu：CUDA_VISIBLE_DEVICES=0 python train_cbhg_ljspeech
use_cuda = torch.cuda.is_available()
assert use_cuda is True

global_step = 0
global_epoch = 0
cuda_num = 2


def eval_model_generate(spec, spec_pred, length, checkpoint_dir, global_step):
  print("EVAL LENGTH:", length)
  print("EVAL SPEC PRED SHAPE:", spec_pred.shape)
  # GriffinLim的power和iter不会设置，也不知道有什么影响
  y_pred = spectrogram2wav_for_ppg_cbhg(spec_pred, gl_power = 1., gl_iter = 100)
  pred_wav_path = join(checkpoint_dir, "checkpoint_step_{}_pred.wav".format(global_step))
  write_wav(pred_wav_path, y_pred, 16000)
  pred_spec_path = join(checkpoint_dir, "checkpoint_step_{}_pred_spec.npy".format(global_step))
  np.save(pred_spec_path, spec_pred)


  print("EVAL LENGTH:", length)
  print("EVAL SPEC SHAPE:", spec.shape)
  y = spectrogram2wav_for_ppg_cbhg(spec, gl_power = 1., gl_iter = 100)
  orig_wav_path = join(checkpoint_dir, "checkpoint_step_{}_original.wav".format(global_step))
  write_wav(orig_wav_path, y, 16000)
  orig_spec_path = join(checkpoint_dir, "checkpoint_step_{}_orig_spec.npy".format(global_step))
  np.save(orig_spec_path, spec)
  


def train_dblstm(device, model, data_loader, optimizer, writer, checkpoint_dir):
  # optimize classification
  cross_entropy_loss = nn.CrossEntropyLoss()
  criterion = nn.MSELoss()
  l1_loss = nn.NLLLoss()

  # from kuaishou 
  my_l1_loss = nn.L1Loss()

  global global_step, global_epoch
  while global_epoch < hparams.nepochs:
      running_loss = 0.0
      for step, (ppgs, mels, specs, lengths) in tqdm(enumerate(data_loader)):
          model.train()
          optimizer.zero_grad()

          # print("OUTER PPGS 1:", ppgs.size())
          # print("OUTER MFCCS 1:", mfccs.size())
          # print("OUTER LENGTH:", lengths)

          ppgs = ppgs.to(device)
          mels = mels.to(device)
          specs = specs.to(device)

          sorted_lengths, indices = torch.sort(lengths.view(-1), dim=0, descending=True)
          sorted_lengths = sorted_lengths.long().numpy()
          ppgs, mels, specs = ppgs[indices], mels[indices], specs[indices]
          ppgs, mels, specs = Variable(ppgs).float(), Variable(mels).float(), Variable(specs).float()
          
          # y = y[:, :sorted_lengths[0]]
          # print("OUTER PPGS 2:", ppgs.size())
          # print("OUTER MFCCS 2:", mfccs.size())

          # Apply model
          mels_pred, specs_pred = model(ppgs)

          # print("OUTER MFCCS PRED:", mfccs_pred.size())

          loss = 0.0
          for batch in range(hparams.batch_size):
            # print("batch num:", batch, "batch length:", sorted_lengths[batch])
            mel_loss = my_l1_loss(mels_pred[batch, :sorted_lengths[batch], :], mels[batch, :sorted_lengths[batch], :])
            spec_loss = my_l1_loss(specs_pred[batch, :sorted_lengths[batch], :], specs[batch, :sorted_lengths[batch], :])
            loss += (mel_loss + spec_loss)
          phoneme_size = torch.sum(lengths).to(device).float()
          # loss = loss / phoneme_size
          loss = loss / hparams.batch_size

          print("Check Loss：", loss)

          if global_step > 0 and global_step % hparams.checkpoint_interval == 0:

            print("Save intermediate states at step {}".format(global_step))
            checkpoint_path = os.path.join(checkpoint_dir, "checkpoint_step{:09d}.pth".format(global_step))
            torch.save({
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "global_step": global_step,
                "global_epoch": global_epoch,
            }, checkpoint_path)
            print("Saved checkpoint:", checkpoint_path)

            eval_model_generate(specs[0].cpu().data.numpy(), specs_pred[0].cpu().data.numpy(), sorted_lengths[0], checkpoint_dir, global_step)

          # Update
          loss.backward()
          if hparams.clip_thresh > 0:
              grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hparams.clip_thresh)
          optimizer.step()

          # Logs
          writer.add_scalar("loss", float(loss.item()), global_step)

          running_loss += loss.item()
          global_step += 1

      averaged_loss = running_loss / (len(data_loader))
      writer.add_scalar("loss (per epoch)", averaged_loss, global_epoch)
      global_epoch += 1


if __name__ == '__main__':


   
  os.makedirs(checkpoint_dir, exist_ok=True)

  if checkpoint is not None:
      load_checkpoint(checkpoint_path, model, optimizer, reset_optimizer)

  device = torch.device("cuda" if use_cuda else "cpu")
  # device = torch.cuda.device(2)

  print('CHECK data_root:', data_root)
  print('CHECK hparams.data_root:', hparams.data_root)

  dblstm_dataset = DBlstmDtaset(data_root)
  dblstm_dataloader = DataLoader(dblstm_dataset, batch_size=hparams.batch_size, num_workers=hparams.num_workers, shuffle=True, drop_last=True)

  model = DBLSTM(hparams.batch_size).to(device)
  # model = DBLSTM(hparams.batch_size).cuda(cuda_num)

  optimizer = optim.Adam(model.parameters(), lr=hparams.learning_rate)

  writer = SummaryWriter(log_dir=log_event_path)

  print("Begin to train！")
  train_dblstm(device, model, dblstm_dataloader, optimizer, writer, checkpoint_dir)
  print("Finished")