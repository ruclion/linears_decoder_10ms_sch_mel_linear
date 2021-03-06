import os
import numpy as np
from audio import hparams as audio_hparams
from audio import load_wav, wav2unnormalized_mfcc, wav2normalized_db_mel, wav2normalized_db_spec


# 超参数个数：16
hparams = {
    'sample_rate': 16000,
    'preemphasis': 0.97,
    'n_fft': 400,
    'hop_length': 160,
    'win_length': 400,
    'num_mels': 80,
    'n_mfcc': 13,
    'window': 'hann',
    'fmin': 30.,
    'fmax': 7600.,
    'ref_db': 20,  
    'min_db': -80.0,  
    'griffin_lim_power': 1.5,
    'griffin_lim_iterations': 60,  
    'silence_db': -28.0,
    'center': False,
}

assert hparams == audio_hparams


wav_dir = 'wavs_16000_960'
mfcc_dir = 'MFCCs'
mel_dir = 'MELs'
spec_dir = 'SPECs'

def main():
    #这一部分用于处理LibriSpeech格式的数据集。
    for first_dir in os.listdir(wav_dir):
        for second_dir in os.listdir(os.path.join(wav_dir, first_dir)):
            for third_dir in os.listdir(os.path.join(os.path.join(wav_dir,first_dir), second_dir)):
                third_mfcc_dir = os.path.join(os.path.join(os.path.join(mfcc_dir,first_dir),second_dir), third_dir)
                third_mel_dir = os.path.join(os.path.join(os.path.join(mel_dir,first_dir),second_dir), third_dir)
                third_spec_dir = os.path.join(os.path.join(os.path.join(spec_dir,first_dir),second_dir), third_dir)
                third_wav_dir = os.path.join(os.path.join(os.path.join(wav_dir,first_dir),second_dir), third_dir)
                #print('Now in the '+mfcc_dir+' from '+ third_wav_dir)
                if not os.path.exists(third_mfcc_dir):
                    os.makedirs(third_mfcc_dir)

                wav_files = [os.path.join(third_wav_dir, f) for f in os.listdir(third_wav_dir) if f.endswith('.wav')]
                print('Extracting MFCC from {} to {}...'.format(third_wav_dir, third_mfcc_dir))
                cnt = 0
                for wav_f in wav_files:
                    wav_arr = load_wav(wav_f, sr=hparams['sample_rate'])
                    mfcc_feats = wav2unnormalized_mfcc(wav_arr)
                    mel_feats = wav2normalized_db_mel(wav_arr)
                    spec_feats = wav2normalized_db_spec(wav_arr)

                    save_name = wav_f.split('/')[-1].split('.')[0] + '.npy'
                    mfcc_save_name = os.path.join(third_mfcc_dir, save_name)
                    mel_save_name = os.path.join(third_mel_dir, save_name)
                    spec_save_name = os.path.join(third_spec_dir, save_name)
                    np.save(mfcc_save_name, mfcc_feats)
                    np.save(mel_save_name, mel_feats)
                    np.save(spec_save_name, spec_feats)
                    cnt += 1
                    print(cnt)
        #             break
        #         break
        #     break
        # break
        # 提取完毕以后，需要手动将3个文件夹的东西mv到同一个，和ppg一样的2338个文件夹
    return


if __name__ == '__main__':
    main()
