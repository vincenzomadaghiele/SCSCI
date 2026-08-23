import os
import copy
import librosa
from torch.utils.data import DataLoader
from flow_utils.data import AudioDataset

# make npz file with file['chars'] = None, file['params'] = params, file['audio'] = audio
# (npz files in ./00_dataset/raw/*.npz)
# data is a spectral dataset encoded by AudioDataset (npy files in ./00_dataset/mfcc/*.npy)
# make new CompSynthesizerDataset and use it to train model
# (maybe make dict with parameter names ?) --> we'll see if needed for or after training

if __name__ == '__main__': 

    SYNTH_NAME = 'Benjolin'
    dataset_dir = f'./synth/datasets/{SYNTH_NAME}'
    dset = AudioDataset(dataset_dir, paramsdir=f'./synth/params/{SYNTH_NAME}.txt', 
                        data='mel_mfcc', mean={'mfcc':0.0,'mel':0.0}, var={'mfcc':1.0, 'mel':1.0})
    loader = DataLoader(dset, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
    for i, (spec, pars, wav) in enumerate(loader):
        print(f'--- extracting data {i} ---')
        print(spec.shape)
        print(wav.shape)
