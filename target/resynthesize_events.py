import torch
from torchvision import transforms
from utils.transforms import LogTransform, NormalizeTensor

import numpy as np
import librosa

from CONSTANTS import *
from corpus.extract_events import adjustAmplitude, detectOnsets, computeEventFeatures, detectRests, plotEvents

if __name__ == '__main__': 

    # LOAD FLOW MODEL
    device = torch.device('cpu')
    model_path = './outputs/models'
    model_name = 'vae_flow_mel_mse_0_gated_cnn_planar_flow_post_maf_3'
    model = torch.load(f'{model_path}/{model_name}.model', weights_only=False, map_location=device).to(device)
    model.eval()
    # pp = torch.from_numpy(np.array(synth_parameters)[:-1] + 0.00001).float().view(1,-1).to(device)
    # current_latents = model.regression_model.flow.inverse(pp)[0].detach().numpy()[0]
    # print(current_latents.shape)

    # LOAD AUDIO TRACK
    filepath = 'corpus/guitar1.wav'
    signal, _ = librosa.load(filepath, sr=sr, mono=True)
    signal = adjustAmplitude(signal, FFT_window, hop_size)
    print(f'Loading audiofile {filepath} with length {signal.shape[0]} samples')

    # SEGMENT INTO EVENTS
    onsets = detectOnsets( signal, sr, min_event_duration_samples=min_event_duration_samples )
    print(f'detected {len(onsets)} onsets...')
    events = computeEventFeatures( signal, onsets, sr, FFT_window, hop_size, mfcc_N=num_MFCC, mels_N=num_MEL, compute_rests=True )
    last_sample_n = signal.shape[0]-1
    events = detectRests( events, num_MEL, min_event_duration_samples, max_event_duration_samples, FFT_window, hop_size, last_sample_n )
    # this_file_event = {}
    # this_file_event['filename'] = filepath
    # this_file_event['events'] = events
    # plotEvents(this_file_event, sr)

    # RESYNTHESIZE EACH EVENT
    # melspectrogram --> flow --> params --> synth --> 
    # pad each event with zeros or cut 
    mean, var = -6.77166223526001, 1.205236792564392 # load this info from dataset
    tr = []
    tr.append(LogTransform(clip=1e-3))
    tr.append(NormalizeTensor(mean, var))
    transform = transforms.Compose(tr)
    num_MELS, melspec_size = 64, 80
    for event in events:
        prep_event = np.zeros((num_MELS, melspec_size))
        prep_event[:event['mel'].shape[0], :event['mel'].shape[1]] = event['mel'][:num_MELS,:melspec_size]
        event['mel'] = transform(torch.from_numpy(prep_event).float()).to(device)
        _, z_tilde, _ = model.ae_model(event['mel'].reshape(1, num_MELS, melspec_size)) # latent vector
        predicted_parameters = model.regression_model(z_tilde) # parameters
        print(predicted_parameters)
        # use paramters to resynthesize
        # call synth and generate output with params and event['duration'] 



    # RECOMBINE INTO FILE WITH EVENT STARTS

    # SAVE AUDIOFILE

