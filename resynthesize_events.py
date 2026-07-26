import torch
from torchvision import transforms
from utils.transforms import LogTransform, NormalizeTensor

import os
import json
import librosa
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf

import json
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from CONSTANTS import *
from corpus.extract_events import extractEvents
from synth.synths import Synth

def makeCrownWindow(size):
    crown_window = np.ones(size)
    fade_perc = 0.01 # percentage of fade in/out
    fade_num_samples = int(fade_perc * size)
    fade_in = np.linspace(0,1,fade_num_samples)
    fade_out = np.linspace(1,0,fade_num_samples)
    crown_window[0:fade_num_samples] = fade_in
    crown_window[crown_window.shape[0]-fade_num_samples:] = fade_out
    return crown_window

if __name__ == '__main__': 

    # save models by synth
    SYNTH_NAME = 'Theremin'

    # LOAD FLOW MODEL
    device = torch.device('cpu')
    model_path = f'./synth/trained_flows/{SYNTH_NAME}/models'
    model_name = 'vae_flow_mel_mse_0_gated_cnn_planar_flow_post_maf_3'
    model = torch.load(f'{model_path}/{model_name}.model', weights_only=False, map_location=device).to(device)
    model.eval()

    # LOAD AUDIO TRACK
    corpus_name = 'guitar-duo'
    song_name = 'recording3'
    src_track_name = 'guitar1'
    tgt_track_name = 'guitar2'
    filepath = f'./corpus/{corpus_name}/{song_name}/{tgt_track_name}.wav'
    signal, _ = librosa.load(filepath, sr=sr, mono=True)
    # SEGMENT INTO EVENTS
    events = extractEvents(signal, FFT_window, sr, hop_size, num_MFCC, num_MEL, min_event_duration_samples, max_event_duration_samples)
    # this_file_event = {}
    # this_file_event['filename'] = filepath
    # this_file_event['events'] = events
    # plotEvents(this_file_event, sr)

    # LOAD DATASET STATS FOR NORMALIZATION
    data_stats_path = f'./synth/datasets/{SYNTH_NAME}/means.json'
    with open(f'./synth/datasets/{SYNTH_NAME}/means.json') as f:
        d = json.load(f)
        mean = d['mel']
    with open(f'./synth/datasets/{SYNTH_NAME}/vars.json') as f:
        d = json.load(f)
        var = d['mel']

    tr = []
    tr.append(LogTransform(clip=1e-3))
    tr.append(NormalizeTensor(mean, var))
    transform = transforms.Compose(tr)
    num_MELS, melspec_size = 64, 80
    RECORDING_TIME = int(melspec_size * FFT_window / sr)
    synth = Synth(SYNTH_NAME, sr, hop_size, RECORDING_TIME)
    param_names = synth.getInputs()
    N_params = len(param_names)

    # RESYNTHESIZE EACH EVENT
    # melspectrogram --> flow --> params --> synth --> sound
    paths = []
    path = []
    original_reconstruct = np.zeros_like(signal)
    resynthesized_reconstruct = np.zeros_like(signal)
    for event in events:
        # detect rest
        if event['mel'].mean() != 0:
            # pad each event with zeros or cut 
            prep_event = np.zeros((num_MELS, melspec_size))
            prep_event[:event['mel'].shape[0], :event['mel'].shape[1]] = event['mel'][:num_MELS,:melspec_size]
            event['mel'] = transform(torch.from_numpy(prep_event).float()).to(device)
            with torch.no_grad():
                _, z_tilde, _ = model.ae_model(event['mel'].reshape(1, num_MELS, melspec_size)) # latent vector
                predicted_parameters = model.regression_model(z_tilde) # parameters
                event['latents'] = z_tilde[0].detach().cpu().numpy() # seq2seq learns from this
            event['pred_params'] = predicted_parameters[0].detach().cpu().numpy()
            # use paramters to resynthesize
            synth.setParameters(param_names, event['pred_params'].tolist())
            synthesized_event = synth.forward() # add possibility of 'note off' to handle different durations
            if synthesized_event.shape[0] > event['event_duration']:
                event['resynthesized'] = synthesized_event[:event['event_duration']]
            else:
                newevent = np.zeros(event['event_duration'])
                newevent[:synthesized_event.shape[0]] = synthesized_event
                event['resynthesized'] = newevent
            # RECOMBINE INTO FILE WITH EVENT STARTS
            original_reconstruct[event['event_start']:event['event_start']+event['event_duration']] = signal[event['event_start']:event['event_start']+event['event_duration']]
            resynthesized_reconstruct[event['event_start']:event['event_start']+event['event_duration']] = event['resynthesized'] * makeCrownWindow(event['event_duration'])
            path.append(event['latents'])
        else:
            # event['latents'] = np.zeros(5)
            paths.append(path)
            path = []
            original_reconstruct[event['event_start']:event['event_start']+event['event_duration']] = np.zeros(event['event_duration'])
            resynthesized_reconstruct[event['event_start']:event['event_start']+event['event_duration']] = np.zeros(event['event_duration'])
    
    # SAVE AUDIOFILE
    SAVE_FILEPATH = f'./corpus/{corpus_name}/{song_name}/resynthesized'
    os.makedirs(SAVE_FILEPATH, exist_ok=True)
    sf.write(f'{SAVE_FILEPATH}/{tgt_track_name}_{SYNTH_NAME}_resynthesized.wav', resynthesized_reconstruct, sr, subtype='PCM_24')

    # MAKE DUAL FIGURE
    fig, ax = plt.subplots(2, figsize=(16, 6))
    librosa.display.waveshow(y=original_reconstruct, sr=sr, color="black", axis='time', ax=ax[0], alpha=0.2)
    librosa.display.waveshow(y=resynthesized_reconstruct, sr=sr, color="black", axis='time', ax=ax[1], alpha=0.2)
    ax[0].set_title('original')
    ax[1].set_title('resynthesized')
    ax[0].set_xticks([])
    ax[0].set_xlabel('')
    ax[1].set_xlabel('time')
    ax[0].set_ylim([-1,1])
    ax[1].set_ylim([-1,1])
    fig.savefig(f'{SAVE_FILEPATH}/{tgt_track_name}_{SYNTH_NAME}_resynthesized.png')

    # PLOT PATH ON THE TIMBRE SPACE
    X = np.load(f'./synth/trained_flows/{SYNTH_NAME}/visualize/{model_name}/latents.npy')
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    pca = PCA(n_components=3)
    X_PCA = pca.fit_transform(X)
    plot_scaler = StandardScaler()
    X_PCA_scaled = plot_scaler.fit_transform(X_PCA)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(X_PCA_scaled[:,0], X_PCA_scaled[:,1], s=1)
    for path in paths:
        path_latents = np.array(path)
        if len(path_latents.shape) == 1:
            path_latents = path_latents.reshape(1, -1)
        if path_latents.shape[0] > 0 and path_latents.shape[1] > 0:
            path_latents_scaled = scaler.transform(path_latents)
            path_latents_PCA = pca.transform(path_latents_scaled)
            path_latents_plot = plot_scaler.transform(path_latents_PCA)
            ax.scatter(path_latents_plot[:,0], path_latents_plot[:,1], s=5, c='r')
            plt.plot(path_latents_plot[:,0], path_latents_plot[:,1], color='r', alpha=0.7, linestyle='--')
    ax.set_title("latent space")
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    # plt.show()
    plt.savefig(f'{SAVE_FILEPATH}/2D_PCA.png')

    # save pre-processed event duos (w. src-latents, tgt-latents + models used)
    
    # better multi-voice synth with duration envelope
