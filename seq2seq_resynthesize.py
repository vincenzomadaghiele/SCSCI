import torch
from torchvision import transforms
from flow_utils.transforms import LogTransform, NormalizeTensor

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

from VAE_train import Encoder, Decoder, Model


def makeCrownWindow(size):
    crown_window = np.ones(size)
    fade_perc = 0.01 # percentage of fade in/out
    fade_num_samples = int(fade_perc * size)
    fade_in = np.linspace(0,1,fade_num_samples)
    fade_out = np.linspace(1,0,fade_num_samples)
    crown_window[0:fade_num_samples] = fade_in
    crown_window[crown_window.shape[0]-fade_num_samples:] = fade_out
    return crown_window

def melToVAElatents(mels, VAE_model, image_size=64):
    device = torch.device("cpu")
    VAE_model.to(device)
    VAE_transform = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size)])
    prep_event = librosa.power_to_db(mels, ref=np.max)
    prep_event = torch.clip(torch.from_numpy((prep_event+80) / 80), min=0, max=1)
    prep_event = torch.unsqueeze(prep_event, 0).to(torch.float32)
    prep_event = VAE_transform(prep_event)
    prep_event = prep_event.reshape(prep_event.shape[0], image_size*image_size).to(device)
    with torch.no_grad():
        event_latents, _ = VAE_model.Encoder(prep_event)
    return event_latents[0].detach().cpu().numpy()

def melToFlowLatents(mels, flow_model, mean, var, num_MELS=64, melspec_size=80):
    tr = []
    tr.append(LogTransform(clip=1e-3))
    tr.append(NormalizeTensor(mean, var))
    flow_transform = transforms.Compose(tr)
    device = torch.device("cpu")
    flow_model.to(device)
    prep_event = np.zeros((num_MELS, melspec_size)) # pad each event with zeros or cut 
    prep_event[:mels.shape[0], :mels.shape[1]] = mels[:num_MELS,:melspec_size]
    mels = flow_transform(torch.from_numpy(prep_event).float()).to(device)
    with torch.no_grad():
        _, z_tilde, _ = flow_model.ae_model(mels.reshape(1, num_MELS, melspec_size)) # latent vector
        predicted_parameters = flow_model.regression_model(z_tilde) # parameters
    return z_tilde[0].detach().cpu().numpy(), predicted_parameters[0].detach().cpu().numpy()


if __name__ == '__main__': 

    # save models by synth
    SYNTH_NAME = 'TimbreTheremin'

    # LOAD FLOW MODEL
    device = torch.device('cpu')
    model_path = f'./synth/trained_flows/{SYNTH_NAME}/models'
    model_name = 'vae_flow_mel_mse_0_gated_cnn_planar_flow_post_maf_3'
    # model_name = 'vae_flow_mel_mse_0_gated_cnn_maf_flow_post_maf_3'
    model = torch.load(f'{model_path}/{model_name}.model', weights_only=False, map_location=device).to(device)
    model.eval()

    # LOAD AUDIO TRACK
    corpus_name = 'guitar-duo'
    song_name = 'recording5'
    src_track_name = 'guitar1'
    tgt_track_name = 'guitar2'
    filepath = f'./corpus/{corpus_name}/{song_name}/{tgt_track_name}.wav'
    signal, _ = librosa.load(filepath, sr=sr, mono=True)
    events = extractEvents(signal, FFT_window, sr, hop_size, num_MFCC, num_MEL, min_event_duration_samples, max_event_duration_samples)

    # LOAD FLOW MODEL
    num_MELS, melspec_size = 64, 80
    RECORDING_TIME = int(melspec_size * FFT_window / sr)
    synth = Synth(SYNTH_NAME, sr, hop_size, RECORDING_TIME)
    param_names = synth.getInputs()
    N_params = len(param_names)
    data_stats_path = f'./synth/datasets/{SYNTH_NAME}/means.json'
    with open(f'./synth/datasets/{SYNTH_NAME}/means.json') as f:
        d = json.load(f)
        mean = d['mel']
    with open(f'./synth/datasets/{SYNTH_NAME}/vars.json') as f:
        d = json.load(f)
        var = d['mel']

    # THESE ARE THE SAME FOR ALL SYNTHS
    frequency_param_name = "frequency"
    gain_param_name = "gain"
    gate_param_name = "gate"
    envelope_param_names = ['A','D','S','R']
    # THESE CAN CHANGE AMONG SYNTHS
    non_timbre_params_names = [frequency_param_name, gain_param_name, gate_param_name]
    non_timbre_params_names += envelope_param_names
    timbre_param_names = [p for p in param_names if p not in non_timbre_params_names]


    # LOAD VAE
    corpus_path = f'corpus/{corpus_name}'
    VAE_model_name = f'{corpus_path}/trained_VAE/VAE.pth'
    hidden_dim = 128 # load these from pre-saved JSON
    latent_dim = 8
    image_size = 64
    image_size_flat = image_size * image_size
    encoder = Encoder(input_dim=image_size_flat, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = image_size_flat)
    VAE_model = Model(Encoder=encoder, Decoder=decoder).to(device)
    VAE_model.load_state_dict(torch.load(VAE_model_name, weights_only=True))
    VAE_model.eval()

    # COMPUTE LATENTS
    VAE_latents = []
    flow_latents = []
    event_durations = []
    event_starts = []
    original_reconstruct = np.zeros_like(signal)
    resynthesized_reconstruct = np.zeros_like(signal)
    for event in events:
        event_durations.append(event['event_duration'])
        event_starts.append(event['event_start'])
        if event['rms'] != 0:
            mels = event['mel']
            duration = event['event_duration']
            rms = event['rms']
            pitch = event['pitch']
            # COMPUTE VAE LATENTS
            event_latents = melToVAElatents(mels, VAE_model)
            VAE_latents.append(event_latents)
            # COMPUTE FLOW LATENTS
            z_tilde, predicted_parameters = melToFlowLatents(mels, model, mean, var)
            flow_latents.append(z_tilde)
            # RE-SYNTHESIZE SOUND EVENT
            print(duration / sr)
            print(pitch)
            print(rms)
            predicted_parameters = predicted_parameters.tolist()
            synth = Synth(SYNTH_NAME, sr, hop_size, RECORDING_TIME)
            synth.setParameters(timbre_param_names, predicted_parameters)
            synth.setParameters([frequency_param_name], [pitch])
            synth.setParameters([gain_param_name], [float(np.clip(rms*5, 0, 1))])
            synth.setParameters(envelope_param_names, [0.3, 0.3, 0.2, 0.1])
            synthesized_event = synth.forward(duration / sr) 
            synth.close()
            if synthesized_event.shape[0] > event['event_duration']:
                event['resynthesized'] = synthesized_event[:event['event_duration']]
            else:
                newevent = np.zeros(event['event_duration'])
                newevent[:synthesized_event.shape[0]] = synthesized_event
                event['resynthesized'] = newevent
            # RECOMBINE INTO FILE WITH EVENT STARTS
            original_reconstruct[event['event_start']:event['event_start']+event['event_duration']] = signal[event['event_start']:event['event_start']+event['event_duration']]
            resynthesized_reconstruct[event['event_start']:event['event_start']+event['event_duration']] = event['resynthesized'] * makeCrownWindow(event['event_duration'])

        # else:
        #     flow_latents.append('R')
        #     VAE_latents.append('R')

    
    # # SAVE AUDIOFILE
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

    # PLOT PATH ON THE FLOW TIMBRE SPACE
    X = np.load(f'./synth/trained_flows/{SYNTH_NAME}/visualize/{model_name}/latents.npy')
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    pca = PCA(n_components=3)
    X_PCA = pca.fit_transform(X)
    plot_scaler = StandardScaler()
    X_PCA_scaled = plot_scaler.fit_transform(X_PCA)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(X_PCA_scaled[:,0], X_PCA_scaled[:,1], s=1)
    # for path in flow_latents:
    path_latents = np.array(flow_latents)
    if len(path_latents.shape) == 1:
        path_latents = path_latents.reshape(1, -1)
    if path_latents.shape[0] > 0 and path_latents.shape[1] > 0:
        path_latents_scaled = scaler.transform(path_latents)
        path_latents_PCA = pca.transform(path_latents_scaled)
        path_latents_plot = plot_scaler.transform(path_latents_PCA)
        ax.scatter(path_latents_plot[:,0], path_latents_plot[:,1], s=5, c='r')
        plt.plot(path_latents_plot[:,0], path_latents_plot[:,1], color='r', alpha=0.7, linestyle='--')
    ax.set_title("flow latent space")
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    plt.savefig(f'{SAVE_FILEPATH}/2D_PCA_flow.png')

    # PLOT PATH ON THE VAE TIMBRE SPACE
    X = np.load(f'./corpus/{corpus_name}/trained_VAE/latents.npy')
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    pca = PCA(n_components=3)
    X_PCA = pca.fit_transform(X)
    plot_scaler = StandardScaler()
    X_PCA_scaled = plot_scaler.fit_transform(X_PCA)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(X_PCA_scaled[:,0], X_PCA_scaled[:,1], s=1)
    # for path in flow_latents:
    path_latents = np.array(VAE_latents)
    if len(path_latents.shape) == 1:
        path_latents = path_latents.reshape(1, -1)
    if path_latents.shape[0] > 0 and path_latents.shape[1] > 0:
        path_latents_scaled = scaler.transform(path_latents)
        path_latents_PCA = pca.transform(path_latents_scaled)
        path_latents_plot = plot_scaler.transform(path_latents_PCA)
        ax.scatter(path_latents_plot[:,0], path_latents_plot[:,1], s=5, c='r')
        plt.plot(path_latents_plot[:,0], path_latents_plot[:,1], color='r', alpha=0.7, linestyle='--')
    ax.set_title("VAE latent space")
    ax.set_xlabel('PCA 1')
    ax.set_ylabel('PCA 2')
    plt.savefig(f'{SAVE_FILEPATH}/2D_PCA_VAE.png')
