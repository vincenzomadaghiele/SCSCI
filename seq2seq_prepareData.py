import os
import json
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt

import torch
from torchvision import transforms
from utils.transforms import LogTransform, NormalizeTensor

from CONSTANTS import * 
from VAE_train import Encoder, Decoder, Model

from synth.synths import Synth
from resynthesize import melToVAElatents, melToFlowLatents, makeCrownWindow

if __name__ == '__main__': 

    DEVICE = torch.device("cpu")

    # LOAD EVENT SEQUENCES CORPUS
    corpus_name = 'guitar-duo'
    src_track_name = 'guitar1'
    tgt_track_name = 'guitar2'
    corpus_path = f'corpus/{corpus_name}'
    song_names = os.listdir(corpus_path)
    src_tgt = []
    song_names = [c for c in song_names if os.path.isdir(f'{corpus_path}/{c}') and c not in ['info','trained_VAE','.DS_Store','src_full','finetune']]
    song_names.sort()
    src_corpus_files = [f'{corpus_path}/{song_name}/mel/{src_track_name}.npy' for song_name in song_names]
    tgt_corpus_files = [f'{corpus_path}/{song_name}/mel/{tgt_track_name}.npy' for song_name in song_names]

    # LOAD VAE
    VAE_model_name = f'{corpus_path}/trained_VAE/VAE.pth'
    hidden_dim = 128 # load these from pre-saved JSON
    latent_dim = 8
    image_size = 64
    image_size_flat = image_size * image_size
    encoder = Encoder(input_dim=image_size_flat, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = image_size_flat)
    VAE_model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)
    VAE_model.load_state_dict(torch.load(VAE_model_name, weights_only=True))
    VAE_model.eval()


    SYNTH_NAME = 'TimbreTheremin'
    num_MELS, melspec_size = 64, 80
    RECORDING_TIME = int(melspec_size * FFT_window / sr)
    synth = Synth(SYNTH_NAME, sr, hop_size, RECORDING_TIME)
    param_names = synth.getInputs()

    # THESE ARE THE SAME FOR ALL SYNTHS
    frequency_param_name = "frequency"
    gain_param_name = "gain"
    gate_param_name = "gate"
    envelope_param_names = ['A','D','S','R']
    # THESE CAN CHANGE AMONG SYNTHS
    non_timbre_params_names = [frequency_param_name, gain_param_name, gate_param_name]
    non_timbre_params_names += envelope_param_names
    timbre_param_names = [p for p in param_names if p not in non_timbre_params_names]


    # LOAD FLOW MODEL
    SYNTH_NAME = 'TimbreTheremin'
    model_path = f'./synth/trained_flows/{SYNTH_NAME}/models'
    model_name = 'vae_flow_mel_mse_0_gated_cnn_planar_flow_post_maf_3'
    flow_model = torch.load(f'{model_path}/{model_name}.model', weights_only=False, map_location=DEVICE).to(DEVICE)
    flow_model.eval()
    # LOAD DATASET STATS FOR NORMALIZATION
    data_stats_path = f'./synth/datasets/{SYNTH_NAME}/means.json'
    with open(f'./synth/datasets/{SYNTH_NAME}/means.json') as f:
        d = json.load(f)
        mean = d['mel']
    with open(f'./synth/datasets/{SYNTH_NAME}/vars.json') as f:
        d = json.load(f)
        var = d['mel']
    N_latents_tgt = 7 # retreive this from flow train JSON

    with open(f"synth/params/{SYNTH_NAME}.txt", "r") as file:
        params = file.readlines()
    N_latents_tgt = len(params)

    # COMPUTE LATENTS
    # compute src latents using VAE
    # transform = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size)])
    src_events = []
    for song in src_corpus_files:
        event_sequence = np.load(song, allow_pickle=True)
        for event in event_sequence:
            # detect rest
            if event['rms'] != 0:
                # COMPUTE VAE LATENTS
                event_latents = melToVAElatents(event['mel'], VAE_model)
                event_latents = event_latents.tolist()
            else:
                event_latents = (np.zeros(latent_dim)).tolist()
            event_latents.append(event['rms'])
            event_latents.append(event['pitch'])
            event_latents.append(event['event_duration'])
            event_latents.append(event['event_start'])
            event_latents.append(song)
            src_events.append(event_latents)
    src_events = np.array(src_events)
    print(src_events.shape)
    columns = [f'latent-{j}' for j in range(latent_dim)]
    columns.append('rms')
    columns.append('pitch')
    columns.append('event_duration')
    columns.append('event_start')
    columns.append('filename')
    df_src = pd.DataFrame(src_events, columns=columns)
    print(df_src.head())
    df_src.to_csv(f"{corpus_path}/src_events.csv")



    # compute tgt latents using flow
    # tr = []
    # tr.append(LogTransform(clip=1e-3))
    # tr.append(NormalizeTensor(mean, var))
    # transform = transforms.Compose(tr)

    # encode all mels and find range
    # if range of latents is smaller than total range of latents
    # scale latents to cover the whole space
    tgt_events = []
    for song in tgt_corpus_files:

        song_name = song.split('/')[-3]
        print(song_name)
        signal, _ = librosa.load(f"{corpus_path}/{song_name}/{tgt_track_name}.wav", sr=sr, mono=True)
        original_reconstruct = np.zeros_like(signal)
        resynthesized_reconstruct = np.zeros_like(signal)

        event_sequence = np.load(song, allow_pickle=True)
        for event in event_sequence:
            # detect rest
            if event['rms'] != 0:
                event_latents, _ = melToFlowLatents(event['mel'], flow_model, mean, var)
                event_latents = event_latents.tolist()
            else:
                event_latents = (np.zeros(N_latents_tgt)).tolist()
            event_latents.append(event['rms'])
            event_latents.append(event['pitch'])
            event_latents.append(event['event_duration'])
            event_latents.append(event['event_start'])
            event_latents.append(song)
            tgt_events.append(event_latents)

            # RESYNTHESIZE
            # for event in tgt_events:
            if event['rms'] != 0:
                # find min max latents of flow model (from training data)
                # find min max of resynthesized latents
                # rescale resynthezied latents and generate sound from scaled latents params
                z_tilde, predicted_parameters = melToFlowLatents(event['mel'], flow_model, mean, var)
                synth.close()
                synth = Synth(SYNTH_NAME, sr, hop_size, RECORDING_TIME)
                synth.setParameters(timbre_param_names, predicted_parameters.tolist())
                synth.setParameters([frequency_param_name], [event['pitch']])
                synth.setParameters([gain_param_name], [float(np.clip(event['rms']*5, 0, 1))])
                synth.setParameters(envelope_param_names, [0.3, 0.3, 0.2, 0.1])
                synthesized_event = synth.forward(int(event['event_duration'] - (event['event_duration']/2)) / sr) 
                if synthesized_event.shape[0] > event['event_duration']:
                    event['resynthesized'] = synthesized_event[:event['event_duration']]
                else:
                    newevent = np.zeros(event['event_duration'])
                    newevent[:synthesized_event.shape[0]] = synthesized_event
                    event['resynthesized'] = newevent
                # RECOMBINE INTO FILE WITH EVENT STARTS
                original_reconstruct[event['event_start']:event['event_start']+event['event_duration']] = signal[event['event_start']:event['event_start']+event['event_duration']]
                resynthesized_reconstruct[event['event_start']:event['event_start']+event['event_duration']] = event['resynthesized'] * makeCrownWindow(event['event_duration'])

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
        plt.close(fig)

    tgt_events = np.array(tgt_events)
    print(tgt_events.shape)
    columns = [f'latent-{j}' for j in range(N_latents_tgt)]
    columns.append('rms')
    columns.append('pitch')
    columns.append('event_duration')
    columns.append('event_start')
    columns.append('filename')
    df_tgt = pd.DataFrame(tgt_events, columns=columns)
    print(df_tgt.head())
    df_tgt.to_csv(f"{corpus_path}/tgt_events.csv")

    # if finetune
    # add finetune to dfs

    finetune_path = f'{corpus_path}/finetune'
    song_names = os.listdir(finetune_path)
    song_names = [c for c in song_names if os.path.isdir(f'{finetune_path}/{c}') and c not in ['.DS_Store', '__pycache__']]
    for song in song_names:
        fintune_src = pd.read_csv(f"{finetune_path}/{song}/src_events.csv", index_col=0)
        fintune_src['filename'] = [f'{finetune_path}/{song}/performance_src.wav' for _ in range(fintune_src.values.shape[0])]
        fintune_tgt = pd.read_csv(f"{finetune_path}/{song}/tgt_events.csv", index_col=0)
        fintune_tgt['filename'] = [f'{finetune_path}/{song}/performance_tgt.wav' for _ in range(fintune_tgt.values.shape[0])]
        df_src = pd.concat([df_src, fintune_src])
        df_tgt = pd.concat([df_tgt, fintune_tgt])
        # df_src = df_src.reset_index()
        # df_tgt = df_tgt.reset_index()
    df_src.to_csv(f"{corpus_path}/src_events.csv")
    df_tgt.to_csv(f"{corpus_path}/tgt_events.csv")
