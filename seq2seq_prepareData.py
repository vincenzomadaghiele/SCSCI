import os
import json
import librosa
import numpy as np
import pandas as pd

import torch
from torchvision import transforms
from utils.transforms import LogTransform, NormalizeTensor

from CONSTANTS import * 
from VAE_train import Encoder, Decoder, Model

from resynthesize import melToVAElatents, melToFlowLatents

if __name__ == '__main__': 

    DEVICE = torch.device("cpu")

    # LOAD EVENT SEQUENCES CORPUS
    corpus_name = 'guitar-duo'
    src_track_name = 'guitar1'
    tgt_track_name = 'guitar2'
    corpus_path = f'corpus/{corpus_name}'
    song_names = os.listdir(corpus_path)
    src_tgt = []
    song_names = [c for c in song_names if os.path.isdir(f'{corpus_path}/{c}') and c not in ['info','trained_VAE','.DS_Store','src_full']]
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
    transform = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size)])
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
    tr = []
    tr.append(LogTransform(clip=1e-3))
    tr.append(NormalizeTensor(mean, var))
    transform = transforms.Compose(tr)
    tgt_events = []
    for song in tgt_corpus_files:
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

