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
    SYNTH_NAME = 'Theremin'
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
    N_latents_tgt = 5 # retreive this from flow train JSON


    # COMPUTE LATENTS
    # compute src latents using VAE
    transform = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size)])
    src_events = []
    for song in src_corpus_files:
        event_sequence = np.load(song, allow_pickle=True)
        for event in event_sequence:
            # detect rest
            if event['mel'].mean() != 0 and event['mel'].shape[1] > 0:
                prep_event = librosa.power_to_db(event['mel'], ref=np.max)
                prep_event = torch.clip(torch.from_numpy((prep_event+80) / 80), min=0, max=1)
                prep_event = torch.unsqueeze(prep_event, 0).to(torch.float32)
                prep_event = transform(prep_event)
                prep_event = prep_event.reshape(prep_event.shape[0], image_size_flat).to(DEVICE)
                with torch.no_grad():
                    event_latents, _ = VAE_model.Encoder(prep_event)
                event_latents = event_latents[0].detach().cpu().numpy().tolist()
            else:
                # event_latents = np.zeros(latent_dim).tolist()
                event_latents = (np.zeros(latent_dim)-6).tolist()

                # ## REMOVE IF DOESN'T WORK ##
                # prep_event = librosa.power_to_db(np.zeros((64,64)), ref=np.max)
                # prep_event = torch.clip(torch.from_numpy((prep_event+80) / 80), min=0, max=1)
                # prep_event = torch.unsqueeze(prep_event, 0).to(torch.float32)
                # prep_event = transform(prep_event)
                # prep_event = prep_event.reshape(prep_event.shape[0], image_size_flat).to(DEVICE)
                # with torch.no_grad():
                #     event_latents, _ = VAE_model.Encoder(prep_event)
                # event_latents = event_latents[0].detach().cpu().numpy().tolist()
                # ## REMOVE IF DOESN'T WORK ##

            event_latents.append(event['event_duration'])
            event_latents.append(event['event_start'])
            event_latents.append(song)
            src_events.append(event_latents)
    src_events = np.array(src_events)
    print(src_events.shape)
    columns = [f'latent-{j}' for j in range(src_events.shape[1]-3)]
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
            if event['mel'].mean() != 0:
                # pad each event with zeros or cut 
                prep_event = np.zeros((num_MEL, melspec_size_flow))
                prep_event[:event['mel'].shape[0], :event['mel'].shape[1]] = event['mel'][:num_MEL,:melspec_size_flow]
                event['mel'] = transform(torch.from_numpy(prep_event).float()).to(DEVICE)
                with torch.no_grad():
                    _, z_tilde, _ = flow_model.ae_model(event['mel'].reshape(1, num_MEL, melspec_size_flow)) # latent vector
                    # predicted_parameters = model.regression_model(z_tilde) # parameters
                event_latents = z_tilde[0].detach().cpu().numpy().tolist() # seq2seq learns from this
            else:
                # event_latents = np.zeros(N_latents_tgt).tolist()
                event_latents = (np.zeros(N_latents_tgt)-6).tolist()

                # ## REMOVE IF DOESN'T WORK ##
                # prep_event = np.zeros((num_MEL, melspec_size_flow))
                # event['mel'] = transform(torch.from_numpy(prep_event).float()).to(DEVICE)
                # with torch.no_grad():
                #     _, z_tilde, _ = flow_model.ae_model(event['mel'].reshape(1, num_MEL, melspec_size_flow)) # latent vector
                #     # predicted_parameters = model.regression_model(z_tilde) # parameters
                # event_latents = z_tilde[0].detach().cpu().numpy().tolist() # seq2seq learns from this
                # ## REMOVE IF DOESN'T WORK ##

            event_latents.append(event['event_duration'])
            event_latents.append(event['event_start'])
            event_latents.append(song)
            tgt_events.append(event_latents)
    tgt_events = np.array(tgt_events)
    print(tgt_events.shape)
    columns = [f'latent-{j}' for j in range(tgt_events.shape[1]-3)]
    columns.append('event_duration')
    columns.append('event_start')
    columns.append('filename')
    df_tgt = pd.DataFrame(tgt_events, columns=columns)
    print(df_tgt.head())
    df_tgt.to_csv(f"{corpus_path}/tgt_events.csv")

