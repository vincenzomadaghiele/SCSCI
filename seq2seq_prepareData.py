import os
import json
import torch
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt

from CONSTANTS import * 
from VAE_train import Encoder, Decoder, Model

from seq2seq_resynthesize import resynthesizeFlowVAE, plotDualWaveform, plotFlowTimbre, plotVAETimbre, melToVAElatents, melToFlowLatents

if __name__ == '__main__': 

    DEVICE = torch.device("cpu")

    # LOAD EVENT SEQUENCES CORPUS
    corpus_name = 'andrea-study'
    src_track_name = 'andrea-src'
    tgt_track_name = 'andrea-tgt'
    corpus_path = f'corpus/{corpus_name}'
    song_names = os.listdir(corpus_path)
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

    # LOAD FLOW MODEL
    SYNTH_NAME = 'FM'
    model_path = f'./synth/trained_flows/{SYNTH_NAME}/models'
    model_name = 'vae_flow_mel_mse_0_gated_cnn_planar_flow_post_maf_3'
    flow_model = torch.load(f'{model_path}/{model_name}.model', weights_only=False, map_location=DEVICE).to(DEVICE)
    flow_model.eval()
    # LOAD DATASET STATS FOR NORMALIZATION
    data_stats_path = f'./synth/datasets/{SYNTH_NAME}/means.json'
    with open(f'./synth/datasets/{SYNTH_NAME}/means.json') as f:
        d = json.load(f)
        flow_mean = d['mel']
    with open(f'./synth/datasets/{SYNTH_NAME}/vars.json') as f:
        d = json.load(f)
        flow_var = d['mel']
    with open(f"synth/params/{SYNTH_NAME}.txt", "r") as file:
        params = file.readlines()
    N_latents_tgt = len(params)

    # COMPUTE LATENTS
    src_events = []
    for song in src_corpus_files:
        event_sequence = np.load(song, allow_pickle=True)
        for event in event_sequence:
            if event['rms'] != 0:
                event_latents = melToVAElatents(event['mel'], VAE_model)
                event_latents = event_latents.tolist()
            else:
                event_latents = (np.zeros(latent_dim)).tolist()
            event_latents += [event['rms'], event['pitch'], event['event_duration'], event['event_start'], song]
            src_events.append(event_latents)
    src_events = np.array(src_events)
    columns = [f'latent-{j}' for j in range(latent_dim)]
    columns += ['rms', 'pitch', 'event_duration', 'event_start', 'filename']
    df_src = pd.DataFrame(src_events, columns=columns)
    df_src.to_csv(f"{corpus_path}/src_events.csv")

    # COMPUTE MEAN OF FLOW LATENT SEQUENCES
    latents = []
    for song in tgt_corpus_files:
        event_sequence = np.load(song, allow_pickle=True)
        for event in event_sequence:
            if event['rms'] != 0:
                z_tilde, predicted_parameters = melToFlowLatents(event['mel'], flow_model, flow_mean, flow_var)
                latents.append(z_tilde)
    latents = np.array(latents)
    latents_mean = latents.mean(axis=0)
    latents_std = latents.std(axis=0)
    total_latents = np.load(f'./synth/trained_flows/{SYNTH_NAME}/visualize/{model_name}/latents.npy')
    total_latents_mean = total_latents.mean(axis=0)
    total_latents_std = total_latents.std(axis=0)
    latents_max = total_latents.max(axis=0)
    latents_min = total_latents.min(axis=0)

    tgt_events = []
    timbreRescale = True # rescale timbre for resynthesis
    for song in tgt_corpus_files:
        event_sequence = np.load(song, allow_pickle=True)
        for event in event_sequence:
            if event['rms'] != 0:
                event_latents, _ = melToFlowLatents(event['mel'], flow_model, flow_mean, flow_var)
                if timbreRescale:
                    event_latents = (event_latents - latents_mean) * (total_latents_std / latents_std) + total_latents_mean
                    event_latents = np.clip(event_latents, latents_min, latents_max)
                event_latents = event_latents.tolist()
            else:
                event_latents = (np.zeros(N_latents_tgt)).tolist()
            event_latents += [event['rms'], event['pitch'], event['event_duration'], event['event_start'], song]
            tgt_events.append(event_latents)

    tgt_events = np.array(tgt_events)
    columns = [f'latent-{j}' for j in range(N_latents_tgt)]
    columns += ['rms', 'pitch', 'event_duration', 'event_start', 'filename']
    df_tgt = pd.DataFrame(tgt_events, columns=columns)
    print(df_tgt.head())
    df_tgt.to_csv(f"{corpus_path}/tgt_events_{SYNTH_NAME}.csv")

    # resynthesize target files
    for song in tgt_corpus_files:
        song_name = song.split('/')[-3]
        signal, _ = librosa.load(f"{corpus_path}/{song_name}/{tgt_track_name}.wav", sr=sr, mono=True)
        # RESYNTHESIZE FILE
        filepath = f'./corpus/{corpus_name}/{song_name}/{tgt_track_name}.wav'
        signal, _ = librosa.load(filepath, sr=sr, mono=True)
        flow_latents, VAE_latents, original_reconstruct, resynthesized_reconstruct = resynthesizeFlowVAE(signal, flow_model, VAE_model, flow_mean, flow_var, SYNTH_NAME, flow_model_name=model_name,
                                                                                                         latents_mean=latents_mean, latents_std=latents_std, 
                                                                                                         total_latents_mean=total_latents_mean, total_latents_std=total_latents_std, 
                                                                                                         latents_min=latents_min, latents_max=latents_max
                                                                                                         )
        # SAVE AUDIOFILE
        SAVE_FILEPATH = f'./corpus/{corpus_name}/{song_name}/resynthesized_{tgt_track_name}_{SYNTH_NAME}'
        os.makedirs(SAVE_FILEPATH, exist_ok=True)
        sf.write(f'{SAVE_FILEPATH}/{tgt_track_name}_{SYNTH_NAME}_resynthesized.wav', resynthesized_reconstruct, sr, subtype='PCM_24')
        # MAKE DUAL FIGURE
        plotDualWaveform(original_reconstruct, resynthesized_reconstruct, SAVE_FILEPATH)
        X = np.load(f'./synth/trained_flows/{SYNTH_NAME}/visualize/{model_name}/latents.npy')
        plotFlowTimbre(X, flow_latents, SAVE_FILEPATH)
        # PLOT PATH ON THE VAE TIMBRE SPACE
        X = np.load(f'./corpus/{corpus_name}/trained_VAE/latents.npy')
        plotVAETimbre(X, VAE_latents, SAVE_FILEPATH)

    # IF FINETUNE
    includeFinetune = False
    if includeFinetune:
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
        df_src.to_csv(f"{corpus_path}/src_events.csv")
        df_tgt.to_csv(f"{corpus_path}/tgt_events.csv")
