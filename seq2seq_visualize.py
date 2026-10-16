import torch
import librosa
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from CONSTANTS import * 

from scipy.ndimage.filters import gaussian_filter
from corpus.extract_events import *

from seq2seq_resynthesize import resynthesizeFlowVAE, plotDualWaveform, plotFlowTimbre, plotVAETimbre, melToVAElatents, melToFlowLatents
from VAE_train import Encoder, Decoder, Model

if __name__ == '__main__': 

    # LOAD VAE TIMBRE SPACE
    corpus_name = 'andrea-study-plus'
    X = np.load(f'./corpus/{corpus_name}/trained_VAE/latents.npy')
    VAE_scaler = StandardScaler()
    X = VAE_scaler.fit_transform(X)
    pca_VAE = PCA(n_components=3)
    X_PCA = pca_VAE.fit_transform(X)
    plot_scaler_VAE = StandardScaler()
    X_PCA_VAE = plot_scaler_VAE.fit_transform(X_PCA)

    # LOAD FLOW TIMBRE SPACE
    SYNTH_NAME = 'Rung'
    model_name = 'vae_flow_mel_mse_0_gated_cnn_planar_flow_post_maf_3'
    X = np.load(f'./synth/trained_flows/{SYNTH_NAME}/visualize/{model_name}/latents.npy')
    scaler_FLOW = StandardScaler()
    X = scaler_FLOW.fit_transform(X)
    pca_FLOW  = PCA(n_components=3)
    X_PCA = pca_FLOW.fit_transform(X)
    plot_scaler_FLOW  = StandardScaler()
    X_PCA_FLOW = plot_scaler_FLOW.fit_transform(X_PCA)


    # path = f'transformer/trained/elena-study-finetune_Granular/1790248738-transformerDiscreteT2V-B/01_live_recordings'
    # song_name = '1791309108'
    path = f'transformer/trained/andrea-study-plus_Rung/1790768071-transformerDiscreteT2V/01_live_recordings'
    song_name = '1791308201'
    signal_src, _ = librosa.load(f'{path}/{song_name}/performance_src.wav', sr=sr, mono=True)
    signal_tgt, _ = librosa.load(f'{path}/{song_name}/performance_tgt.wav', sr=sr, mono=True)

    SRC_events = extractEvents(signal_src, FFT_window, sr, hop_size, num_MFCC, num_MEL, 
                                min_event_duration_samples, max_event_duration_samples)

    # ADAPT TGT CSV
    TGT_events = extractEvents(signal_tgt, FFT_window, sr, hop_size, num_MFCC, num_MEL, 
                                min_event_duration_samples, max_event_duration_samples)


    DEVICE = torch.device("cpu")

    # LOAD VAE
    corpus_path = f'corpus/{corpus_name}'
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

    # COMPUTE SRC LATENTS
    src_events = []
    for event in SRC_events:
        if event['rms'] != 0:
            event_latents = melToVAElatents(event['mel'], VAE_model)
            event_latents = event_latents.tolist()
        else:
            event_latents = (np.zeros(latent_dim)).tolist()
        event_latents += [event['rms'], event['pitch'], event['event_duration'], event['event_start'], song_name]
        src_events.append(event_latents)
    src_events = np.array(src_events)
    columns = [f'latent-{j}' for j in range(latent_dim)]
    columns += ['rms', 'pitch', 'event_duration', 'event_start', 'filename']
    df_src = pd.DataFrame(src_events, columns=columns)

    # LOAD FLOW MODEL
    model_path = f'./synth/trained_flows/{SYNTH_NAME}/models'
    model_name = 'vae_flow_mel_mse_0_gated_cnn_planar_flow_post_maf_3'
    flow_model = torch.load(f'{model_path}/{model_name}.model', weights_only=False, map_location=DEVICE).to(DEVICE)
    flow_model.eval()
    # latent_dim = 7

    # # COMPUTE SRC LATENTS
    # tgt_events = []
    # for event in TGT_events:
    #     if event['rms'] != 0:
    #         event_latents, _ = melToFlowLatents(event['mel'], flow_model, -5.195, 3.284)
    #         event_latents = event_latents.tolist()
    #     else:
    #         event_latents = (np.zeros(latent_dim)).tolist()
    #     event_latents += [event['rms'], event['pitch'], event['event_duration'], event['event_start'], song_name]
    #     tgt_events.append(event_latents)
    # tgt_events = np.array(tgt_events)
    # columns = [f'latent-{j}' for j in range(latent_dim)]
    # columns += ['rms', 'pitch', 'event_duration', 'event_start', 'filename']
    # df_tgt = pd.DataFrame(tgt_events, columns=columns)


    # PLOT EVENTS
    sample_rate = 22050
    sample_start_s = 360+40
    sample_duration_s = 40

    SRC_file_event = {}
    SRC_file_event['filename'] = f'{path}/{song_name}/performance_src.wav'
    SRC_file_event['song_name'] = 'performance_src'
    SRC_file_event['events'] = SRC_events
    TGT_file_event = {}
    TGT_file_event['filename'] = f'{path}/{song_name}/performance_tgt.wav'
    TGT_file_event['song_name'] = 'performance_tgt'
    TGT_file_event['events'] = TGT_events
    # plotEvents(SRC_file_event, sr, sample_start_s=sample_start_s, sample_duration_s=sample_duration_s)
    plotPairedEvents(SRC_file_event, TGT_file_event, sample_rate)
    plotPairedEvents(SRC_file_event, TGT_file_event, sample_rate, sample_start_s=sample_start_s, sample_duration_s=sample_duration_s)


    # filter time for scatterplot
    sample_start = sample_start_s * sample_rate
    df_src['event_start'] = df_src['event_start'].astype(np.float32)
    df_src['event_duration'] = df_src['event_duration'].astype(np.float32)
    if sample_duration_s:
        sample_duration = sample_duration_s * sample_rate
    df_src = df_src[df_src['event_start'] >= sample_start]
    if sample_duration_s and (sample_start+sample_duration) < signal_tgt.shape[0]:
        df_src = df_src[df_src['event_start'] < sample_start+sample_duration]

    N_latents_src = 8
    VAE_latents = df_src[[f'latent-{j}' for j in range(N_latents_src)]].values.astype(np.float32)
    VAE_latents = VAE_latents[~np.all(VAE_latents == 0, axis=1)]
    X = np.load(f'./corpus/{corpus_name}/trained_VAE/latents.npy')

    # NORMALIZE LATENTS FOR PLOT
    path_latents_scaled = VAE_scaler.transform(VAE_latents)
    path_latents_PCA = pca_VAE.transform(path_latents_scaled)
    path_latents_plot = plot_scaler_VAE.transform(path_latents_PCA)

    fig, ax = plt.subplots(figsize=(6, 6))
    heatmap, xedges, yedges = np.histogram2d(X_PCA_VAE[:,0], X_PCA_VAE[:,1], bins=250)
    heatmap = gaussian_filter(heatmap, sigma=4)
    def stretch_contrast(img):
        img_min = np.min(img)
        img_max = np.max(img)
        return (img - img_min) / (img_max - img_min)
    heatmap = stretch_contrast(heatmap)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax.cla()
    # ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='magma', alpha=1)
    ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='cividis', alpha=1)
    # ax.set_xlim([-2.4, 2.4])
    # ax.set_ylim([-2.4, 2.4])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.axis('off')
    ax.margins(0)
    # cmap = plt.get_cmap('Blues')
    cmap = plt.get_cmap('Reds')
    for i in range(1,path_latents_plot.shape[0]-1):
        color = cmap((i / (path_latents_plot.shape[0]-2))/2+0.4)    
        ax.scatter(path_latents_plot[i-1,0], path_latents_plot[i-1,1], s=15, color=color)
        plt.plot(path_latents_plot[i-1:i+1,0], path_latents_plot[i-1:i+1,1], color=color, alpha=1.0, linestyle='-', lw=1.0)
    ax.scatter(path_latents_plot[i,0], path_latents_plot[i,1], s=10, color=color)
    plt.show()


    # LOAD FLOW LATENTS OF TGT
    df_tgt = pd.read_csv(f'{path}/{song_name}/tgt_events.csv')
    print(df_tgt.head())

    # filter time for scatterplot
    sample_start = sample_start_s * sample_rate
    # sample_start = sample_start_s * 27817
    df_tgt['event_start'] = df_tgt['event_start'].astype(np.float32)
    df_tgt['event_duration'] = df_tgt['event_duration'].astype(np.float32)
    if sample_duration_s:
        sample_duration = sample_duration_s * sample_rate
    df_tgt = df_tgt[df_tgt['event_start'] >= sample_start]
    if sample_duration_s and (sample_start+sample_duration) < signal_tgt.shape[0]:
        df_tgt = df_tgt[df_tgt['event_start'] < sample_start+sample_duration]

    N_latents_tgt = 3
    FLOW_latents = df_tgt[[f'latent-{j}' for j in range(N_latents_tgt)]].values.astype(np.float32)
    FLOW_latents = FLOW_latents[~np.all(FLOW_latents == 0, axis=1)]
    # X = np.load(f'./synth/trained_flows/{SYNTH_NAME}/visualize/{model_name}/latents.npy')

    # NORMALIZE LATENTS FOR PLOT
    path_latents_scaled = scaler_FLOW.transform(FLOW_latents)
    path_latents_PCA = pca_FLOW.transform(path_latents_scaled)
    path_latents_plot = plot_scaler_FLOW.transform(path_latents_PCA)

    mask = path_latents_plot > X_PCA_FLOW.max()
    path_latents_plot = path_latents_plot[~np.any(mask, axis=1)]
    mask = path_latents_plot < X_PCA_FLOW.min()
    path_latents_plot = path_latents_plot[~np.any(mask, axis=1)]

    fig, ax = plt.subplots(figsize=(6, 6))
    heatmap, xedges, yedges = np.histogram2d(X_PCA_FLOW[:,0], X_PCA_FLOW[:,1], bins=250)
    heatmap = gaussian_filter(heatmap, sigma=4)
    def stretch_contrast(img):
        img_min = np.min(img)
        img_max = np.max(img)
        return (img - img_min) / (img_max - img_min)
    heatmap = stretch_contrast(heatmap)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
    ax.cla()
    ax.imshow(heatmap.T, extent=extent, origin='lower', cmap='magma', alpha=1)
    # ax.set_xlim([-2.4, 2.4])
    # ax.set_ylim([-2.4, 2.4])
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
    ax.axis('off')
    ax.margins(0)
    cmap = plt.get_cmap('Blues')
    for i in range(1,path_latents_plot.shape[0]-1):
        color = cmap((i / (path_latents_plot.shape[0]-2))/2+0.4)    
        ax.scatter(path_latents_plot[i-1,0], path_latents_plot[i-1,1], s=15, color=color)
        plt.plot(path_latents_plot[i-1:i+1,0], path_latents_plot[i-1:i+1,1], color=color, alpha=1.0, linestyle='-', lw=1.0)
    ax.scatter(path_latents_plot[i,0], path_latents_plot[i,1], s=10, color=color)
    plt.show()
