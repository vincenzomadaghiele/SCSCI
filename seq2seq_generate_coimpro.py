import os
import json
import math
import time
import joblib
import librosa
import numpy as np
import pandas as pd
import soundfile as sf

import torch
from torchvision import transforms

from CONSTANTS import * 
from transformer.transformerDiscreteT2V import TransformerDiscreteT2V
from corpus.extract_events import extractEvents, plotEvents, plotPairedEvents
from VAE_train import Encoder, Decoder, Model
from synth.synths import Synth


def makeCrownWindow(size, fade_perc = 0.01):
    crown_window = np.ones(size)
    fade_num_samples = int(fade_perc * size)
    fade_in = np.linspace(0,1,fade_num_samples)
    fade_out = np.linspace(1,0,fade_num_samples)
    crown_window[0:fade_num_samples] = fade_in
    crown_window[crown_window.shape[0]-fade_num_samples:] = fade_out
    return crown_window


def generateCoImprovisation(src_audio_filepath, SYNTH_NAME, VAE_model, flow_model, model, 
                            training_parameters, 
                            src_scaler, tgt_scaler, src_time_scaler, tgt_time_scaler, 
                            useTimbre=True, DEVICE='cpu', model_dir=''):

    src_signal, _ = librosa.load(src_audio_filepath, sr=sr, mono=True)
    song_name = src_audio_filepath.split('/')[-2]

    # LOAD SYNTH
    RECORDING_TIME = int(melspec_size_flow * FFT_window / sr)
    synth = Synth(SYNTH_NAME, sr, hop_size, RECORDING_TIME)
    param_names = synth.getInputs()
    timbre_param_names = [p for p in param_names if p not in non_timbre_params_names]
    synth.close()

    if useTimbre:
        N_latents_tgt = training_parameters['N_latents_tgt']
    else:
        N_latents_tgt = 0

    image_size = 64
    latent_dim = 8
    image_size_flat = image_size * image_size

    pred_signal = np.zeros(src_signal.shape[0])
    event_sequence = extractEvents(src_signal, FFT_window, sr, hop_size, num_MFCC, num_MEL, min_event_duration_samples, max_event_duration_samples)
    transform = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size)])
    src_events = []
    for event in event_sequence:
        # detect rest
        if event['rms'] != 0:
            prep_event = librosa.power_to_db(event['mel'], ref=np.max)
            prep_event = torch.clip(torch.from_numpy((prep_event+80) / 80), min=0, max=1)
            prep_event = torch.unsqueeze(prep_event, 0).to(torch.float32)
            prep_event = transform(prep_event)
            prep_event = prep_event.reshape(prep_event.shape[0], image_size_flat).to(DEVICE)
            with torch.no_grad():
                event_latents, _ = VAE_model.Encoder(prep_event)
            event_latents = event_latents[0].detach().cpu().numpy().tolist()
        else:
            event_latents = (np.zeros(latent_dim)).tolist()
        event_latents += [event['rms'], event['pitch'], event['event_duration'], event['event_start'], src_audio_filepath]
        src_events.append(event_latents)
    src_events = np.array(src_events)
    columns = [f'latent-{j}' for j in range(training_parameters['N_latents_src'])]
    columns += ['rms', 'pitch', 'event_duration', 'event_start', 'filename']
    signal_df = pd.DataFrame(src_events, columns=columns)

    # normalize for input
    features_source = training_parameters['features_source']
    features_target = training_parameters['features_target']
    input_src = signal_df[features_source].values.astype(float)
    times_src = signal_df['event_start'].values.astype(float)

    # WINDOW_N_SAMPLES = FFT_window * sr
    WINDOW_N_SAMPLES = training_parameters['window_size_s'] * sr

    # predict output 
    input_src_array = np.zeros((1,len(features_source)+1))
    input_tgt_array = np.zeros((1,len(features_target)+1))

    # predict each time a new target token is finished (or tot tokens in advace)
    pred_signal_sample = 0
    PRED_TIME_SAMPLES = 0
    tgt_past_events = []
    tgt_past_events_starttimes = []

    # for stats
    inference_times = []
    model_inference_times = []
    sequence_update_times = []
    while pred_signal_sample < src_signal.shape[0]:

        start_inf_time = time.time()
        input_src_array_norm = np.zeros_like(input_src_array)
        input_tgt_array_norm = np.zeros_like(input_tgt_array)
        input_src_array_norm[:,:-1] = src_scaler.transform(input_src_array[:,:-1])
        input_tgt_array_norm[:,:-1] = tgt_scaler.transform(input_tgt_array[:,:-1])
        input_src_array_norm[:,-1] = src_time_scaler.transform(input_src_array[:,-1].reshape(-1,1)).reshape(-1)
        input_tgt_array_norm[:,-1] = tgt_time_scaler.transform(input_tgt_array[:,-1].reshape(-1,1)).reshape(-1)
        ones_src = np.ones((training_parameters['max_sequence_length'],len(features_source)+1)) * training_parameters['pad_token']
        ones_tgt = np.ones((training_parameters['max_sequence_length'],len(features_target)+1)) * training_parameters['pad_token']
        ones_src[:input_src_array_norm.shape[0],:] = input_src_array_norm
        ones_tgt[:input_tgt_array_norm.shape[0],:] = input_tgt_array_norm
        input_src_array_norm = ones_src
        input_tgt_array_norm = ones_tgt

        input_src_array_norm = torch.tensor(input_src_array_norm.reshape(1, input_src_array_norm.shape[0], input_src_array_norm.shape[1])).type(torch.FloatTensor).to(DEVICE)
        input_tgt_array_norm = torch.tensor(input_tgt_array_norm.reshape(1, input_tgt_array_norm.shape[0], input_tgt_array_norm.shape[1])).type(torch.FloatTensor).to(DEVICE)
        start_model_time = time.time()
        with torch.no_grad():
            src_pad_mask = model.create_pad_mask(input_src_array_norm, pad_token=training_parameters['pad_token']).to(DEVICE).to(torch.bool)
            tgt_pad_mask = model.create_pad_mask(input_tgt_array_norm, pad_token=training_parameters['pad_token']).to(DEVICE).to(torch.bool)
            y = model(input_src_array_norm, input_tgt_array_norm, src_pad_mask=src_pad_mask[:,:,0], tgt_pad_mask=tgt_pad_mask[:,:,0]).detach().cpu().numpy()
        end_model_time = time.time()
        y_pred = tgt_scaler.inverse_transform(y.reshape(1, -1)).reshape(-1)
        pred_latents = y_pred[:N_latents_tgt]
        pred_rms = y_pred[N_latents_tgt]
        pred_frequency = np.clip(y_pred[N_latents_tgt+1], 50, 11050)
        pred_duration = int(np.clip(y_pred[N_latents_tgt+2], 882, 88200))

        # quantize pitch
        quantizePitch = True
        if quantizePitch:
            MIDI = int(69 + 12 * math.log2(pred_frequency / 440))
            pred_frequency = 440 * 2**((MIDI - 69) / 12) # tune to scale

        y_pred[N_latents_tgt] = pred_rms
        y_pred[N_latents_tgt+1] = pred_frequency
        y_pred[N_latents_tgt+2] = pred_duration
        
        # detect rest
        if pred_rms < 0.0001:
            sound_segment = np.zeros(pred_duration)
            y_pred[:N_latents_tgt] = np.zeros_like(y_pred[:N_latents_tgt])
            y_pred[N_latents_tgt] = 0
            y_pred[N_latents_tgt+1] = 0
            print(f'[{pred_signal_sample/sr:.3f}] SYNTH REST  | rms: {0:.3f}, pitch: {0:.3f}, duration: {pred_duration / sr:.3f}')
        else:
            synth = Synth(SYNTH_NAME, sr, hop_size, RECORDING_TIME)
            synth.setParameters(['frequency'], [pred_frequency])
            # synth.setParameters(['gain'], [float(np.clip(pred_rms*20, 0, 1))])
            synth.setParameters(['gain'], [float(pred_rms*RMS_multiplier, 0, 1)])
            synth.setParameters(envelope_param_names, ADSR_standard)
            if useTimbre:
                next_latents = torch.tensor(pred_latents).reshape(1, -1).type(torch.FloatTensor).to(DEVICE)
                with torch.no_grad():
                    predicted_parameters = flow_model.regression_model(next_latents) # parameters
                predicted_parameters = predicted_parameters[0].detach().cpu().numpy()
                synth.setParameters(timbre_param_names, predicted_parameters.tolist())
            else:
                synth.setParameters(timbre_param_names, np.ones(len(timbre_param_names)).tolist() * 0.5)
            sound_segment = synth.forward((pred_duration - pred_duration/4) / sr)
            synth.close()
            sound_segment = sound_segment[:pred_duration]
            print(f'[{pred_signal_sample/sr:.3f}] SYNTH EVENT | rms: {pred_rms:.3f}, pitch: {pred_frequency:.3f}, duration: {pred_duration / sr:.3f}')
        # update model inputs at time of end of last predicted token
        tgt_past_events.append(y_pred)
        tgt_past_events_starttimes.append(pred_signal_sample)
        # update input sequences
        if pred_signal_sample + int(pred_duration) < src_signal.shape[0]:
            start_sequpdate_time = time.time()
            # concatenate synthesized sample
            crown_window = makeCrownWindow(int(sound_segment.shape[0]), fade_perc=0.01)
            pred_signal[int(pred_signal_sample):int(pred_signal_sample)+int(pred_duration)] = sound_segment * crown_window
            pred_signal_sample += int(pred_duration)
            # COMPUTE NEXT WINDOW START
            window_start_idx = (pred_signal_sample - PRED_TIME_SAMPLES) - WINDOW_N_SAMPLES
            # get source events in window before current
            indices = np.where((times_src >= window_start_idx) & (times_src < pred_signal_sample))
            # add time difference index
            x_src_window_times = (pred_signal_sample - PRED_TIME_SAMPLES) - times_src[indices]
            input_src_array = np.concatenate((input_src[indices], x_src_window_times.reshape(-1,1)), axis=1)[-training_parameters['max_sequence_length']:,:]
            # get target events in window before current
            times_tgt = np.array(tgt_past_events_starttimes)
            input_tgt = np.array(tgt_past_events)
            indices = np.where((times_tgt >= window_start_idx) & (times_tgt < pred_signal_sample))
            # add time difference index
            x_tgt_window_times = (pred_signal_sample - PRED_TIME_SAMPLES) - times_tgt[indices]
            input_tgt_array = np.concatenate((input_tgt[indices], x_tgt_window_times.reshape(-1,1)), axis=1)[-training_parameters['max_sequence_length']:,:]
            # update metrics
            end_sequpdate_time = time.time()
            end_inf_time = time.time()
            inference_times.append(end_inf_time-start_inf_time)
            model_inference_times.append(end_model_time-start_model_time)
            sequence_update_times.append(end_sequpdate_time-start_sequpdate_time)
        else:
            break

    print()
    print(f'Mean total inference time: {np.array(inference_times).mean():.6f}')
    print(f'Mean model inference time: {np.array(model_inference_times).mean():.6f}')
    print(f'Mean sequence update time: {np.array(sequence_update_times).mean():.6f}')
    print()
    print('Exporting...')

    # EXPORT AUDIOFILES
    results_dir = f'{model_dir}/00_results/{song_name}'
    os.makedirs(results_dir, exist_ok=True)
    out_signal = np.concatenate((src_signal.reshape(-1,1), pred_signal.reshape(-1,1)), axis=1)
    sf.write(f'{results_dir}/{song_name}_combined.wav', out_signal, sr, subtype='PCM_24')
    sf.write(f'{results_dir}/{song_name}_{training_parameters["source_track_name"]}.wav', src_signal, sr, subtype='PCM_24')
    sf.write(f'{results_dir}/{song_name}_{SYNTH_NAME}.wav', pred_signal, sr, subtype='PCM_24')

    columns = [f'latent-{j}' for j in range(N_latents_tgt)]
    # columns = []
    columns.append('rms')
    columns.append('pitch')
    columns.append('event_duration')
    df_tgt = pd.DataFrame(tgt_past_events, columns=columns)
    df_tgt['event_start'] = tgt_past_events_starttimes
    df_tgt.to_csv(f"{results_dir}/tgt_events.csv")

    df_tgt['filename'] = [f'{results_dir}/{song_name}_{SYNTH_NAME}.wav' for _ in range(df_tgt.values.shape[0])]
    plotEvents(df_tgt, sr, save_dir=results_dir, isDfin=True)
    plotPairedEvents(signal_df, df_tgt, sr, save_dir=results_dir, isDfin=True)

    df_tgt = pd.DataFrame(tgt_past_events, columns=features_target)
    df_tgt["event_start"] = tgt_past_events_starttimes
    print(df_tgt.head())
    df_tgt.to_csv(f"{results_dir}/{song_name}_{SYNTH_NAME}_tgt_events.csv")
    return df_tgt


np.set_printoptions(precision=3, suppress=True)

if __name__ == '__main__': 

    # device 
    DEVICE = torch.device("cpu")
    corpus_name = 'guitar-duo'
    src_track_name = 'guitar1'
    SYNTH_NAME = 'TimbreTheremin'
    transformer_name = '1787065997-transformerDiscreteT2V'
    song_name = 'recording7'
    model_name = 'vae_flow_mel_mse_0_gated_cnn_planar_flow_post_maf_3'

    # LOAD TRANSFORMER
    model_dir = f'transformer/trained/{corpus_name}_{SYNTH_NAME}/{transformer_name}'
    with open(f'{model_dir}/training_config.json', 'r') as f:
        training_parameters = json.load(f)
    model_load_path = f'{model_dir}/best_model.pth'
    training_parameters['use_timbre'] = True
    training_parameters['N_latents_src'] = 8
    training_parameters['N_latents_tgt'] = 2
    if training_parameters['use_timbre']:
        N_latents_tgt = training_parameters['N_latents_tgt']
        features_source = [f'latent-{j}' for j in range(training_parameters['N_latents_src'])]
        features_source += ['rms', 'pitch', 'event_duration']
        features_target = [f'latent-{j}' for j in range(training_parameters['N_latents_tgt'])]
        features_target += ['rms', 'pitch', 'event_duration']
    else:
        N_latents_tgt = 0
        features_source = ['rms', 'pitch', 'event_duration']
        features_target = ['rms', 'pitch', 'event_duration']

    # create model
    seq2seq_model = TransformerDiscreteT2V(
            num_featuresin=len(features_source)+1,
            num_featuresout=len(features_target)+1,
            num_frequency=training_parameters['num_frequency'],
            max_seq_len=training_parameters['max_sequence_length'],
            dim_model=training_parameters['dim_model'], 
            num_heads=training_parameters['num_heads'], 
            num_encoder_layers=training_parameters['num_encoder_layers'], 
            num_decoder_layers=training_parameters['num_decoder_layers'], 
            dim_feedforward=training_parameters['dim_feedforward'], 
            dropout_p=training_parameters['dropout_p']
    ).to(DEVICE)
    seq2seq_model.load_state_dict(torch.load(model_load_path, weights_only=True))
    seq2seq_model.eval()
    # LOAD MODEL UTILS
    src_scaler = joblib.load(f'{model_dir}/source_scaler.pkl')
    tgt_scaler = joblib.load(f'{model_dir}/target_scaler.pkl')
    src_time_scaler = joblib.load(f'{model_dir}/source_time_scaler.pkl')
    tgt_time_scaler = joblib.load(f'{model_dir}/target_time_scaler.pkl')

    # LOAD VAE FOR SRC ENCODING
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

    # LOAD FLOW MODEL FOR TGT COMPUTATION
    model_path = f'./synth/trained_flows/{SYNTH_NAME}/models'
    flow_model = torch.load(f'{model_path}/{model_name}.model', weights_only=False, map_location=DEVICE).to(DEVICE)
    flow_model.eval()

    # LOAD AUDIO FILE
    src_audio_filepath = f'corpus/{corpus_name}/{song_name}/{src_track_name}.wav'
    generateCoImprovisation(src_audio_filepath, SYNTH_NAME, VAE_model, flow_model, seq2seq_model, 
                            training_parameters, 
                            src_scaler, tgt_scaler, src_time_scaler, tgt_time_scaler, 
                            training_parameters['use_timbre'], DEVICE)
