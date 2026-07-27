import time
import json
import joblib
import librosa
import numpy as np
import pandas as pd
from signalflow import * 

import torch

from CONSTANTS import * 
from transformer.transformerDiscreteT2V import TransformerDiscreteT2V
from VAE_train import Encoder, Decoder, Model
from corpus.onset_detect import OnsetDetection
from resynthesize import melToVAElatents, melToFlowLatents
from synth.patches.theremin import ADSRTheremin


# FIX THIS PLEASE!!


if __name__ == "__main__":

    # device 
    device = torch.device("cpu")
    playimpulse = True

    corpus_name = 'moisesdb-BassDrums'
    SYNTH_NAME = 'Theremin'
    model_dir = f'transformer/trained/{corpus_name}_{SYNTH_NAME}/1785140002-transformerDiscreteT2V'
    with open(f'{model_dir}/training_config.json', 'r') as f:
        training_parameters = json.load(f)
    model_load_path = f'{model_dir}/best_model.pth'

    N_latents_src = 8
    N_latents_tgt = 5
    features_source = [f'latent-{j}' for j in range(N_latents_src)]
    features_source.append('event_duration')
    features_target = [f'latent-{j}' for j in range(N_latents_tgt)]
    features_target.append('event_duration')

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

    # LOAD FLOW MODEL
    SYNTH_NAME = 'Theremin'
    model_path = f'./synth/trained_flows/{SYNTH_NAME}/models'
    model_name = 'vae_flow_mel_mse_0_gated_cnn_planar_flow_post_maf_3'
    flow_model = torch.load(f'{model_path}/{model_name}.model', weights_only=False, map_location=device).to(device)
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

    # create model
    model = TransformerDiscreteT2V(
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
    ).to(device)

    model.load_state_dict(torch.load(model_load_path, weights_only=True))
    model.eval()
    src_scaler = joblib.load(f'{model_dir}/source_scaler.pkl')
    tgt_scaler = joblib.load(f'{model_dir}/target_scaler.pkl')
    src_time_scaler = joblib.load(f'{model_dir}/source_time_scaler.pkl')
    tgt_time_scaler = joblib.load(f'{model_dir}/target_time_scaler.pkl')

    # START AUDIO GRAPH
    config = AudioGraphConfig()
    config.sample_rate = sr
    config.output_buffer_size = FFT_window
    config.output_device_name = "UltraLite-mk5"
    config.input_device_name = "UltraLite-mk5"
    graph = AudioGraph(config)
    audio_in = AudioIn() 
    onsetDetector = OnsetDetection(sr=sr)
    impulse = Impulse(frequency=0)
    graph.play([audio_in*4, impulse]) 

    if SYNTH_NAME == 'Theremin':
        synth = ADSRTheremin() 
    synth.play()
    param_names = list(synth.inputs.keys())

    # INTIALIZE AGENT
    update_interval_s = hop_size / sr
    input_buffer = Buffer(1, FFT_window)
    min_feedback_writer_delay = graph.output_buffer_size / graph.sample_rate # samples
    graph.add_node(FeedbackBufferWriter(input_buffer, audio_in, min_feedback_writer_delay))

    input_sound = input_buffer.data[0,:]
    prev_onset_time = time.time()
    start_time = prev_onset_time

    WINDOW_N_SAMPLES = training_parameters['window_size_s'] * sr

    # predict output 
    input_src_array = np.zeros((1,len(features_source)+1))
    input_tgt_array = np.zeros((1,len(features_target)+1))

    features = np.zeros(len(features_source))
    last_onset = 0
    PRED_TIME_SAMPLES = 0
    pred_signal_sample = 0
    src_signal_sample = 0
    model_inference_time_s = 0.003
    inference_time_samples = model_inference_time_s * sr
    current_src_onset = np.zeros(FFT_window)
    current_tgt_onset = np.zeros(FFT_window)
    tgt_samples_left = 0
    silence_threshold = 0.01 # SILENCE THREHSOLD NEEDS TO BE ADAPTED TO LIVE SITUATION

    # log events
    src_past_events = []
    src_past_events_starttimes = []
    tgt_past_events = []
    tgt_past_events_starttimes = []
    while True:
        signal = input_buffer.data[0,:]
        # LISTEN TO INPUT (detect onset in current window)
        onsets_samples = onsetDetector(signal.reshape(1,-1))
        if onsets_samples:
            if onsets_samples[0] != last_onset:
                # print('NEW ONSET!')
                print('ONSET')
                print(onsets_samples)

                # analyze last onset
                current_src_onset = np.concatenate((current_src_onset, signal[:onsets_samples[0]]))
                print(current_src_onset.shape)
                # compute features for last onset

                mel = librosa.feature.melspectrogram(y=current_src_onset, sr=sr, n_fft=FFT_window, 
                                                    n_mels=num_MEL, hop_length=hop_size, fmin=30, fmax=11000)
                mel = mel[:num_MEL, :melspec_size_flow]
                event_latents = melToVAElatents(mel, VAE_model)
                event_latents = event_latents.tolist()
                event_latents.append(current_src_onset.shape[0]) # event duration
                features = np.array(event_latents)
                # print(features)

                # append event to event log
                src_past_events.append(features.reshape(-1))
                src_past_events_starttimes.append(src_signal_sample)
                src_signal_sample += int(current_src_onset.shape[0])
                # print(f'NEW ONSET --> amp: {features.reshape(-1)[0]:.3f} duration: {features.reshape(-1)[-1]:.3f}')

                # start new onset
                current_src_onset = signal[onsets_samples[0]:]
                last_onset = onsets_samples[0]
                if playimpulse:
                    graph.play(StereoPanner(Impulse(frequency=0), -1))
 
        else:
            # current_src_onset = np.concatenate((current_src_onset, signal[hop_size:]))

            # check if src onset is finished and a rest has to be added
            rms = librosa.feature.rms(y=signal, frame_length=FFT_window, hop_length=hop_size)
            if rms[0][:32].mean() < silence_threshold:
                # we are inside a rest

                # check if max rest size has been reached
                if current_src_onset.shape[0] > max_event_duration_samples:
                    # print('REST ENDED, NEW REST BEGINS')
                    print('REST')
                    features = np.zeros(len(features_source))-5 # it's a rest so features are all -5
                    features[-1] = max_event_duration_samples
                    # print(features)

                    # append event to event log
                    src_past_events.append(features.reshape(-1))
                    src_past_events_starttimes.append(src_signal_sample)
                    src_signal_sample += int(max_event_duration_samples)

                    print(max_event_duration_samples)
                    # print(f'NEW REST --> amp: {features.reshape(-1)[0]:.3f} duration: {features.reshape(-1)[-1]:.3f}')
                    current_src_onset = np.zeros(signal[hop_size:].shape)
                else:
                    # print('IN A REST')
                    current_src_onset = np.concatenate((current_src_onset, np.zeros(signal[current_src_onset.shape[0]-max_event_duration_samples:].shape)))
            else:
                # we are inside an onset, need to check if it ends
                end_event_sample = -1
                for k, rms_value in enumerate(rms[0].tolist()):
                    if rms_value > silence_threshold:
                        end_event_sample=k
                        break
                if end_event_sample != -1:
                    # print('ONSET ENDED, STARTING A REST')
                    print('REST')
                    # analyze last onset
                    current_src_onset = np.concatenate((current_src_onset, signal[:end_event_sample]))
                    # compute features for last onset
                    print(current_src_onset.shape)

                    mel = librosa.feature.melspectrogram(y=current_src_onset, sr=sr, n_fft=FFT_window, 
                                                        n_mels=num_MEL, hop_length=hop_size, fmin=30, fmax=11000)
                    mel = mel[:num_MEL, :melspec_size_flow]
                    event_latents = melToVAElatents(mel, VAE_model)
                    event_latents = event_latents.tolist()
                    event_latents.append(current_src_onset.shape[0]) # event duration
                    features = np.array(event_latents)
                    # print(features)

                    # append event to event log
                    src_past_events.append(features.reshape(-1))
                    src_past_events_starttimes.append(src_signal_sample)
                    src_signal_sample += int(max_event_duration_samples)

                    # print(f'NEW ONSET --> amp: {features.reshape(-1)[0]:.3f} duration: {features.reshape(-1)[-1]:.3f}')
                    current_src_onset = np.zeros(signal[hop_size:].shape)
                else:
                    # print('IN AN ONSET')
                    current_src_onset = np.concatenate((current_src_onset, signal[hop_size:]))



        # CHECK IF IT IS TIME TO GENERATE NEW OUTPUT (detect onset in current window)
        if tgt_samples_left - inference_time_samples < FFT_window:

            # COMPUTE RESPONSE
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
            input_src_array_norm = torch.tensor(input_src_array_norm.reshape(1, input_src_array_norm.shape[0], input_src_array_norm.shape[1])).type(torch.FloatTensor).to(device)
            input_tgt_array_norm = torch.tensor(input_tgt_array_norm.reshape(1, input_tgt_array_norm.shape[0], input_tgt_array_norm.shape[1])).type(torch.FloatTensor).to(device)
            start_model_time = time.time()
            with torch.no_grad():
                src_pad_mask = model.create_pad_mask(input_src_array_norm, pad_token=training_parameters['pad_token']).to(device)
                tgt_pad_mask = model.create_pad_mask(input_tgt_array_norm, pad_token=training_parameters['pad_token']).to(device)
                tgt_mask = model.get_tgt_mask(input_tgt_array_norm.size(1)).to(device)
                y = model(input_src_array_norm, input_tgt_array_norm, tgt_mask=tgt_mask, src_pad_mask=src_pad_mask[:,:,0], tgt_pad_mask=tgt_pad_mask[:,:,0]).detach().cpu().numpy()
            y_denorm = tgt_scaler.inverse_transform(y.reshape(y.shape[0], y.shape[2]))
            next_token = y_denorm[-1,:]


            y_pred = next_token
            pred_duration = int(y_pred[-1])
            # detect rest
            if all(y_pred[:-1].tolist()) < -4:
                # sound_segment = np.zeros(pred_duration)
                pass
            else:
                next_latents = torch.tensor(y_pred[:-1]).reshape(1, -1).type(torch.FloatTensor).to(device)
                with torch.no_grad():
                    predicted_parameters = flow_model.regression_model(next_latents) # parameters
                predicted_parameters = predicted_parameters[0].detach().cpu().numpy()
                # synth.setParameters(param_names, predicted_parameters.tolist())
                # sound_segment = synth.forward() # add possibility of 'note off' to handle different durations synth.forward(duration)
                # sound_segment = sound_segment[:pred_duration]
                print(y_pred[:-1])
                print(next_latents)
                print(predicted_parameters)
                print(pred_duration)

                predicted_parameters = predicted_parameters.tolist()
                for i, param_name in enumerate(param_names):
                    synth.set_input(param_name, predicted_parameters[i]) 
                synth.trigger() # toDo: trigger with ms latency and generate new patch with auto-free each new synth

            end_lookup_time = time.time()

            # retreive sample and play
            # closest_index = (target_corpus_df[features_target] - next_token).abs().idxmin()
            # closest_row = target_corpus_df.loc[closest_index[0]]
            # sound_segment = concatenative_corpus[closest_row['filename']][int(closest_row['event_start']):int(closest_row['event_start'])+int(closest_row['event_duration'])]
            # crown_window = makeCrownEnvelope(int(closest_row['event_duration']), fade_perc=0.01)
            # buf = Buffer(sound_segment * crown_window)
            # player = BufferPlayer(buf)
            # delay_s = tgt_samples_left / sr
            # graph.play(StereoPanner(OneTapDelay(input=player, delay_time=delay_s), 1))

            # append event to event log
            tgt_past_events.append(next_token.reshape(-1))
            tgt_past_events_starttimes.append(pred_signal_sample)
            # tgt_past_events = tgt_past_events[:-training_parameters['max_sequence_length']]
            # tgt_past_events_starttimes = tgt_past_events_starttimes[:-training_parameters['max_sequence_length']]
            pred_signal_sample += int(pred_duration)

            if len(src_past_events) > 0 and len(tgt_past_events) > 0:

                # get source events in window before current
                input_src = np.array(src_past_events)
                times_src = np.array(src_past_events_starttimes)#.reshape(-1,1)
                window_start_idx = (pred_signal_sample - PRED_TIME_SAMPLES) - WINDOW_N_SAMPLES
                indices = np.where((times_src >= window_start_idx) & (times_src < pred_signal_sample))
                x_src_window_times = (pred_signal_sample - PRED_TIME_SAMPLES) - times_src[indices].reshape(-1,1) # add time difference index
                input_src_array = np.concatenate((input_src[indices], x_src_window_times.reshape(-1,1)), axis=1)[-training_parameters['max_sequence_length']:,:]

                # # remove old events from memory
                # src_past_events_starttimes = times_src[indices].tolist()
                # src_past_events = input_src[indices].tolist()

                # update sequences
                input_tgt = np.array(tgt_past_events)
                times_tgt = np.array(tgt_past_events_starttimes)#.reshape(-1,1)
                indices = np.where((times_tgt >= window_start_idx) & (times_tgt < pred_signal_sample))
                x_tgt_window_times = (pred_signal_sample - PRED_TIME_SAMPLES) - times_tgt[indices].reshape(-1,1) # add time difference index
                input_tgt_array = np.concatenate((input_tgt[indices], x_tgt_window_times.reshape(-1,1)), axis=1)[-training_parameters['max_sequence_length']:,:]
                
                # # remove old events from memory
                # tgt_past_events_starttimes = times_tgt[indices].tolist()
                # tgt_past_events = input_tgt[indices].tolist()

            # samples countdown
            tgt_samples_left = pred_duration - (FFT_window - (tgt_samples_left - inference_time_samples))
        else:
            tgt_samples_left -= hop_size 




        # check every 
        time.sleep(update_interval_s)

    graph.wait()
