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
from synth.patches.timbreTheremin import TimbreTheremin


np.set_printoptions(precision=3, suppress=True)


if __name__ == "__main__":

    # device 
    device = torch.device("cpu")
    playimpulse = True

    corpus_name = 'guitar-duo'
    SYNTH_NAME = 'TimbreTheremin'
    flow_model_name = 'vae_flow_mel_mse_0_gated_cnn_planar_flow_post_maf_3'
    transformer_name = '1786739157-transformerDiscreteT2V'

    model_dir = f'transformer/trained/{corpus_name}_{SYNTH_NAME}/{transformer_name}'
    with open(f'{model_dir}/training_config.json', 'r') as f:
        training_parameters = json.load(f)
    model_load_path = f'{model_dir}/best_model.pth'

    N_latents_src = 8
    N_latents_tgt = 2
    features_source = [f'latent-{j}' for j in range(N_latents_src)]
    features_source.append('rms')
    features_source.append('pitch')
    features_source.append('event_duration')
    features_target = [f'latent-{j}' for j in range(N_latents_tgt)]
    features_target.append('rms')
    features_target.append('pitch')
    features_target.append('event_duration')

    features_source = ['rms', 'pitch', 'event_duration']
    features_target = ['rms', 'pitch', 'event_duration']


    # LOAD VAE
    timbre_pitch = True
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
    model_path = f'./synth/trained_flows/{SYNTH_NAME}/models'
    flow_model = torch.load(f'{model_path}/{flow_model_name}.model', weights_only=False, map_location=device).to(device)
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

    if SYNTH_NAME == 'TimbreTheremin':
        synth = TimbreTheremin() 
    synth.play()
    param_names = list(synth.inputs.keys())
    # THESE ARE THE SAME FOR ALL SYNTHS
    frequency_param_name = "frequency"
    gain_param_name = "gain"
    gate_param_name = "gate"
    envelope_param_names = ['A','D','S','R']
    # THESE CAN CHANGE AMONG SYNTHS
    non_timbre_params_names = [frequency_param_name, gain_param_name, gate_param_name]
    non_timbre_params_names += envelope_param_names
    timbre_param_names = [p for p in param_names if p not in non_timbre_params_names]


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
    SAVE_TIMER = 50 # save tables every SAVE_TIMER iterations
    while True:
        signal = input_buffer.data[0,:]
        # CHECK IF THERE ARE NEW ONSET IN THE NEW WINDOW
        onsets_samples = onsetDetector(signal.reshape(1,-1))
        if onsets_samples:
            # THERE ARE ONSETS! THE PREVIOUS EVENT HAS ENDED
            if onsets_samples[0] != last_onset:
                # PREVIOUS EVENT FINISHED BECAUSE NEW EVENT STARTED
                # print(onsets_samples)

                # analyze finished event
                current_src_onset = np.concatenate((current_src_onset, signal[:onsets_samples[0]]))
                # print(current_src_onset.shape)

                # compute features of event
                rms = librosa.feature.rms(y=current_src_onset, frame_length=FFT_window, hop_length=hop_size)
                pitch = librosa.yin(current_src_onset, fmin=50, fmax=11000, sr=sr, frame_length=FFT_window, hop_length=hop_size)
                event_duration = current_src_onset.shape[0]
                if timbre_pitch:
                    # transpose to base freq for mel computation
                    BASE_FREQ = 220
                    base_pitch = librosa.hz_to_midi(BASE_FREQ)
                    this_pitch = librosa.hz_to_midi(pitch.mean())
                    current_src_onset = librosa.effects.pitch_shift(current_src_onset, sr=sr, n_steps=this_pitch-base_pitch)
                mel = librosa.feature.melspectrogram(y=current_src_onset, sr=sr, n_fft=FFT_window, n_mels=num_MEL, hop_length=hop_size, fmin=50, fmax=11000)
                mel = mel[:num_MEL, :melspec_size_flow]
                event_latents = melToVAElatents(mel, VAE_model)
                event_latents = event_latents.tolist()
                event_latents = []
                event_latents.append(rms.mean()) 
                event_latents.append(pitch.mean()) 
                event_latents.append(np.clip(event_duration, 882, 88200)) 
                features = np.array(event_latents)
                print(f'rms: {rms.mean():.3f}, pitch: {pitch.mean():.3f}, duration: {event_duration /sr:.3f}')

                # append event to event log
                src_past_events.append(features.reshape(-1))
                src_past_events_starttimes.append(src_signal_sample)
                src_signal_sample += int(current_src_onset.shape[0])

                # start new onset
                current_src_onset = signal[onsets_samples[0]:]
                last_onset = onsets_samples[0]
                if playimpulse:
                    graph.play(StereoPanner(Impulse(frequency=0), -1))
            # IF PREVIOUS EVENT WAS A REST WE NEED TO APPEND IT
        else:
            # THERE ARE NO ONSETS
            rms = librosa.feature.rms(y=signal, frame_length=FFT_window, hop_length=hop_size) # check for rests 
            if rms.mean() < silence_threshold: # CHECK IF WE ARE INSIDE A REST
                if current_src_onset.shape[0] > max_event_duration_samples: # CHECK IF REST IS ENDED AND ANOTHER REST STARTS
                    # print('REST')
                    features = np.zeros(len(features_source)) 
                    features[-1] = max_event_duration_samples

                    # append event to event log
                    src_past_events.append(features.reshape(-1))
                    src_past_events_starttimes.append(src_signal_sample)
                    src_signal_sample += int(max_event_duration_samples)
                    print(f'rms: {0.0}, pitch: {0.0}, duration: {int(max_event_duration_samples) / sr:.3f}')


                    # start new rest event
                    current_src_onset = np.zeros(current_src_onset.shape[0] - max_event_duration_samples)
                else:
                    # WE ARE IN A REST THAT HASN'T ENDED
                    current_src_onset = np.concatenate((current_src_onset, np.zeros(signal[hop_size:].shape[0])))
            else:
                # we are inside an onset, need to check if it ends
                end_event_sample = -1
                for k, rms_value in enumerate(rms[0].tolist()):
                    if rms_value > silence_threshold:
                        end_event_sample=k
                        break
                if end_event_sample != -1: # ONSET HAS ENDED
                    # print('REST')
                    # analyze finished event
                    current_src_onset = np.concatenate((current_src_onset, signal[:end_event_sample]))
                    # print(current_src_onset.shape)

                    # compute features of event
                    rms = librosa.feature.rms(y=current_src_onset, frame_length=FFT_window, hop_length=hop_size)
                    pitch = librosa.yin(current_src_onset, fmin=50, fmax=11000, sr=sr, frame_length=FFT_window, hop_length=hop_size)
                    event_duration = current_src_onset.shape[0]
                    if timbre_pitch:
                        # transpose to base freq for mel computation
                        BASE_FREQ = 220
                        base_pitch = librosa.hz_to_midi(BASE_FREQ)
                        this_pitch = librosa.hz_to_midi(pitch.mean())
                        current_src_onset = librosa.effects.pitch_shift(current_src_onset, sr=sr, n_steps=this_pitch-base_pitch)
                    mel = librosa.feature.melspectrogram(y=current_src_onset, sr=sr, n_fft=FFT_window, n_mels=num_MEL, hop_length=hop_size, fmin=50, fmax=11000)
                    mel = mel[:num_MEL, :melspec_size_flow]
                    event_latents = melToVAElatents(mel, VAE_model)
                    event_latents = event_latents.tolist()
                    event_latents = []
                    event_latents.append(rms.mean()) 
                    event_latents.append(pitch.mean()) 
                    event_latents.append(np.clip(event_duration, 882, 88200)) 
                    features = np.array(event_latents)
                    print(f'rms: {rms.mean():.3f}, pitch: {pitch.mean():.3f}, duration: {event_duration / sr:.3f}')

                    # append event to event log
                    src_past_events.append(features.reshape(-1))
                    src_past_events_starttimes.append(src_signal_sample)
                    src_signal_sample += int(max_event_duration_samples)

                    # current_src_onset = np.zeros(signal[hop_size:].shape)
                    current_src_onset = signal[end_event_sample:]
                else:
                    # ONSET DIDN'T END
                    current_src_onset = np.concatenate((current_src_onset, signal[hop_size:]))


        # CHECK IF IT IS TIME TO GENERATE NEW OUTPUT
        if tgt_samples_left - inference_time_samples < FFT_window:
            # STOP CURRENT PLAYBACK
            synth.set_input('gate', 0) 

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
                src_pad_mask = model.create_pad_mask(input_src_array_norm, pad_token=training_parameters['pad_token']).to(device).to(torch.bool)
                tgt_pad_mask = model.create_pad_mask(input_tgt_array_norm, pad_token=training_parameters['pad_token']).to(device).to(torch.bool)
                # tgt_mask = model.get_tgt_mask(input_tgt_array_norm.size(1)).to(device)
                y = model(input_src_array_norm, input_tgt_array_norm, src_pad_mask=src_pad_mask[:,:,0], tgt_pad_mask=tgt_pad_mask[:,:,0]).detach().cpu().numpy()
            # y_denorm = tgt_scaler.inverse_transform(y.reshape(y.shape[0], y.shape[2]))
            y_pred = tgt_scaler.inverse_transform(y.reshape(1, -1)).reshape(-1)

            N_latents_tgt = 0
            # pred_latents = y_pred[:N_latents_tgt]
            pred_rms = y_pred[N_latents_tgt]
            pred_frequency = np.clip(y_pred[N_latents_tgt+1], 50, 11050)
            pred_duration = int(np.clip(y_pred[N_latents_tgt+2], 882, 88200))

            y_pred[N_latents_tgt] = pred_rms
            y_pred[N_latents_tgt+1] = pred_frequency
            y_pred[N_latents_tgt+2] = pred_duration
        
            print(y.reshape(-1), y_pred)

            if pred_rms < 0.0001:
                sound_segment = np.zeros(pred_duration)
                y_pred[N_latents_tgt] = 0
                y_pred[N_latents_tgt+1] = 0
                # y_pred = np.zeros_like(y_pred)
            else:
                # next_latents = torch.tensor(pred_latents).reshape(1, -1).type(torch.FloatTensor).to(DEVICE)
                # with torch.no_grad():
                #     predicted_parameters = flow_model.regression_model(next_latents) # parameters
                # predicted_parameters = predicted_parameters[0].detach().cpu().numpy()
                # print(predicted_parameters)
                if SYNTH_NAME == 'TimbreTheremin':
                    synth = TimbreTheremin() 
                synth.set_input('frequency', pred_frequency) 
                synth.set_input('gain', float(np.clip(pred_rms*5, 0, 1))) 
                synth.set_input('frequency', pred_frequency) 
                env_params = [0.3, 0.3, 0.2, 0.1]
                for i, param_name in enumerate(envelope_param_names):
                    synth.set_input(param_name, env_params[i]) 
                timbre_params = [0.3, 0.3, 0.2, 0.1]
                # timbre_params = predicted_parameters.tolist()
                for i, param_name in enumerate(timbre_param_names):
                    synth.set_input(param_name, timbre_params[i]) 
                synth.set_input('gate', 1) 
                synth.play()



            # # pred_duration = int(y_pred[-1])
            # # detect rest
            # if all(y_pred[:-1].tolist()) < -4:
            #     # sound_segment = np.zeros(pred_duration)
            #     pass
            # else:
            #     next_latents = torch.tensor(y_pred[:-1]).reshape(1, -1).type(torch.FloatTensor).to(device)
            #     with torch.no_grad():
            #         predicted_parameters = flow_model.regression_model(next_latents) # parameters
            #     predicted_parameters = predicted_parameters[0].detach().cpu().numpy()
            #     # synth.setParameters(param_names, predicted_parameters.tolist())
            #     # sound_segment = synth.forward() # add possibility of 'note off' to handle different durations synth.forward(duration)
            #     # sound_segment = sound_segment[:pred_duration]
            #     print(y_pred[:-1])
            #     print(next_latents)
            #     print(predicted_parameters)
            #     print(pred_duration)

            #     predicted_parameters = predicted_parameters.tolist()
            #     for i, param_name in enumerate(param_names):
            #         synth.set_input(param_name, predicted_parameters[i]) 
            #     synth.trigger() # toDo: trigger with ms latency and generate new patch with auto-free each new synth

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
            tgt_past_events.append(y_pred)
            tgt_past_events_starttimes.append(pred_signal_sample)
            pred_signal_sample += int(pred_duration)

            if len(src_past_events) > 0 and len(tgt_past_events) > 0:

                # get source events in window before current
                input_src = np.array(src_past_events)
                times_src = np.array(src_past_events_starttimes)#.reshape(-1,1)
                window_start_idx = (pred_signal_sample - PRED_TIME_SAMPLES) - WINDOW_N_SAMPLES
                indices = np.where((times_src >= window_start_idx) & (times_src < pred_signal_sample))
                x_src_window_times = (pred_signal_sample - PRED_TIME_SAMPLES) - times_src[indices].reshape(-1,1) # add time difference index
                input_src_array = np.concatenate((input_src[indices], x_src_window_times.reshape(-1,1)), axis=1)[-training_parameters['max_sequence_length']:,:]

                # update sequences
                input_tgt = np.array(tgt_past_events)
                times_tgt = np.array(tgt_past_events_starttimes)#.reshape(-1,1)
                indices = np.where((times_tgt >= window_start_idx) & (times_tgt < pred_signal_sample))
                x_tgt_window_times = (pred_signal_sample - PRED_TIME_SAMPLES) - times_tgt[indices].reshape(-1,1) # add time difference index
                input_tgt_array = np.concatenate((input_tgt[indices], x_tgt_window_times.reshape(-1,1)), axis=1)[-training_parameters['max_sequence_length']:,:]

            # samples countdown
            tgt_samples_left = pred_duration - (FFT_window - (tgt_samples_left - inference_time_samples))
        else:
            tgt_samples_left -= hop_size 

        # check every 
        time.sleep(update_interval_s)

    graph.wait()
