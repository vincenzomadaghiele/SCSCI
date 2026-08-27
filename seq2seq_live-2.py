import os
import time
import json
import joblib
import librosa
import numpy as np
import pandas as pd
import math

import torch

from signalflow import * 
from CONSTANTS import * 

from synth.patches.timbreTheremin import TimbreTheremin
from synth.patches.FM import FM
from synth.patches.rung import Rung
from synth.patches.granular import Granular
from synth.patches.benjolin import Benjolin
from transformer.transformerDiscreteT2V import TransformerDiscreteT2V
from VAE_train import Encoder, Decoder, Model
from corpus.extract_events import detectOnsets
from seq2seq_resynthesize import melToVAElatents

import warnings
warnings.filterwarnings('ignore')

## THIS ONE IS THE BASELINE

def adjustAmplitude(signal, FFT_window_size, hop_size, signal_max=None):
    rms = librosa.feature.rms(y=signal, frame_length=FFT_window_size, hop_length=hop_size)
    if rms.max() < 0.7: # normalize amplitude
        if signal_max:
            signal *= 0.6/signal_max
        else:
            signal *= 0.6/signal.max()
    return signal


if __name__ == "__main__":

    # LOAD VAE FOR SRC ENCODING
    DEVICE = torch.device("cpu")
    corpus_name = 'petter-study'
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

    # LOAD NORMALIZING FLOW MODEL FOR TGT ENCODING AND DECODING
    SYNTH_NAME = 'FM'
    useTimbre = True
    quantizePitch = True
    flow_model_name = 'vae_flow_mel_mse_0_gated_cnn_planar_flow_post_maf_3'
    model_path = f'./synth/trained_flows/{SYNTH_NAME}/models'
    flow_model = torch.load(f'{model_path}/{flow_model_name}.model', weights_only=False, map_location=DEVICE).to(DEVICE)
    flow_model.eval()

    # LOAD TRANSFORMER MODEL TO COMPUTE RESPONSES
    transformer_name = '1787764834-transformerDiscreteT2V'
    model_dir = f'transformer/trained/{corpus_name}_{SYNTH_NAME}/{transformer_name}'
    with open(f'{model_dir}/training_config.json', 'r') as f:
        training_parameters = json.load(f)

    # ESTABLISH FEATURES FOR TRANSFORMER
    N_latents_src = training_parameters['N_latents_src']
    N_latents_tgt = training_parameters['N_latents_tgt']
    features_source = training_parameters['features_source']
    features_target = training_parameters['features_target']

    # TRANSFORMER
    model_load_path = f'{model_dir}/best_model.pth'
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
    ).to(DEVICE)
    model.load_state_dict(torch.load(model_load_path, weights_only=True, map_location=torch.device('cpu')))
    model.eval()
    src_scaler = joblib.load(f'{model_dir}/source_scaler.pkl')
    tgt_scaler = joblib.load(f'{model_dir}/target_scaler.pkl')
    src_time_scaler = joblib.load(f'{model_dir}/source_time_scaler.pkl')
    tgt_time_scaler = joblib.load(f'{model_dir}/target_time_scaler.pkl')

    # START AUDIO GRAPH
    config = AudioGraphConfig()
    config.sample_rate = sr
    config.output_buffer_size = hop_size
    config.output_device_name = "UltraLite-mk5"
    config.input_device_name = "UltraLite-mk5"
    # config.output_device_name = "Scarlett 2i2 USB"
    # config.input_device_name = "Scarlett 2i2 USB"
    graph = AudioGraph(config)
    audio_in = AudioIn(1)
    audio_in = audio_in[0]
    panned_audio_in = StereoPanner( audio_in, pan=1)
    graph.play(panned_audio_in) 
    impulse = StereoPanner( Impulse(frequency=0), pan=0)
    graph.play(impulse) 

    # LOAD SYNTHESIZER
    if SYNTH_NAME == 'TimbreTheremin':
        synth = TimbreTheremin() 
    elif SYNTH_NAME == 'FM':
        synth = FM() 
    elif SYNTH_NAME == 'Rung':
        synth = Rung(graph) 
    elif SYNTH_NAME == 'Benjolin':
        synth = Benjolin(graph) 
    elif SYNTH_NAME == 'Granular':
        sample_path = "./synth/patches/samples/tibetan-ball.wav"
        sample, _ = librosa.load(sample_path, sr=graph.sample_rate, mono=True)
        pitch = librosa.yin(sample, fmin=100, fmax=11000, sr=graph.sample_rate, 
                            frame_length=graph.output_buffer_size*4, hop_length=graph.output_buffer_size)
        pitch = [p for p in pitch.tolist() if p != graph.sample_rate]
        base_pitch_hz = float(np.array(pitch).mean())
        synth = Granular(sample_path, base_pitch_hz) 
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

    if useTimbre:
        N_latents_tgt = training_parameters['N_latents_tgt']
    else:
        N_latents_tgt = 0

    # BUFFER FOR RECORDING SOUND
    update_interval_s = hop_size / sr
    input_buffer = Buffer(1, FFT_window)
    min_feedback_writer_delay = graph.output_buffer_size / graph.sample_rate # samples
    # graph.add_node(FeedbackBufferWriter(input_buffer, audio_in, min_feedback_writer_delay))
    graph.add_node(HistoryBufferWriter(input_buffer, audio_in*2))

    # ARRAYS FOR SAVING INPUTS AND OUTPUTS
    WINDOW_N_SAMPLES = training_parameters['window_size_s'] * sr
    input_src_array = np.zeros((1,len(features_source)+1))
    input_tgt_array = np.zeros((1,len(features_target)+1))

    # CONSTANTS FOR GENERATION
    PRED_TIME_SAMPLES = 0
    pred_signal_sample = 0
    src_signal_sample = 0
    model_inference_time_s = 0.003
    inference_time_samples = model_inference_time_s * sr
    tgt_samples_left = 0
    silence_threshold = 0.003 # SILENCE THREHSOLD NEEDS TO BE ADAPTED TO LIVE SITUATION

    # UTILS FUNCTION
    def detectRest(event):
        rms = librosa.feature.rms(y=event, frame_length=FFT_window, hop_length=hop_size)[0]
        end_event_frame = -1
        for k in range(rms.shape[0]):
            if rms[k] < silence_threshold:
                end_event_frame = k
                break
        if end_event_frame != -1: 
            end_event_sample = librosa.frames_to_samples(end_event_frame, hop_length=hop_size)
            return [event[:end_event_sample], event[end_event_sample:]]
        else:
            return [event]

    timbre_pitch = True
    def extractEventFeats(signal, time=0):
        # compute features of event
        rms = librosa.feature.rms(y=signal, frame_length=FFT_window, hop_length=hop_size)
        pitch = librosa.yin(signal, fmin=50, fmax=11000, sr=sr, frame_length=FFT_window, hop_length=hop_size)
        event_duration = signal.shape[0]
        if timbre_pitch:
            # transpose to base freq for mel computation
            BASE_FREQ = 220
            base_pitch = librosa.hz_to_midi(BASE_FREQ)
            this_pitch = librosa.hz_to_midi(pitch.mean())
            signal = librosa.effects.pitch_shift(signal, sr=sr, n_steps=this_pitch-base_pitch)
        mel = librosa.feature.melspectrogram(y=signal, sr=sr, n_fft=FFT_window, n_mels=num_MEL, hop_length=hop_size, fmin=50, fmax=11000)
        mel = mel[:num_MEL, :melspec_size_flow]
        event_latents = melToVAElatents(mel, VAE_model)
        event_latents = event_latents.tolist()
        # event_latents = [] # remove this if using latents
        event_latents.append(rms.mean()) 
        event_latents.append(pitch.mean()) 
        event_latents.append(np.clip(event_duration, 882, 88200)) 
        features = np.array(event_latents)
        print(f'[{time/sr:.3f}] HUMAN EVENT | rms: {rms.mean():.3f}, pitch: {pitch.mean():.3f}, duration: {event_duration / sr:.3f}')
        return features

    def makeRest(signal, time=0):
        event_duration = np.clip(signal.shape[0], 882, 88200)
        event_latents = np.zeros(latent_dim).tolist()
        # event_latents = [] # remove this if using latents
        event_latents.append(0) 
        event_latents.append(0) 
        event_latents.append(event_duration) 
        features = np.array(event_latents)
        print(f'[{time/sr:.3f}] HUMAN REST  | rms: {0}, pitch: {0}, duration: {event_duration / sr:.3f}')
        return features


    # log events
    event_samples = np.zeros(hop_size)
    start_event_sample = 0
    prev_onset_position = 0
    found_onsets_general = []
    src_past_events = []
    src_past_events_starttimes = []
    tgt_past_events = []
    tgt_past_events_starttimes = []

    tot_events = []
    tot_src_signal = []
    try:
        # save files
        timestamp = int(time.time())
        save_dir = f'{model_dir}/01_live_recordings/{timestamp}-baseline'
        os.makedirs(save_dir, exist_ok=True)
        graph.start_recording(f'{save_dir}/performance.wav', num_channels=2)
        start_time = time.time()
        count_frames = 0

        while True:
            signal = input_buffer.data[0,:].copy()
            # print(signal.shape)
            count_frames += hop_size
            tot_src_signal.append(signal[-hop_size:])

            time_start = time.time()
            if count_frames > FFT_window:

                rms = librosa.feature.rms(y=signal, frame_length=FFT_window, hop_length=hop_size)[0]
                # MAKE ADAPTIVE ADJUST AMP
                # if rms.mean() > silence_threshold:
                #     adj_signal = adjustAmplitude(signal, FFT_window, hop_size)
                # else: 
                #     adj_signal = np.zeros_like(signal)
                # onsets_samples = detectOnsets(adj_signal, sr, min_event_duration_samples=min_event_duration_samples)
                onsets_samples = detectOnsets(signal, sr, min_event_duration_samples=min_event_duration_samples)

                # check if onset already found in previous windows
                alreadyFound = False
                newOnsets = []
                for onset in onsets_samples:
                    general_onset_position = count_frames - FFT_window + onset
                    for found_onset in found_onsets_general:
                        if general_onset_position < found_onset + min_event_duration_samples and general_onset_position > found_onset - min_event_duration_samples:
                            alreadyFound = True
                    if not alreadyFound:
                        found_onsets_general.append(general_onset_position)
                        if general_onset_position > prev_onset_position - min_event_duration_samples:
                            newOnsets.append(onset)

                event_samples = np.concatenate((event_samples, signal[-hop_size:].reshape(-1)))
                if newOnsets:
                    # graph.play(Impulse(frequency=0)) 
                    # event before onset
                    event = event_samples[:newOnsets[0]-FFT_window]
                    # (1) BEFORE THE ONSET
                    event_with_rest = detectRest(event)
                    if len(event_with_rest) == 2:
                        if event_with_rest[0].shape[0] == 0:
                            # (1.1) end of a rest
                            event_to_save = event_with_rest[1]
                            if event_to_save.shape[0] > 0:
                                tot_events.append(event_to_save)
                                src_past_events.append(makeRest(event_to_save, time=count_frames))
                                src_past_events_starttimes.append(start_event_sample)
                                start_event_sample += event_to_save.shape[0]
                        else:
                            # (1.2) end of previous event + rest
                            event_to_save = event_with_rest[0]
                            if event_to_save.shape[0] > 0:
                                tot_events.append(event_to_save)
                                src_past_events.append(extractEventFeats(event_to_save, time=count_frames))
                                src_past_events_starttimes.append(start_event_sample)
                                start_event_sample += event_to_save.shape[0]
                            event_to_save = event_with_rest[1]
                            if event_to_save.shape[0] > 0:
                                tot_events.append(event_to_save)
                                src_past_events.append(makeRest(event_to_save))
                                src_past_events_starttimes.append(start_event_sample)
                                start_event_sample += event_to_save.shape[0]
                    else:
                        # (1.3) end of previous event coincides with new onset (no rest)
                        event_to_save = event_with_rest[0]
                        if event_to_save.shape[0] > 0:
                            tot_events.append(event_to_save)
                            src_past_events.append(extractEventFeats(event_to_save, time=count_frames))
                            src_past_events_starttimes.append(start_event_sample)
                            start_event_sample += event_to_save.shape[0]

                    # (2) AFTER THE ONSET
                    event = event_samples[newOnsets[0]-FFT_window:]
                    event_with_rest = detectRest(event)
                    if len(event_with_rest) == 2:
                        # (2.1) event + rest that keeps going into next window
                        event_to_save = event_with_rest[0]
                        if event_to_save.shape[0] > 0:
                            tot_events.append(event_to_save)
                            src_past_events.append(extractEventFeats(event_to_save, time=count_frames))
                            src_past_events_starttimes.append(start_event_sample)
                            start_event_sample += event_to_save.shape[0]
                        event_samples = np.zeros_like(event_with_rest[1])
                    else:
                        # (2.2) event keeps going into next window
                        event_samples = event_with_rest[0]
                else:
                    # (3) NO ONSET
                    event_with_rest = detectRest(event_samples)
                    if len(event_with_rest) == 2:
                        # (3.1) event + rest that keeps going into next window
                        if event_with_rest[0].shape[0] > 0:
                            event_to_save = event_with_rest[0]
                            if event_to_save.shape[0] > 0:
                                tot_events.append(event_to_save)
                                src_past_events.append(extractEventFeats(event_to_save, time=count_frames))
                                src_past_events_starttimes.append(start_event_sample)
                                start_event_sample += event_to_save.shape[0]
                            event_samples = np.zeros_like(event_with_rest[1])
                        else:
                            event_samples = event_with_rest[1]
                            if event_samples.shape[0] > max_event_duration_samples:
                                # (3.2) rest too long: cut and next keeps going into next window
                                event_to_save = event_samples[:max_event_duration_samples - event_samples.shape[0]]
                                if event_to_save.shape[0] > 0:
                                    tot_events.append(event_to_save)
                                    src_past_events.append(makeRest(event_to_save, time=count_frames))
                                    src_past_events_starttimes.append(start_event_sample)
                                    start_event_sample += event_to_save.shape[0]
                                event_samples = np.zeros_like(event_samples[max_event_duration_samples - event_samples.shape[0]:])
                            else:
                                # (3.3) rest previous that keeps going into next window
                                event_samples = np.zeros_like(event_samples)
                    else:
                        event_samples = event_with_rest[0]
                        if event_samples.shape[0] > max_event_duration_samples:
                            # (3.4) event too long: cut and next keeps going into next window
                            event_to_save = event_samples[:max_event_duration_samples - event_samples.shape[0]]
                            if event_to_save.shape[0] > 0:
                                tot_events.append(event_to_save)
                                src_past_events.append(extractEventFeats(event_to_save, time=count_frames))
                                src_past_events_starttimes.append(start_event_sample)
                                start_event_sample += event_to_save.shape[0]
                            event_samples = event_samples[max_event_duration_samples - event_samples.shape[0]:]
                        else:
                            # (3.5) event previous that keeps going into next window
                            pass

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
                # input_src_array_norm = torch.tensor(input_src_array_norm.reshape(1, input_src_array_norm.shape[0], input_src_array_norm.shape[1])).type(torch.FloatTensor).to(DEVICE)
                # input_tgt_array_norm = torch.tensor(input_tgt_array_norm.reshape(1, input_tgt_array_norm.shape[0], input_tgt_array_norm.shape[1])).type(torch.FloatTensor).to(DEVICE)
                # start_model_time = time.time()
                # with torch.no_grad():
                #     src_pad_mask = model.create_pad_mask(input_src_array_norm, pad_token=training_parameters['pad_token']).to(DEVICE).to(torch.bool)
                #     tgt_pad_mask = model.create_pad_mask(input_tgt_array_norm, pad_token=training_parameters['pad_token']).to(DEVICE).to(torch.bool)
                #     # tgt_mask = model.get_tgt_mask(input_tgt_array_norm.size(1)).to(device)
                #     y = model(input_src_array_norm, input_tgt_array_norm, src_pad_mask=src_pad_mask[:,:,0], tgt_pad_mask=tgt_pad_mask[:,:,0]).detach().cpu().numpy()
                # # y_denorm = tgt_scaler.inverse_transform(y.reshape(y.shape[0], y.shape[2]))
                # y = (input_src_array_norm.mean(axis=0) + input_tgt_array_norm.mean(axis=0)) / 2
                y = (input_tgt_array_norm.mean(axis=0)[:-1] + (2* np.random.rand(1, input_tgt_array_norm.shape[1]-1) -1 ))
                y_pred = tgt_scaler.inverse_transform(y).reshape(-1)

                # N_latents_tgt = 0
                pred_latents = y_pred[:N_latents_tgt]
                pred_rms = y_pred[N_latents_tgt]
                pred_frequency = np.clip(y_pred[N_latents_tgt+1], 50, 11050)
                pred_duration = int(np.clip(y_pred[N_latents_tgt+2], 882, 88200))

                # quantize pitch
                if quantizePitch:
                    MIDI = int(69 + 12 * math.log2(pred_frequency/440))
                    pred_frequency = 440 * 2**((MIDI - 69)/12) # tune to scale

                y_pred[N_latents_tgt] = pred_rms
                y_pred[N_latents_tgt+1] = pred_frequency
                y_pred[N_latents_tgt+2] = pred_duration
            
                # print(y.reshape(-1), y_pred)

                if pred_rms < 0.0001:
                    sound_segment = np.zeros(pred_duration)
                    y_pred[:N_latents_tgt] = np.zeros_like(y_pred[:N_latents_tgt])
                    y_pred[N_latents_tgt] = 0
                    y_pred[N_latents_tgt+1] = 0
                    # y_pred = np.zeros_like(y_pred)
                    print(f'[{count_frames/sr:.3f}] SYNTH REST  | rms: {0:.3f}, pitch: {0:.3f}, duration: {pred_duration / sr:.3f}')
                else:
                    if SYNTH_NAME == 'TimbreTheremin':
                        synth = TimbreTheremin() 
                    elif SYNTH_NAME == 'FM':
                        synth = FM() 
                    # elif SYNTH_NAME == 'Rung':
                    #     synth = Rung(graph) 
                    # elif SYNTH_NAME == 'Benjolin':
                    #     synth = Benjolin(graph) 
                    elif SYNTH_NAME == 'Granular':
                        sample_path = "./synth/patches/samples/tibetan-ball.wav"
                        sample, _ = librosa.load(sample_path, sr=graph.sample_rate, mono=True)
                        pitch = librosa.yin(sample, fmin=100, fmax=11000, sr=graph.sample_rate, 
                                            frame_length=graph.output_buffer_size*4, hop_length=graph.output_buffer_size)
                        pitch = [p for p in pitch.tolist() if p != graph.sample_rate]
                        base_pitch_hz = float(np.array(pitch).mean())
                        synth = Granular(sample_path, base_pitch_hz)
                    synth.set_input('frequency', pred_frequency) 
                    synth.set_input('gain', float(pred_rms*RMS_multiplier))
                    for i, param_name in enumerate(envelope_param_names):
                        synth.set_input(param_name, ADSR_standard[i]) 
                    if useTimbre:
                        next_latents = torch.tensor(pred_latents).reshape(1, -1).type(torch.FloatTensor).to(DEVICE)
                        with torch.no_grad():
                            predicted_parameters = flow_model.regression_model(next_latents) # parameters
                        predicted_parameters = predicted_parameters[0].detach().cpu().numpy()
                        timbre_params = predicted_parameters.tolist()
                    else:
                        timbre_params = [0.3, 0.3, 0.2, 0.1]
                    for i, param_name in enumerate(timbre_param_names):
                        synth.set_input(param_name, timbre_params[i]) 
                    synth.set_input('gate', 1) 
                    if SYNTH_NAME == 'TimbreTheremin' or SYNTH_NAME == 'Granular' or SYNTH_NAME == 'FM':
                        synth.play()
                    print(f'[{count_frames/sr:.3f}] SYNTH EVENT | rms: {pred_rms:.3f}, pitch: {pred_frequency:.3f}, duration: {pred_duration / sr:.3f}')

                end_lookup_time = time.time()
                # append event to event log
                tgt_past_events.append(y_pred)
                pred_signal_sample = count_frames
                tgt_past_events_starttimes.append(pred_signal_sample)
                # pred_signal_sample += int(pred_duration)

                if len(src_past_events) > 0 and len(tgt_past_events) > 0:
                    window_start_idx = (pred_signal_sample - PRED_TIME_SAMPLES) - WINDOW_N_SAMPLES

                    # get source events in window before current
                    input_src = np.array(src_past_events)
                    times_src = np.array(src_past_events_starttimes)#.reshape(-1,1)
                    indices = np.where((times_src >= window_start_idx) & (times_src < pred_signal_sample))
                    x_src_window_times = (pred_signal_sample - PRED_TIME_SAMPLES) - times_src[indices].reshape(-1,1) # add time difference index
                    input_src_array = np.concatenate((input_src[indices], x_src_window_times.reshape(-1,1)), axis=1)[-training_parameters['max_sequence_length']:,:]
                    if input_src_array.shape[0] == 0:
                        input_src_array = np.zeros(N_latents_src+2).tolist() + [882, training_parameters['window_size_s']-1]
                        # input_src_array = np.zeros(2).tolist() + [882, 10]
                        input_src_array = np.array(input_src_array).reshape(1,-1)

                    # update sequences
                    input_tgt = np.array(tgt_past_events)
                    times_tgt = np.array(tgt_past_events_starttimes)#.reshape(-1,1)
                    indices = np.where((times_tgt >= window_start_idx) & (times_tgt < pred_signal_sample))
                    x_tgt_window_times = (pred_signal_sample - PRED_TIME_SAMPLES) - times_tgt[indices].reshape(-1,1) # add time difference index
                    input_tgt_array = np.concatenate((input_tgt[indices], x_tgt_window_times.reshape(-1,1)), axis=1)[-training_parameters['max_sequence_length']:,:]
                    if input_tgt_array.shape[0] == 0:
                        input_tgt_array = np.zeros(N_latents_tgt+2).tolist() + [882, training_parameters['window_size_s']-1]
                        # input_tgt_array = np.zeros(2).tolist() + [882, 10]
                        input_tgt_array = np.array(input_tgt_array).reshape(1,-1)

                # samples countdown
                tgt_samples_left = pred_duration - (FFT_window - (tgt_samples_left - inference_time_samples))
            else:
                tgt_samples_left -= hop_size 


            time_end = time.time()
            time.sleep( update_interval_s - update_interval_s * 0.30) # reduce 30% time (not sure why..)

    except KeyboardInterrupt:
        
        graph.stop_recording()
        end_time = time.time()
        print(f'total performance time: {end_time-start_time}')

        columns = [f'latent-{j}' for j in range(N_latents_src)]
        # columns = []
        columns.append('rms')
        columns.append('pitch')
        columns.append('event_duration')
        df_src = pd.DataFrame(src_past_events, columns=columns)
        df_src['event_start'] = np.array(src_past_events_starttimes) + 0.012 * sr
        df_src.to_csv(f"{save_dir}/src_events.csv")

        columns = [f'latent-{j}' for j in range(N_latents_tgt)]
        # columns = []
        columns.append('rms')
        columns.append('pitch')
        columns.append('event_duration')
        df_tgt = pd.DataFrame(tgt_past_events, columns=columns)
        df_tgt['event_start'] = np.array(tgt_past_events_starttimes) + 0.012 * sr
        df_tgt.to_csv(f"{save_dir}/tgt_events.csv")

        # Load stereo file (mono=False keeps both channels)
        signal, _ = librosa.load(f'{save_dir}/performance.wav', mono=False)
        import soundfile as sf
        sf.write(f'{save_dir}/performance_src.wav', signal[1], sr)
        sf.write(f'{save_dir}/performance_tgt.wav', signal[0], sr)

        sf.write(f'{save_dir}/performance_src_events.wav', np.concatenate(tot_events), sr)
        sf.write(f'{save_dir}/performance_src_all.wav', np.concatenate(tot_src_signal), sr)

        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(2, figsize=(10, 4))
        ax[0].plot(signal[1])
        ax[1].plot(np.concatenate(tot_events))
        for ii in range(len(src_past_events_starttimes)):
            ax[0].axvline(src_past_events_starttimes[ii], -1, 1, color='red', ls='--', lw=0.1)
            ax[1].axvline(src_past_events_starttimes[ii], -1, 1, color='red', ls='--', lw=0.1)
        ax[0].set_ylim([-1,1])
        ax[1].set_ylim([-1,1])
        plt.savefig(f'{save_dir}/performance_src_events.png')
        plt.close(fig)

        from corpus.extract_events import extractEvents, plotEvents, plotPairedEvents
        df_src['filename'] = [f'{save_dir}/performance_src.wav' for _ in range(df_src.values.shape[0])]
        plotEvents(df_src, sr, save_dir=save_dir, isDfin=True)
        df_tgt['filename'] = [f'{save_dir}/performance_tgt.wav' for _ in range(df_tgt.values.shape[0])]
        plotEvents(df_tgt, sr, save_dir=save_dir, isDfin=True)
        plotPairedEvents(df_src, df_tgt, sr, save_dir=save_dir, isDfin=True)

        # PLOT PATH ON THE VAE TIMBRE SPACE
        from seq2seq_resynthesize import plotFlowTimbre, plotVAETimbre
        VAE_latents = df_src[[f'latent-{j}' for j in range(N_latents_src)]].values
        X = np.load(f'./corpus/{corpus_name}/trained_VAE/latents.npy')
        plotVAETimbre(X, VAE_latents, save_dir)
        flow_latents = df_tgt[[f'latent-{j}' for j in range(N_latents_tgt)]].values
        X = np.load(f'./synth/trained_flows/{SYNTH_NAME}/visualize/{flow_model_name}/latents.npy')
        plotFlowTimbre(X, flow_latents, save_dir)

        print('Finished performance')