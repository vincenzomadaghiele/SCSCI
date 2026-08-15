import time
import librosa
import numpy as np

import torch

from signalflow import * 
from CONSTANTS import * 

from VAE_train import Encoder, Decoder, Model
from corpus.extract_events import detectOnsets
from resynthesize import melToVAElatents


def adjustAmplitude(signal, FFT_window_size, hop_size, signal_max=None):
    rms = librosa.feature.rms(y=signal, frame_length=FFT_window_size, hop_length=hop_size)
    if rms.max() < 0.7: # normalize amplitude
        if signal_max:
            signal *= 0.6/signal_max
        else:
            signal *= 0.6/signal.max()
    return signal


if __name__ == "__main__":

    # START AUDIO GRAPH
    config = AudioGraphConfig()
    config.sample_rate = sr
    config.output_buffer_size = hop_size
    config.output_device_name = "UltraLite-mk5"
    config.input_device_name = "UltraLite-mk5"
    graph = AudioGraph(config)
    audio_in = AudioIn() 
    impulse = Impulse(frequency=0)
    graph.play(impulse) 
    graph.play([audio_in, audio_in]) 

    # BUFFER FOR RECORDING SOUND
    update_interval_s = hop_size / sr
    input_buffer = Buffer(1, FFT_window)
    min_feedback_writer_delay = graph.output_buffer_size / graph.sample_rate # samples
    graph.add_node(FeedbackBufferWriter(input_buffer, audio_in, min_feedback_writer_delay))

    # LOAD VAE FOR SRC ENCODING
    DEVICE = torch.device("cpu")
    corpus_name = 'guitar-duo'
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
    def extractEventFeats(signal):
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
        event_latents = [] # remove this if using latents
        event_latents.append(rms.mean()) 
        event_latents.append(pitch.mean()) 
        event_latents.append(np.clip(event_duration, 882, 88200)) 
        features = np.array(event_latents)
        print(f'EVENT | rms: {rms.mean():.3f}, pitch: {pitch.mean():.3f}, duration: {event_duration / sr:.3f}')
        return features

    def makeRest(signal):
        event_duration = signal.shape[0]
        event_latents = np.zeros(latent_dim).tolist()
        event_latents.append(0) 
        event_latents.append(0) 
        event_latents.append(np.clip(event_duration, 882, 88200)) 
        features = np.array(event_latents)
        print(f'REST  | rms: {0}, pitch: {0}, duration: {event_duration / sr:.3f}')
        return features


    event_samples = np.zeros(hop_size)
    tot_events = []
    tot_features = []
    tot_event_start_samples = []
    found_onsets_general = []
    start_event_sample = 0
    prev_onset_position = 0
    count_frames = 0
    while True:
        signal = input_buffer.data[0,:]
        # print(signal.shape)
        count_frames += hop_size

        if count_frames > FFT_window:

            rms = librosa.feature.rms(y=signal, frame_length=FFT_window, hop_length=hop_size)[0]
            # if rms.mean() > silence_threshold / 2:
            #     signal = adjustAmplitude(signal, FFT_window, hop_size)
            # else: 
            #     signal = np.zeros_like(signal)
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
                graph.play(Impulse(frequency=0)) 

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
                            tot_features.append(makeRest(event_to_save))
                            start_event_sample += event_to_save.shape[0]
                            tot_event_start_samples.append(start_event_sample)
                    else:
                        # (1.2) end of previous event + rest
                        event_to_save = event_with_rest[0]
                        if event_to_save.shape[0] > 0:
                            tot_events.append(event_with_rest[0])
                            tot_features.append(extractEventFeats(event_with_rest[0]))
                            start_event_sample += event_with_rest[0].shape[0]
                            tot_event_start_samples.append(start_event_sample)
                        event_to_save = event_with_rest[1]
                        if event_to_save.shape[0] > 0:
                            tot_events.append(event_to_save)
                            tot_features.append(makeRest(event_to_save))
                            start_event_sample += event_to_save.shape[0]
                            tot_event_start_samples.append(start_event_sample)
                else:
                    # (1.3) end of previous event coincides with new onset (no rest)
                    event_to_save = event_with_rest[0]
                    if event_to_save.shape[0] > 0:
                        tot_events.append(event_to_save)
                        tot_features.append(extractEventFeats(event_to_save))
                        start_event_sample += event_to_save.shape[0]
                        tot_event_start_samples.append(start_event_sample)

                # (2) AFTER THE ONSET
                event = event_samples[newOnsets[0]-FFT_window:]
                event_with_rest = detectRest(event)
                if len(event_with_rest) == 2:
                    # (2.1) event + rest that keeps going into next window
                    event_to_save = event_with_rest[0]
                    if event_to_save.shape[0] > 0:
                        tot_events.append(event_to_save)
                        tot_features.append(extractEventFeats(event_to_save))
                        start_event_sample += event_to_save.shape[0]
                        tot_event_start_samples.append(start_event_sample)
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
                            tot_features.append(extractEventFeats(event_to_save))
                            start_event_sample += event_to_save.shape[0]
                            tot_event_start_samples.append(start_event_sample)
                        event_samples = np.zeros_like(event_with_rest[1])
                    else:
                        event_samples = event_with_rest[1]
                        if event_samples.shape[0] > max_event_duration_samples:
                            # (3.2) rest too long: cut and next keeps going into next window
                            event_to_save = event_samples[:max_event_duration_samples - event_samples.shape[0]]
                            if event_to_save.shape[0] > 0:
                                tot_events.append(event_to_save)
                                tot_features.append(makeRest(event_to_save))
                                start_event_sample += event_to_save.shape[0]
                                tot_event_start_samples.append(start_event_sample)
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
                            tot_features.append(extractEventFeats(event_to_save))
                            start_event_sample += event_to_save.shape[0]
                            tot_event_start_samples.append(start_event_sample)
                        event_samples = event_samples[max_event_duration_samples - event_samples.shape[0]:]
                    else:
                        # (3.5) event previous that keeps going into next window
                        pass

        time.sleep(update_interval_s)
