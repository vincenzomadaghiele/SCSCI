import os
import json
import time
import joblib
import librosa
import numpy as np
import pandas as pd
import soundfile as sf
import matplotlib.pyplot as plt

import torch
from torchvision import transforms

from CONSTANTS import * 
from corpus.extract_events import extractEvents, plotEvents, plotPairedEvents, detectOnsets, adjustAmplitude
from VAE_train import Encoder, Decoder, Model

from corpus.onset_detect import OnsetDetection


if __name__ == '__main__': 

    # device 
    DEVICE = torch.device("cpu")

    corpus_name = 'guitar-duo'
    src_track_name = 'guitar1'
    SYNTH_NAME = 'TimbreTheremin'
    transformer_name = '1786739157-transformerDiscreteT2V'
    song_name = 'recording3'

    # LOAD AUDIO FILE
    src_audio_filepath = f'corpus/{corpus_name}/{song_name}/{src_track_name}.wav'
    src_signal, _ = librosa.load(src_audio_filepath, sr=sr, mono=True)

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

    # onsetDetector = OnsetDetection(sr=sr)
    # onsets_samples = onsetDetector(src_signal.reshape(1,-1))
    # print(onsets_samples)

    src_signal = adjustAmplitude(src_signal, FFT_window, hop_size)
    onsets_samples = detectOnsets(src_signal, sr, min_event_duration_samples=min_event_duration_samples)
    print(onsets_samples)

    # fig, ax = plt.subplots(1, figsize=(8, 3))
    # librosa.display.waveshow(y=src_signal, sr=sr, color="black", axis='time', ax=ax, alpha=0.2)
    # for i in range(len(onsets_samples)):
    #     ax.axvline(onsets_samples[i] / sr,-1,1, color='red', ls='--')
    # plt.show()
    # plt.close(fig)


    # src_signal, _ = librosa.load(src_audio_filepath, sr=sr, mono=True)
    # src_signal = src_signal.reshape(-1)
    # event_samples = np.zeros(hop_size)
    # tot_duration = src_signal.shape[0]
    # events = []
    # for i in range(FFT_window, tot_duration, hop_size):
    #     signal = src_signal[i-FFT_window:i] # get window
    #     event_samples = np.concatenate((event_samples, signal[-hop_size:])) # add new part to latest event
    #     event_samples = adjustAmplitude(event_samples, FFT_window, hop_size) # adjust amplitude
    #     onsets_samples = detectOnsets(event_samples, sr, min_event_duration_samples=min_event_duration_samples) # look for onsets in whole event

    #     previous_onset = 0
    #     for j in range(len(onsets_samples)-1):
    #         event = event_samples[previous_onset:onsets_samples[j]]
    #         # check for rest
    #         rms = librosa.feature.rms(y=event, frame_length=FFT_window, hop_length=hop_size)[0]
    #         if any(r > silence_threshold for r in rms.tolist()):
    #             end_event_frame = -1
    #             for k, rms_value in enumerate(rms.tolist()):
    #                 if rms_value > silence_threshold:
    #                     end_event_frame = k
    #                     break
    #             if end_event_frame != -1: 
    #                 end_event_sample = librosa.frames_to_samples(end_event_frame, hop_length=hop_size)
    #                 # rest was found
    #                 note_event = event[:end_event_sample]
    #                 rest_event = np.zeros(event[end_event_sample:].shape[0])
    #                 events.append(note_event) # append note event
    #                 events.append(rest_event) # append rest because there is another onset afterwards
    #             else:
    #                 events.append(event)
    #         else:
    #             events.append(np.zeros(event.shape[0]))
    #         previous_onset = onsets_samples[j]

    #     if len(onsets_samples) > 0:
    #         event = event_samples[previous_onset:onsets_samples[-1]]
    #         # check for rest
    #         rms = librosa.feature.rms(y=event, frame_length=FFT_window, hop_length=hop_size)[0]
    #         end_event_frame = -1
    #         for k, rms_value in enumerate(rms.tolist()):
    #             if rms_value > silence_threshold:
    #                 end_event_frame = k
    #                 break
    #         if end_event_frame != -1: 
    #             end_event_sample = librosa.frames_to_samples(end_event_frame, hop_length=hop_size)
    #             # last unfinished event will be the rest
    #             note_event = event[:end_event_sample]
    #             events.append(note_event) 
    #             event_samples = np.zeros(event[end_event_sample:].shape[0])
    #         else:
    #             # last unfinished event will be the note
    #             event_samples = event[onsets_samples[-1]:]
    #     else:
    #         # there are no onsets: are we in a rest or the sustain of a previous event?
    #         rms = librosa.feature.rms(y=event_samples, frame_length=FFT_window, hop_length=hop_size)[0]
    #         if all(r < silence_threshold for r in rms.tolist()):
    #             # we are in a rest
    #             event_samples[-hop_size:] = np.zeros(hop_size)

    #     # check if max duration reached
    #     if event_samples.shape[0] > max_event_duration_samples:
    #         events.append(event_samples[:max_event_duration_samples]) 
    #         event_samples = event_samples[max_event_duration_samples:]

    # for event in events:
    #     fig, ax = plt.subplots(1, figsize=(6, 3))
    #     ax.plot(event)
    #     ax.set_ylim([-1,1])
    #     plt.show()
    #     plt.close(fig)

    def detectRest(event):
        rms = librosa.feature.rms(y=event, frame_length=FFT_window, hop_length=hop_size)[0]
        end_event_frame = -1
        for k in range(rms.shape[0]):
            if rms[k] < silence_threshold:
                end_event_frame = k
                break
        if end_event_frame != -1: 
            # there is a rest in event before 
            end_event_sample = librosa.frames_to_samples(end_event_frame, hop_length=hop_size)
            return [event[:end_event_sample], event[end_event_sample:]]
        else:
            return [event]

    src_signal, _ = librosa.load(src_audio_filepath, sr=sr, mono=True)
    src_signal = src_signal.reshape(-1)
    found_onsets_general = []
    tot_duration = src_signal.shape[0]
    prev_onset_position = 0
    inEvent = False # if window starts in event or in rest
    event_samples = np.zeros(hop_size)
    tot_events = []
    for i in range(FFT_window, tot_duration, hop_size):
        signal = src_signal[i-FFT_window:i]
        signal = adjustAmplitude(signal, FFT_window, hop_size)
        onsets_samples = detectOnsets(signal, sr, min_event_duration_samples=min_event_duration_samples)

        # check if onset already found in previous windows
        alreadyFound = False
        newOnsets = []
        for onset in onsets_samples:
            general_onset_position = i - FFT_window + onset
            for found_onset in found_onsets_general:
                if general_onset_position < found_onset + min_event_duration_samples and general_onset_position > found_onset - min_event_duration_samples:
                    alreadyFound = True
            if not alreadyFound:
                found_onsets_general.append(general_onset_position)
                if general_onset_position > prev_onset_position - min_event_duration_samples:
                    newOnsets.append(onset)

        event_samples = np.concatenate((event_samples, signal[-hop_size:].reshape(-1)))
        if newOnsets:
            # event before onset
            event = event_samples[:newOnsets[0]-FFT_window]

            # (1) BEFORE THE ONSET
            event_with_rest = detectRest(event)
            if len(event_with_rest) == 2:
                if event_with_rest[0].shape[0] == 0:
                    # (1.1) end of a rest
                    print(f'rest duration: {event_with_rest[1].shape[0]}')
                    tot_events.append(event_with_rest[1])
                else:
                    # (1.2) end of previous event + rest
                    print(f'event duration: {event_with_rest[0].shape[0]}')
                    tot_events.append(event_with_rest[0])
                    print(f'rest duration: {event_with_rest[1].shape[0]}')
                    tot_events.append(event_with_rest[1])
            else:
                # (1.3) end of previous event coincides with new onset (no rest)
                print(f'event duration: {event_with_rest[0].shape[0]}')
                tot_events.append(event_with_rest[0])
            
            # (2) AFTER THE ONSET
            event = event_samples[newOnsets[0]-FFT_window:]
            event_with_rest = detectRest(event)
            if len(event_with_rest) == 2:
                # (2.1) event + rest that keeps going into next window
                print(f'event duration: {event_with_rest[0].shape[0]}')
                tot_events.append(event_with_rest[0])
                # event_samples = event_with_rest[1]
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
                    print(f'event duration: {event_with_rest[0].shape[0]}')
                    tot_events.append(event_with_rest[0])
                    # event_samples = event_with_rest[1]
                    event_samples = np.zeros_like(event_with_rest[1])
                else:
                    event_samples = event_with_rest[1]
                    if event_samples.shape[0] > max_event_duration_samples:
                        # (3.2) rest too long: cut and next keeps going into next window
                        tot_events.append(event_samples[:max_event_duration_samples - event_samples.shape[0]])
                        print(f'rest duration: {event_samples[:max_event_duration_samples - event_samples.shape[0]].shape[0]}')
                        # event_samples = event_samples[max_event_duration_samples - event_samples.shape[0]:]
                        event_samples = np.zeros_like(event_samples[max_event_duration_samples - event_samples.shape[0]:])
                    else:
                        # (3.3) rest previous that keeps going into next window
                        # event_samples = event_samples[max_event_duration_samples - event_samples.shape[0]:]
                        event_samples = np.zeros_like(event_samples)
            else:
                event_samples = event_with_rest[0]
                if event_samples.shape[0] > max_event_duration_samples:
                    # (3.4) event too long: cut and next keeps going into next window
                    tot_events.append(event_samples[:max_event_duration_samples - event_samples.shape[0]])
                    print(f'event duration: {event_samples[:max_event_duration_samples - event_samples.shape[0]].shape[0]}')
                    event_samples = event_samples[max_event_duration_samples - event_samples.shape[0]:]
                else:
                    # (3.5) event previous that keeps going into next window
                    pass




        # rms = librosa.feature.rms(y=event_samples, frame_length=FFT_window, hop_length=hop_size)[0]
        # # pitch = librosa.yin(event_samples, fmin=50, fmax=11000, sr=sr, frame_length=FFT_window, hop_length=hop_size)

        # previous_onset = 0
        # newOnsets = [0] + newOnsets + [event_samples.shape[0]]
        # newRestOnsets = []
        # for k in range(len(newOnsets)-1):
        #     latest_event = event_samples[newOnsets[k]:newOnsets[k+1]]

        #     # check for rests
        #     this_rms = rms[librosa.samples_to_frames(newOnsets[k], hop_length=hop_size):librosa.samples_to_frames(newOnsets[k+1], hop_length=hop_size)]
        #     end_event_frame = -1
        #     for k in range(this_rms.shape[0]):
        #         if this_rms[k] < silence_threshold:
        #             end_event_frame = k
        #             break
        #     if end_event_frame != -1: 
        #         # last unfinished event will be the rest
        #         newRestOnsets.append(librosa.frames_to_samples(end_event_frame, hop_length=hop_size))

        # if newOnsets[1:-1]: 
        #     print(newOnsets[1:-1], newRestOnsets)

        # previous_onset = 0
        # for k in range(len(newOnsets)):
        #     if k == 0:
        #         # first event
        #         if inEvent:
        #             # append previous event to sequence
        #             latest_event = event_samples[:newOnsets[k]]
        #             # check for rest at the end of new event
        #             this_rms = rms[:librosa.samples_to_frames(newOnsets[k], hop_length=hop_size)]
        #             end_event_frame = -1
        #             for k, rms_value in enumerate(rms.tolist()):
        #                 if rms_value > silence_threshold:
        #                     end_event_frame = k
        #                     break
        #             if end_event_frame != -1: 
        #                 # last unfinished event will be the rest
        #                 note_event = latest_event[:librosa.frames_to_samples(end_event_frame, hop_length=hop_size)]
        #             # analyze and process
        #             previous_onset = newOnsets[k]
        #             pass
        #         else:
        #             # append rest event to sequence
        #             pass
        #     elif k == len(newOnsets)-1:
        #         # last event
        #         # check for rest 
        #         event_samples = event_samples[newOnsets[k]:]
        #         pass
        #     else:
        #         latest_event = event_samples[previous_onset:newOnsets[k]]
        #         # other onsets
        #         # append previous event to sequence
        #         # check for rest at the end of new event
        #         previous_onset = newOnsets[k]
        #         pass

        # print(i - FFT_window + onset)
        # fig, ax = plt.subplots(1, figsize=(6, 3))
        # ax.plot(signal)
        # ax.axvline(onset,-1,1, color='red', ls='--')
        # ax.set_ylim([-1,1])
        # plt.show()
        # plt.close(fig)

        if not newOnsets:
            # if no onset is found --> check for end of event by maximum duration and append silence if ended
            event_samples = np.concatenate((event_samples, signal[-hop_size:].reshape(-1)))

        # onsets_samples = onsetDetector(signal.reshape(1,-1))


    # for event in tot_events:
    #     fig, ax = plt.subplots(1, figsize=(6, 3))
    #     ax.plot(event)
    #     ax.set_ylim([-1,1])
    #     plt.show()
    #     plt.close(fig)


    print(found_onsets_general)
    onsets_samples = found_onsets_general
    src_signal, _ = librosa.load(src_audio_filepath, sr=sr, mono=True)
    src_signal = adjustAmplitude(src_signal, FFT_window, hop_size)
    fig, ax = plt.subplots(2, figsize=(8, 6))
    librosa.display.waveshow(y=src_signal, sr=sr, color="black", axis='time', ax=ax[0], alpha=0.2)
    # librosa.display.waveshow(y=np.concatenate(tot_events), sr=sr, color="black", axis='time', ax=ax[1], alpha=0.2)
    ax[1].plot(np.concatenate(tot_events))
    for i in range(len(onsets_samples)):
        ax[0].axvline(onsets_samples[i] / sr,-1,1, color='red', ls='--')
    plt.show()
    plt.close(fig)


