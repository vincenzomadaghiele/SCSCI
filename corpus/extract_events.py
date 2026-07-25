import os
import sys
import json
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from corpus.onset_detect import OnsetDetection

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from CONSTANTS import *

def computeFeatures(signal, sample_rate, FFT_window_size, hop_size, 
					mfcc_N=16, mels_N=64, compute_rests=False, 
					silence_threshold=0.0005, max_frames=80):
	rms = librosa.feature.rms(y=signal, frame_length=FFT_window_size, hop_length=hop_size)
	if compute_rests:
		event_duration = [1 if rms_value > silence_threshold else 0 for rms_value in rms[0].tolist()]
		event_duration = (np.array(event_duration) == 1).sum()
		event_duration = librosa.frames_to_samples(event_duration, hop_length=hop_size, n_fft=FFT_window_size)

	mel = librosa.feature.melspectrogram(y=signal, sr=sample_rate, n_fft=FFT_window_size, 
									  	n_mels=mels_N, hop_length=hop_size, fmin=30, fmax=11000)
	mel = mel[:mels_N, :max_frames]

	# mfcc = librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=mfcc_N, hop_length=hop_size)
	# mfcc = mfcc[:mfcc_N, :int(mel.shape[1]*mels_N/mfcc_N)]
	# mfcc = mfcc.reshape(mels_N, -1)

	return rms, mel, event_duration
	# return rms, mfcc, mel, event_duration

def adjustAmplitude(signal, FFT_window_size, hop_size):
	rms = librosa.feature.rms(y=signal, frame_length=FFT_window_size, hop_length=hop_size)
	if rms.max() < 0.7: # normalize amplitude
		signal *= 0.6/signal.max()
	return signal

def detectOnsets(signal, sample_rate, min_event_duration_samples=441):
	onsetDetector = OnsetDetection(sr=sample_rate)
	onsets = onsetDetector(signal.reshape(1,-1))
	newonsets = []
	latest_onset = 0
	for i in range(1,len(onsets)):
		if int(onsets[i]) - latest_onset > min_event_duration_samples:
			newonsets.append(onsets[i])
			latest_onset = onsets[i]
	return newonsets

def computeEventFeatures(signal, onsets, sample_rate, FFT_window_size, hop_size, mfcc_N=16, mels_N=64, compute_rests=True, silence_threshold=0.0005, max_frames=80):
	events = []
	for i in range(len(onsets)-1):
		event_info = {}
		# slice soundfile
		this_onset_signal = signal[onsets[i]:onsets[i+1]]

		# extract features
		rms = librosa.feature.rms(y=this_onset_signal, frame_length=FFT_window_size, hop_length=hop_size)
		if compute_rests:
			event_duration = [1 if rms_value > silence_threshold else 0 for rms_value in rms[0].tolist()]
			event_duration = (np.array(event_duration) == 1).sum()
			duration = librosa.frames_to_samples(event_duration, hop_length=hop_size, n_fft=FFT_window_size)
		mel = librosa.feature.melspectrogram(y=this_onset_signal, sr=sample_rate, n_fft=FFT_window_size, 
											n_mels=mels_N, hop_length=hop_size, fmin=30, fmax=11000)
		mel = mel[:mels_N, :max_frames]

		# save
		event_info['mel'] = mel
		event_info['event_duration'] = duration
		event_info['event_start'] = onsets[i]
		events.append(event_info)

	return events

def detectRests(events, num_MEL, min_rest_length_samples, max_length_samples, FFT_window_size, hop_size, last_sample_n):
	new_events = []
	# check for silent start
	if events[0]['event_start'] > 0:
		dur_counter = int(events[1]['event_start'])
		start_counter = 0
		while dur_counter > 0:
			if dur_counter > min_rest_length_samples:
				new_row_data = {} 
				if dur_counter > max_length_samples:
					new_row_data['event_start'] = start_counter 
					new_row_data['event_duration'] = max_length_samples
					new_row_data['mel'] = np.zeros((num_MEL, librosa.samples_to_frames(new_row_data['event_duration'], hop_length=hop_size, n_fft=FFT_window_size)))
					new_events.append(new_row_data)
					start_counter += max_length_samples
					dur_counter -= max_length_samples
				else:
					# last rest
					if dur_counter > min_rest_length_samples:
						new_row_data['event_start'] = start_counter 
						new_row_data['event_duration'] = dur_counter
						new_row_data['mel'] = np.zeros((num_MEL, librosa.samples_to_frames(new_row_data['event_duration'], hop_length=hop_size, n_fft=FFT_window_size)))
						new_events.append(new_row_data)
					start_counter += max_length_samples
					dur_counter -= max_length_samples
			else:
				dur_counter = 0
	for i in range(len(events) - 1):
		current_row = events[i]
		next_row = events[i+1]
		new_events.append(current_row)
		if current_row['event_duration'] < next_row['event_start'] - current_row['event_start']:
			total_rest_duration = next_row['event_start'] - (current_row['event_start'] + current_row['event_duration'])
			start_counter = current_row['event_start'] + current_row['event_duration']
			while total_rest_duration > 0:
				if total_rest_duration > min_rest_length_samples:
					new_row_data = {} 
					if total_rest_duration > max_length_samples:
						new_row_data['event_start'] = start_counter 
						new_row_data['event_duration'] = max_length_samples
						new_row_data['mel'] = np.zeros((num_MEL, librosa.samples_to_frames(new_row_data['event_duration'], hop_length=hop_size, n_fft=FFT_window_size)))
						new_events.append(new_row_data)
						start_counter += max_length_samples
						total_rest_duration -= max_length_samples
					else:
						# last rest
						if total_rest_duration > min_rest_length_samples:
							new_row_data['event_start'] = start_counter 
							new_row_data['event_duration'] = total_rest_duration
							new_row_data['mel'] = np.zeros((num_MEL, np.clip(librosa.samples_to_frames(new_row_data['event_duration'], hop_length=hop_size, n_fft=FFT_window_size), a_min=0, a_max=None)))
							new_events.append(new_row_data)
						start_counter += max_length_samples
						total_rest_duration -= max_length_samples
				else:
					total_rest_duration = 0
	if last_sample_n:
		start_counter = events[-1]['event_start'] + events[-1]['event_duration']
		total_rest_duration = last_sample_n - start_counter
		while total_rest_duration > 0:
			if total_rest_duration > min_rest_length_samples:
				new_row_data = {} 
				if total_rest_duration > max_length_samples:
					new_row_data['event_start'] = start_counter 
					new_row_data['event_duration'] = max_length_samples
					new_row_data['mel'] = np.zeros((num_MEL, librosa.samples_to_frames(new_row_data['event_duration'], hop_length=hop_size, n_fft=FFT_window_size)))
					new_events.append(new_row_data)
					start_counter += max_length_samples
					total_rest_duration -= max_length_samples
				else:
					# last rest
					if total_rest_duration > min_rest_length_samples:
						new_row_data['event_start'] = start_counter 
						new_row_data['event_duration'] = total_rest_duration
						new_row_data['mel'] = np.zeros((num_MEL, librosa.samples_to_frames(new_row_data['event_duration'], hop_length=hop_size, n_fft=FFT_window_size)))
						new_events.append(new_row_data)
					start_counter += max_length_samples
					total_rest_duration -= max_length_samples
			else:
				total_rest_duration = 0
	return new_events

def plotEvents(corpus_entry, sample_rate, save_dir=None, sample_start_s=0, sample_duration_s=None):
	plot_df_src = pd.DataFrame(columns=['mels','event_start','event_duration'])
	for e in corpus_entry['events']:
		if e['mel'].shape[0] > 0:
			new_row_data = {} 
			new_row_data['mels'] = e['mel'].mean()
			new_row_data['event_start'] = e['event_start']
			new_row_data['event_duration'] = e['event_duration']
			plot_df_src = pd.concat([plot_df_src, pd.DataFrame([new_row_data])], ignore_index=True)

	signal, _ = librosa.load(corpus_entry['filename'], sr=sample_rate, mono=True)

	sample_start = sample_start_s * sample_rate
	if sample_duration_s:
		sample_duration = sample_duration_s * sample_rate

	plot_feature_src = 'mels'
	plot_df_src[plot_feature_src] = 2*((plot_df_src[plot_feature_src] - plot_df_src[plot_feature_src].min())/(plot_df_src[plot_feature_src].max() - plot_df_src[plot_feature_src].min())) - 1
	plot_df_src = plot_df_src[plot_df_src['event_start'] > sample_start]
	if sample_duration_s and (sample_start+sample_duration) < signal.shape[0]:
		plot_df_src = plot_df_src[plot_df_src['event_start'] < sample_start+sample_duration]
	plot_df_src['event_start'] = plot_df_src['event_start'] - sample_start

	fig, ax = plt.subplots(1, figsize=(20, 4))
	if sample_duration_s and (sample_start+sample_duration) < signal.shape[0]:
		librosa.display.waveshow(y=signal[sample_start:sample_start+sample_duration], sr=sample_rate, color="black", axis='time', ax=ax, alpha=0.2)
	elif sample_start < signal.shape[0]:
		librosa.display.waveshow(y=signal[sample_start:], sr=sample_rate, color="black", axis='time', ax=ax, alpha=0.2)
	ax.scatter(librosa.samples_to_time(plot_df_src['event_start'], sr=sample_rate), 
				y=plot_df_src[plot_feature_src], label=plot_feature_src, edgecolors='r', facecolors='none')
	ax.hlines(y=plot_df_src[plot_feature_src],
			xmin=librosa.samples_to_time(plot_df_src['event_start'], sr=sample_rate), 
			xmax=librosa.samples_to_time(plot_df_src['event_start']+plot_df_src['event_duration'], sr=sample_rate), 
			linewidth=1, label='event duration', colors='r')
	ax.vlines(librosa.samples_to_time(plot_df_src['event_start'], sr=sample_rate), 
		-1, 1, color='r', alpha=0.1, linestyle='--', label='onsets', linewidth=0.5)
	ax.legend()
	ax.set_xlabel('time [s]')
	ax.set_ylabel(plot_feature_src)
	figname = corpus_entry['filename'].split('/')[-1].split('.')[0]
	ax.set_title(figname)
	if save_dir:
		plt.savefig(f'{save_dir}/{figname}.png')
	else:
		plt.show()

def plotPairedEvents(src_corpus_entry, tgt_corpus_entry, sample_rate, save_dir=None, sample_start_s=0, sample_duration_s=None):
	plot_df_src = pd.DataFrame(columns=['mels','event_start','event_duration'])
	for e in src_corpus_entry['events']:
		if e['mel'].shape[0] > 0:
			new_row_data = {} 
			new_row_data['mels'] = e['mel'].mean()
			new_row_data['event_start'] = e['event_start']
			new_row_data['event_duration'] = e['event_duration']
			plot_df_src = pd.concat([plot_df_src, pd.DataFrame([new_row_data])], ignore_index=True)
	signal_src, _ = librosa.load(src_corpus_entry['filename'], sr=sample_rate, mono=True)
	sample_start = sample_start_s * sample_rate
	if sample_duration_s:
		sample_duration = sample_duration_s * sample_rate
	plot_feature_src = 'mels'
	plot_df_src[plot_feature_src] = 2*((plot_df_src[plot_feature_src] - plot_df_src[plot_feature_src].min())/(plot_df_src[plot_feature_src].max() - plot_df_src[plot_feature_src].min())) - 1
	plot_df_src = plot_df_src[plot_df_src['event_start'] > sample_start]
	if sample_duration_s and (sample_start+sample_duration) < signal_src.shape[0]:
		plot_df_src = plot_df_src[plot_df_src['event_start'] < sample_start+sample_duration]
	plot_df_src['event_start'] = plot_df_src['event_start'] - sample_start

	plot_df_tgt = pd.DataFrame(columns=['mels','event_start','event_duration'])
	for e in tgt_corpus_entry['events']:
		new_row_data = {} 
		new_row_data['mels'] = e['mel'].mean()
		new_row_data['event_start'] = e['event_start']
		new_row_data['event_duration'] = e['event_duration']
		plot_df_tgt = pd.concat([plot_df_tgt, pd.DataFrame([new_row_data])], ignore_index=True)
	signal_tgt, _ = librosa.load(tgt_corpus_entry['filename'], sr=sample_rate, mono=True)
	sample_start = sample_start_s * sample_rate
	if sample_duration_s:
		sample_duration = sample_duration_s * sample_rate
	plot_feature_tgt = 'mels'
	plot_df_tgt[plot_feature_tgt] = 2*((plot_df_tgt[plot_feature_tgt] - plot_df_tgt[plot_feature_tgt].min())/(plot_df_tgt[plot_feature_tgt].max() - plot_df_tgt[plot_feature_tgt].min())) - 1
	plot_df_tgt = plot_df_tgt[plot_df_tgt['event_start'] > sample_start]
	if sample_duration_s and (sample_start+sample_duration) < signal_tgt.shape[0]:
		plot_df_tgt = plot_df_tgt[plot_df_tgt['event_start'] < sample_start+sample_duration]
	plot_df_tgt['event_start'] = plot_df_tgt['event_start'] - sample_start

	fig, ax = plt.subplots(2, figsize=(16, 6))
	if sample_duration_s and (sample_start+sample_duration) < signal_src.shape[0]:
		librosa.display.waveshow(y=signal_src[sample_start:sample_start+sample_duration], sr=sample_rate, color="black", axis='time', ax=ax[0], alpha=0.2)
	elif sample_start < signal_src.shape[0]:
		librosa.display.waveshow(y=signal_src[sample_start:], sr=sample_rate, color="black", axis='time', ax=ax[0], alpha=0.2)

	if sample_duration_s and (sample_start+sample_duration) < signal_tgt.shape[0]:
		librosa.display.waveshow(y=signal_tgt[sample_start:sample_start+sample_duration], sr=sample_rate, color="black", axis='time', ax=ax[1], alpha=0.2)
	elif sample_start < signal_tgt.shape[0]:
		librosa.display.waveshow(y=signal_tgt[sample_start:], sr=sample_rate, color="black", axis='time', ax=ax[1], alpha=0.2)

	ax[0].scatter(librosa.samples_to_time(plot_df_src['event_start'], sr=sample_rate), 
				y=plot_df_src[plot_feature_src], label=plot_feature_src, edgecolors='r', facecolors='none')
	ax[0].hlines(y=plot_df_src[plot_feature_src],
			xmin=librosa.samples_to_time(plot_df_src['event_start'], sr=sample_rate), 
			xmax=librosa.samples_to_time(plot_df_src['event_start']+plot_df_src['event_duration'], sr=sample_rate), 
			linewidth=1, label='event duration', colors='r')
	ax[0].vlines(librosa.samples_to_time(plot_df_src['event_start'], sr=sample_rate), 
		-1, 1, color='r', alpha=0.1, linestyle='--', label='onsets', linewidth=0.5)

	ax[1].scatter(librosa.samples_to_time(plot_df_tgt['event_start'], sr=sample_rate), 
				y=plot_df_tgt[plot_feature_tgt], label=plot_feature_tgt, edgecolors='b', facecolors='none')
	ax[1].hlines(y=plot_df_tgt[plot_feature_tgt],
			xmin=librosa.samples_to_time(plot_df_tgt['event_start'], sr=sample_rate), 
			xmax=librosa.samples_to_time(plot_df_tgt['event_start']+plot_df_tgt['event_duration'], sr=sample_rate), 
			linewidth=1, label='event duration', colors='b')
	ax[1].vlines(librosa.samples_to_time(plot_df_tgt['event_start'], sr=sample_rate), 
		-1, 1, color='b', alpha=0.1, linestyle='--', label='onsets', linewidth=0.5)

	ax[0].legend()
	ax[1].legend()
	ax[0].set_ylabel(plot_feature_src)
	ax[1].set_ylabel(plot_feature_tgt)
	ax[0].set_xlabel([])
	ax[1].set_xlabel('time [s]')
	ax[0].set_title(src_corpus_entry['filename'].split('/')[-1].split('.')[0])
	ax[1].set_title(tgt_corpus_entry['filename'].split('/')[-1].split('.')[0])
	songname = src_corpus_entry['filename'].split('/')[-2]
	if save_dir:
		fig.savefig(f'{save_dir}/{songname}.png')
	else:
		plt.show()

def extractEvents(signal, 
				  FFT_window, 
				  sr, 
				  hop_size, 
				  num_MFCC, 
				  num_MEL, 
				  min_event_duration_samples, 
				  max_event_duration_samples):
	signal = adjustAmplitude(signal, FFT_window, hop_size)
	onsets = detectOnsets(signal, sr, min_event_duration_samples=min_event_duration_samples)
	print(f'detected {len(onsets)} onsets...')
	events = computeEventFeatures(signal, onsets, sr, FFT_window, hop_size, mfcc_N=num_MFCC, mels_N=num_MEL, compute_rests=True)
	last_sample_n = signal.shape[0]-1
	new_events = detectRests(events, num_MEL, min_event_duration_samples, max_event_duration_samples, FFT_window, hop_size, last_sample_n )
	return new_events


if __name__ == "__main__":

	# extract 
	corpus_name = 'guitar-duo'
	src_track_name = 'guitar1'
	tgt_track_name = 'guitar2'
	corpus_path = f'corpus/{corpus_name}'
	corpus_dirs = os.listdir(corpus_path)
	src_tgt = []
	for track_name in [src_track_name, tgt_track_name]:
		corpus_events = []
		corpus_dirs = [c for c in corpus_dirs if os.path.isdir(f'{corpus_path}/{c}') and c not in ['info','trained_VAE','.DS_Store','src_full']]
		corpus_files = [f'{corpus_path}/{song_name}/{track_name}.wav' for song_name in corpus_dirs]
		for song_name in corpus_dirs:
			os.makedirs(f'{corpus_path}/{song_name}/mel', exist_ok=True)
			os.makedirs(f'{corpus_path}/{song_name}/img', exist_ok=True)

		# save processing parameters
		processing_params = {}
		processing_params['sample_rate'] = sr
		processing_params['FFT_window_size'] = FFT_window
		processing_params['hop_size'] = hop_size
		processing_params['num_MFCC'] = num_MFCC
		processing_params['num_MEL'] = num_MEL
		processing_params['silence_threshold'] = silence_threshold
		processing_params['max_event_duration_samples'] = max_event_duration_samples
		processing_params['min_event_duration_samples'] = min_event_duration_samples
		with open(f'{corpus_path}/featureExtraction_params.json', 'w', encoding='utf-8') as f:
			json.dump(processing_params, f, ensure_ascii=False, indent=4)

		for audiofile in corpus_files:
			trackname = audiofile.split('/')[-1].split('.')[0]
			song_name = audiofile.split('/')[-2]
			signal, _ = librosa.load(audiofile, sr=sr, mono=True)
			new_events = extractEvents(signal, 
							  			FFT_window, sr, hop_size, 
							  			num_MFCC, num_MEL, 
										min_event_duration_samples, max_event_duration_samples)

			# signal = adjustAmplitude(signal, FFT_window, hop_size)
			# print(f'Loading audiofile {audiofile} with length {signal.shape[0]} samples')
			# onsets = detectOnsets(signal, sr, min_event_duration_samples=min_event_duration_samples)
			# print(f'detected {len(onsets)} onsets...')
			# events = computeEventFeatures(signal, onsets, sr, FFT_window, hop_size, mfcc_N=num_MFCC, mels_N=num_MEL, compute_rests=True)
			# last_sample_n = signal.shape[0]-1
			# new_events = detectRests(events, num_MEL, min_event_duration_samples, max_event_duration_samples,
			# 						FFT_window, hop_size, last_sample_n )
			this_file_event = {}
			this_file_event['filename'] = audiofile
			this_file_event['song_name'] = song_name
			this_file_event['events'] = new_events
			np.save(f'./{corpus_path}/{song_name}/mel/{trackname}.npy', np.array(new_events, dtype=object), allow_pickle=True)
			corpus_events.append(this_file_event)
		src_tgt.append(corpus_events)
		for corpus_entry in corpus_events:
			plotEvents(corpus_entry, sr, save_dir=f'./{corpus_path}/{corpus_entry["song_name"]}/img')

	src_events, tgt_events = src_tgt[0], src_tgt[1]
	for i in range(len(src_events)):
		src_song = src_events[i]
		tgt_song = tgt_events[i]
		if src_song['song_name'] == tgt_song['song_name']:
			plotPairedEvents(src_song, tgt_song, sr, save_dir=f'./{corpus_path}/{src_song["song_name"]}/img')
