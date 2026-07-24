import os
import json
import librosa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from source.corpora.onset_detect import OnsetDetection


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
							new_row_data['mel'] = np.zeros((num_MEL, librosa.samples_to_frames(new_row_data['event_duration'], hop_length=hop_size, n_fft=FFT_window_size)))
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
		librosa.display.waveshow(y=signal[sample_start:sample_start+sample_duration], sr=sample_rate, color="black", x_axis='time', ax=ax, alpha=0.2)
	elif sample_start < signal.shape[0]:
		librosa.display.waveshow(y=signal[sample_start:], sr=sample_rate, color="black", x_axis='time', ax=ax, alpha=0.2)
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



if __name__ == "__main__":

	processing_params = {}
	processing_params['sample_rate'] = 22050
	processing_params['FFT_window_size'] = 512
	processing_params['hop_size'] = 256
	processing_params['num_MFCC'] = 16
	processing_params['num_MEL'] = 64
	processing_params['silence_threshold'] = 0.001
	processing_params['max_event_duration_samples'] = 88200
	processing_params['min_event_duration_samples'] = 882

	# extract 
	corpus_path = 'corpora/guitar'
	corpus_files = os.listdir(corpus_path)
	corpus_files = [f'{corpus_path}/{filename}' for filename in corpus_files if filename[-4:] == '.wav']
	save_dir = f'{corpus_path}/info'
	os.makedirs(save_dir, exist_ok=True)
	os.makedirs(f'{save_dir}/mel', exist_ok=True)
	os.makedirs(f'{save_dir}/img', exist_ok=True)
	with open(f'{save_dir}/featureExtraction_params.json', 'w', encoding='utf-8') as f:
		json.dump(processing_params, f, ensure_ascii=False, indent=4)

	corpus_events = []
	for audiofile in corpus_files:
		trackname = audiofile.split('/')[-1].split('.')[0]
		signal, _ = librosa.load(audiofile, sr=processing_params['sample_rate'], mono=True)
		signal = adjustAmplitude(signal, processing_params['FFT_window_size'], processing_params['hop_size'])
		print(f'Loading audiofile {audiofile} with length {signal.shape[0]} samples')
		onsets = detectOnsets(signal, processing_params['sample_rate'], 
							min_event_duration_samples=processing_params['min_event_duration_samples'])
		print(f'detected {len(onsets)} onsets...')
		events = computeEventFeatures(signal, onsets, processing_params['sample_rate'], processing_params['FFT_window_size'], 
								processing_params['hop_size'], mfcc_N=processing_params['num_MFCC'], mels_N=processing_params['num_MEL'], compute_rests=True)
		last_sample_n = signal.shape[0]-1
		new_events = detectRests(events, processing_params['num_MEL'], 
								processing_params['min_event_duration_samples'], 
								processing_params['max_event_duration_samples'],
								processing_params['FFT_window_size'],
								processing_params['hop_size'],
								last_sample_n )
		this_file_event = {}
		this_file_event['filename'] = audiofile
		this_file_event['events'] = new_events
		np.save(f'./{corpus_path}/info/mel/{trackname}.npy', np.array(new_events, dtype=object), allow_pickle=True)
		corpus_events.append(this_file_event)

	for corpus_entry in corpus_events:
		plotEvents(corpus_entry, processing_params['sample_rate'], save_dir=f'./{corpus_path}/info/img')

