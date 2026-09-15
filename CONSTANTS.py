# CONSTANTS
sr = 22050
FFT_window = 512
hop_size = 256
num_MFCC = 16
num_MEL = 64
melspec_size_flow = 80
silence_threshold = 0.001
max_event_duration_samples = 88200
min_event_duration_samples = 882*2

# THESE ARE THE SAME FOR ALL SYNTHS
frequency_param_name = "frequency"
gain_param_name = "gain"
gate_param_name = "gate"
envelope_param_names = ['A','D','S','R']
non_timbre_params_names = [frequency_param_name, 
                           gain_param_name, 
                           gate_param_name]
non_timbre_params_names += envelope_param_names

RMS_multiplier = 5.0
ADSR_standard = [0.2, 0.9, 0.8, 0.5]