import os
import sys
import argparse
from signalflow import *
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from patches.benjolin import Benjolin
from patches.granular import Granular
from patches.snare import Snare
from patches.timbreTheremin import TimbreTheremin

# from CONSTANTS import *
# sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class Synth:
    def __init__(self, 
                synth_name: str,
                sample_rate = 44100,
                hop_size = 512,
                buffer_size = 1
                ):
        self.config = AudioGraphConfig()
        self.config.sample_rate = sample_rate
        self.config.output_buffer_size = hop_size
        self.synth_name = synth_name
        self.buffer_size = buffer_size
        self.connectGraph()
    def connectGraph(self):
        # intialize graph
        self.graph = AudioGraph(output_device=AudioOut_Dummy(2), start=False)
        if self.synth_name == 'Benjolin':
            self.synth = Benjolin(self.graph) 
        elif self.synth_name == 'Granular':
            import librosa
            sample_path = "./patches/samples/tibetan-ball.wav"
            sample, _ = librosa.load(sample_path, sr=self.graph.sample_rate, mono=True)
            pitch = librosa.yin(sample, fmin=200, fmax=self.graph.sample_rate/2, sr=self.graph.sample_rate, 
                                frame_length=self.config.output_buffer_size*2, hop_length=self.config.output_buffer_size)
            pitch = [p for p in pitch.tolist() if p != self.graph.sample_rate]
            self.base_pitch_hz = float(np.array(pitch).mean())
            self.synth = Granular(sample_path, self.base_pitch_hz) 
        elif self.synth_name == 'Snare':
            self.synth = Snare() 
        elif self.synth_name == 'TimbreTheremin':
            self.synth = TimbreTheremin() 
    def forward(self, event_duration=0.1):
        buffer_event = Buffer(1, int(self.graph.sample_rate * event_duration))
        buffer_decay = Buffer(1, int(self.graph.sample_rate * (self.buffer_size-event_duration)))
        self.synth.set_input("gate", 1)
        self.synth.play()
        self.graph.render_to_buffer(buffer_event)
        rendered_event = buffer_event.data[0,:]
        self.synth.set_input("gate", 0) # record decay
        self.graph.render_to_buffer(buffer_decay)
        rendered_decay = buffer_decay.data[0,:]
        rendered_array = np.concatenate((rendered_event, rendered_decay), axis=0)
        return rendered_array
    def setParameters(self, new_parameters_names: list, new_parameters_values: list):
        for i, param_name in enumerate(new_parameters_names):
            self.synth.set_input(param_name, new_parameters_values[i]) 
            # print(f'{param_name}: {new_parameters_values[i]}')
    def getInputs(self):
        return list(self.synth.inputs.keys())
    def resetGraph(self):
        if self.graph:
            self.graph.destroy()
        self.connectGraph()
    def close(self):
        self.graph.destroy()


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--SYNTH_NAME', type=str, default='Theremin', help='synth name among Theremin, Granular, Benjolin, Snare')
    parser.add_argument('--SAVE_FILEPATH', type=str, default='./datasets', help='path of where to save the soundfile')
    parser.add_argument('--GAIN', type=float, default=440, help='value of gain parameter of synth [in Hz]')
    parser.add_argument('--FREQUENCY', type=float, default=0.8, help='value of frequency parameter (if available in synth) [0-1]')
    parser.add_argument('--DURATION', type=float, default=0.1, help='duration of synthesized event in seconds')
    parser.add_argument('--ADSR', type=str, default='0.1-0.1-0.1-0.1', help='ASR envelope parameter values [0-1]')
    parser.add_argument('--TIMBRE_PARAMETERS', type=str, default='0.5-0.5-0.5-0.5-0.5-0.5', help='values timbre parameters [0-1]')
    parser.add_argument('--RECORDING_TIME', type=int, default=5, help='recording time in seconds')
    parser.add_argument('--SAMPLE_RATE', type=int, default=44100, help='sample rate')
    parser.add_argument('--HOP_SIZE', type=int, default=2048, help='FFT hop size')
    args = parser.parse_args(sys.argv[1:])

    synth = Synth(args.SYNTH_NAME, args.SAMPLE_RATE, args.HOP_SIZE, args.RECORDING_TIME)
    timbre_parameter_values = [float(sy) for sy in args.TIMBRE_PARAMETERS.split('-')]
    ADSR = [float(sy) for sy in args.ADSR.split('-')]

    parameters_names = synth.getInputs()

    # THESE ARE THE SAME FOR ALL SYNTHS
    frequency_param_name = "frequency"
    gain_param_name = "gain"
    gate_param_name = "gate"
    envelope_param_names = ['A','D','S','R']
    # THESE CAN CHANGE AMONG SYNTHS
    non_timbre_params_names = [frequency_param_name, gain_param_name, gate_param_name]
    non_timbre_params_names += envelope_param_names
    timbre_param_names = [p for p in parameters_names if p not in non_timbre_params_names]
    timbre_parameter_values = timbre_parameter_values[:len(timbre_param_names)]
    # write timbre params to txt
    with open(f"./params/{args.SYNTH_NAME}.txt", "w") as file:
        file.write("\n".join(str(item) for item in timbre_param_names))

    N_timbre_params = len(timbre_param_names)
    synth.setParameters(timbre_param_names, timbre_parameter_values)
    non_timbre_params_names.remove("gate")
    synth.setParameters(non_timbre_params_names, [args.GAIN, args.FREQUENCY, args.DURATION, ADSR[0], ADSR[1], ADSR[2], ADSR[3]])
    if frequency_param_name in parameters_names:
        synth.setParameters([frequency_param_name], [args.FREQUENCY])

    timbre_parameter_values = '-'.join([str(sy) for sy in timbre_parameter_values])
    sf.write(f'{args.SAVE_FILEPATH}/{timbre_parameter_values}.wav', synth.forward(args.DURATION), args.SAMPLE_RATE, subtype='PCM_24')
    synth.close()
