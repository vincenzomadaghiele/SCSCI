import sys
import argparse
from signalflow import *
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from patches.benjolin import Benjolin
from patches.granular import Granular
from patches.snare import Snare
from patches.theremin import ADSRTheremin

class Synth:
    def __init__(self, 
                synth_name: str,
                sample_rate=44100,
                window_size = 2048,
                hop_size = 512
                ):
        self.config = AudioGraphConfig()
        self.config.sample_rate = sample_rate
        self.config.output_buffer_size = hop_size
        self.synth_name = synth_name
        self.window_size = window_size
        self.hop_size = hop_size
        self.fade_time = hop_size / sample_rate / 8
        self.connectGraph()
    def connectGraph(self):
        # intialize graph
        self.graph = AudioGraph(output_device=AudioOut_Dummy(2), start=False)
        # initialize buffers
        self.buffer = Buffer(1, self.hop_size)
        self.buf = Buffer(1, self.hop_size)
        # intialize graph
        if self.synth_name == 'Benjolin':
            self.synth = Benjolin(self.graph) 
        elif self.synth_name == 'Granular':
            self.synth = Granular() 
        elif self.synth_name == 'Snare':
            self.synth = Snare() 
        else:
            self.synth = ADSRTheremin() 
        self.synth.play()
    def forward(self):
        # make a new recorder
        self.src_rec = BufferRecorder(self.buf, self.synth)
        self.graph.play(self.src_rec)
        self.graph.render_to_buffer(self.buffer)
        return self.buf.data[0,:]
    def resetParameters(self, new_parameters: list):
        for i, param_name in enumerate(list(self.synth.inputs.keys())):
            self.synth.set_input(param_name, new_parameters[i]) 
    def fadeParameters(self, new_parameters: list, fade_time: float=None):
        if fade_time is None:
            fade_time = self.fade_time
        for i, param_name in enumerate(list(self.synth.inputs.keys())):
            current_p = self.synth.inputs[param_name].value
            p_fade = Line(current_p, new_parameters[i], fade_time)
            self.synth.set_input(param_name, p_fade) 
    def getInputs(self):
        return self.synth.inputs
    def resetGraph(self):
        if self.graph:
            self.graph.destroy()
        self.connectGraph()
    def close(self):
        self.graph.destroy()


if __name__ == "__main__":

    # parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--SYNTH_NAME', type=str, default='Theremin',
                        help='synth name among Theremin, Granular, Benjolin, Snare')
    parser.add_argument('--SAVE_FILEPATH', type=str, default='./00_dataset',
                        help='path of where to save the soundfile')
    parser.add_argument('--SYNTH_PARAMETERS', type=str, default='0.5-0.5-0.5-0.5-0.5-0.5-0.5-0.5',
                        help='Granular parameters')
    parser.add_argument('--RECORDING_TIME', type=int, default=30,
                        help='recording time in seconds')
    parser.add_argument('--SAMPLE_RATE', type=int, default=44100,
                        help='sample rate')
    parser.add_argument('--WINDOW_SIZE', type=int, default=4096,
                        help='FFT window size')
    parser.add_argument('--HOP_SIZE', type=int, default=2048,
                        help='FFT hop size')
    args = parser.parse_args(sys.argv[1:])

    ## DEFINE SCRIPT PARAMETERS
    SYNTH_NAME = args.SYNTH_NAME
    SAVE_FILEPATH = args.SAVE_FILEPATH
    SYNTH_PARAMETERS = args.SYNTH_PARAMETERS
    RECORDING_TIME = args.RECORDING_TIME
    sr = args.SAMPLE_RATE
    window_size = args.WINDOW_SIZE
    hop_size = args.HOP_SIZE

    recording_time_samples = RECORDING_TIME * sr
    synth = Synth(SYNTH_NAME, sr, window_size, hop_size)
    N_params = len(synth.getInputs())
    print(synth.getInputs())
    synth_parameters = SYNTH_PARAMETERS.split('-')
    synth_parameters = [float(sy) for sy in synth_parameters]
    synth_parameters.append(1) # assumes "gain" is always last
    synth.resetParameters(synth_parameters)

    # FIX: RECORD ONLY ONE ONSET
    # ALWAYS CHANGE PARAMETERS BY NAME

    STEPS = int(recording_time_samples / hop_size)
    src_array = np.zeros((hop_size*STEPS))
    for n in range(STEPS):
        src_sound = synth.forward()
        src_array[:-hop_size] = src_array[hop_size:] 
        src_array[-hop_size:] = src_sound[-hop_size:] 
    sf.write(f'{SAVE_FILEPATH}/{SYNTH_PARAMETERS}.wav', np.array(src_array), sr, subtype='PCM_24')
