import sys
import argparse
from signalflow import *
import numpy as np
import soundfile as sf
import matplotlib.pyplot as plt
from synth.patches.benjolin import Benjolin
from synth.patches.granular import Granular
from synth.patches.snare import Snare
from synth.patches.theremin import ADSRTheremin

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
        # initialize buffers
        self.buffer = Buffer(1, int(self.graph.sample_rate * self.buffer_size))
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
        # self.synth.set_input("gate", 1)
        self.synth.trigger()
        self.graph.render_to_buffer(self.buffer)
        rendered_array = self.buffer.data[0,:]
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
    parser.add_argument('--SYNTH_NAME', type=str, default='Theremin',
                        help='synth name among Theremin, Granular, Benjolin, Snare')
    parser.add_argument('--SAVE_FILEPATH', type=str, default='./datasets',
                        help='path of where to save the soundfile')
    parser.add_argument('--SYNTH_PARAMETERS', type=str, default='0.5-0.5-0.5-0.5-0.5-0.5',
                        help='Granular parameters')
    parser.add_argument('--RECORDING_TIME', type=int, default=5,
                        help='recording time in seconds')
    parser.add_argument('--SAMPLE_RATE', type=int, default=44100,
                        help='sample rate')
    parser.add_argument('--HOP_SIZE', type=int, default=2048,
                        help='FFT hop size')
    args = parser.parse_args(sys.argv[1:])

    ## DEFINE SCRIPT PARAMETERS
    SYNTH_NAME = args.SYNTH_NAME
    SAVE_FILEPATH = args.SAVE_FILEPATH
    SYNTH_PARAMETERS = args.SYNTH_PARAMETERS
    RECORDING_TIME = args.RECORDING_TIME
    sr = args.SAMPLE_RATE
    hop_size = args.HOP_SIZE

    synth = Synth(SYNTH_NAME, sr, hop_size, RECORDING_TIME)
    synth_parameters = SYNTH_PARAMETERS.split('-')
    synth_parameters = [float(sy) for sy in synth_parameters]

    param_names = synth.getInputs()
    print(param_names)
    param_names.remove("gain")
    N_params = len(param_names)
    synth.setParameters(param_names, synth_parameters)
    synth.setParameters(["gain"], [0.8])
    SYNTH_PARAMETERS = '-'.join([str(sy) for sy in synth_parameters])
    sf.write(f'{SAVE_FILEPATH}/{SYNTH_PARAMETERS}.wav', synth.forward(), sr, subtype='PCM_24')
    synth.close()
