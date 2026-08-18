import time
import random
import math
import numpy as np
from signalflow import *


class TimbreTheremin(Patch):
    def __init__(self, parameter_values: list = [0,0,0,0,0,0,0,0]):
        super().__init__()

        frequency = self.add_input(f"frequency", parameter_values[0]) # Hz
        gain = self.add_input(f"gain", parameter_values[1])
        sine_tri = self.add_input(f"sine_tri", parameter_values[2])
        saw_square = self.add_input(f"saw_square", parameter_values[3])
        A = self.add_input(f"A", parameter_values[4])
        D = self.add_input(f"D", parameter_values[5])
        S = self.add_input(f"S", parameter_values[6])
        R = self.add_input(f"R", parameter_values[7])
        gate = self.add_input("gate", 1.0)

        frequency = Clip(frequency, 50, 11050)
        gain = Clip(gain, 0, 1)
        sine_tri = Clip(sine_tri, 0, 1)
        saw_square = Clip(saw_square, 0, 1)
        A = ScaleLinLin(input=A, a=0, b=1, c=0.001, d=0.01)
        D = ScaleLinLin(input=D, a=0, b=1, c=0.1, d=0.6)
        S = ScaleLinLin(input=S, a=0, b=1, c=0.3, d=1)
        R = ScaleLinLin(input=R, a=0, b=1, c=0.1, d=0.3)

        # create synth
        sine = SineOscillator(frequency)
        tri = TriangleOscillator(frequency)
        saw = SawOscillator(frequency)
        square = SquareOscillator(frequency)
        env = ADSREnvelope(A, D, S, R, gate=gate)
        # env = ASREnvelope(A, S, R)
        output1 = (tri * sine_tri + sine * (1-sine_tri)) * 0.8
        output2 = (saw * saw_square + square * (1-saw_square)) * 0.2
        # combined_output = (output1 + output2)
        # filter = BiquadFilter(input=combined_output, filter_type='low_pass', cutoff=frequency, resonance=0.1, peak_gain=0.0)
        # output = filter * env * gain
        output = (output1 + output2) * env * gain

        # forward
        self.set_output(output)
        self.set_auto_free(True)
        # self.set_trigger_node(env)


if __name__ == "__main__":

    # AUDIO GRAPH
    config = AudioGraphConfig()
    config.output_buffer_size = 256
    graph = AudioGraph(config=config, start=True)

    # GET SYNTH PROPERTIES
    synth = TimbreTheremin()
    param_names = list(synth.inputs.keys())
    param_names.remove("gate")
    param_names.remove("gain")
    N_params = len(param_names)

    # PLAYBACK PROPERTIES
    metro = 0.6
    random_metro = 0.5
    NUM_VOICES = 10
    MAX_DURATION = 4
    voices = [ None ] * NUM_VOICES
    durations = np.zeros(NUM_VOICES)

    # PLAY
    j = 0
    while True:
        voice = TimbreTheremin()
        new_parameters = np.random.rand(N_params).tolist()
        for i, param_name in enumerate(param_names):
            if param_name == "frequency":
                new_parameters[i] = math.log2(new_parameters[i]+1) * 11000 + 50
            voice.set_input(param_name, new_parameters[i]) 
        voice.set_input("gain", 1 / NUM_VOICES)
        voice.set_input("gate", 1)
        # voice.trigger()
        # voice.auto_free = True
        # panner = StereoPanner( voice , pan=float(np.random.rand()))
        voice.play()
        voices[j] = voice
        durations[j] = random.random() * MAX_DURATION
        j += 1
        if j == NUM_VOICES:
            j = 0
        durations[np.argwhere(durations > 0)] = durations[np.argwhere(durations > 0)] - metro
        ended_notes = np.argwhere(durations < 0).T
        for k in ended_notes.tolist()[0]:
            voices[k].set_input("gate", 0)
            durations[k] = 0
        time.sleep(metro + random.random() * random_metro)
