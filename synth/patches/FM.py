from signalflow import *
import time
import numpy as np
import random
import math

class FMOp (Patch):
    def __init__(self, f0, coarse, level, fm=None):
        """
        FM operator that follows the exponential output curve of the Yamaha DX7:
        https://sound.stackexchange.com/a/42979/33063
        """
        super().__init__()
        amplitude = ScaleLinExp(level, 0, 1, 0.0005, 1)
        frequency = f0 * coarse
        if fm is not None:
            frequency = frequency + (f0 * fm * 14)
        sine = SineOscillator(frequency)
        self.output = sine * amplitude


class FM(Patch):
    def __init__(self, parameter_values: list = [0,0,0,0,0,0,0,0,0,0,0,0,0]):
        super().__init__()

        frequency = self.add_input("frequency", parameter_values[0]) # carrier frequency [Hz]
        c0 = self.add_input("parameter-1", parameter_values[1])
        l0 = self.add_input("parameter-2", parameter_values[2])
        c1 = self.add_input("parameter-3", parameter_values[3])
        l1 = self.add_input("parameter-4", parameter_values[4])
        c2 = self.add_input("parameter-5", parameter_values[5])
        l2 = self.add_input("parameter-6", parameter_values[6])
        gain = self.add_input("gain", parameter_values[7])
        A = self.add_input(f"A", parameter_values[8])
        D = self.add_input(f"D", parameter_values[9])
        S = self.add_input(f"S", parameter_values[10])
        R = self.add_input(f"R", parameter_values[11])
        gate = self.add_input("gate", 1.0)

        c0 = ScaleLinLin(input=c0, a=0, b=1, c=0, d=10)
        l0 = ScaleLinLin(input=l0, a=0, b=1, c=0, d=1)
        c1 = ScaleLinLin(input=c1, a=0, b=1, c=0, d=10)
        l1 = ScaleLinLin(input=l1, a=0, b=1, c=0, d=1)
        c2 = ScaleLinLin(input=c2, a=0, b=1, c=0, d=10)
        l2 = ScaleLinLin(input=l2, a=0, b=1, c=0, d=1)
        A = ScaleLinLin(input=A, a=0, b=1, c=0.001, d=0.1)
        D = ScaleLinLin(input=D, a=0, b=1, c=0.01, d=0.3)
        S = ScaleLinLin(input=S, a=0, b=1, c=0.5, d=3)
        R = ScaleLinLin(input=R, a=0, b=1, c=0.6, d=2)

        params = [[c0,l0], [c1,l1], [c2,l2]]

        op2 = FMOp(frequency, *params[2])
        op1 = FMOp(frequency, *params[1], fm=op2)
        op0 = FMOp(frequency, *params[0], fm=op1+op2)
        env = ADSREnvelope(A, D, S, R, gate=gate)
        output = Tanh(op0 + op1) * gain * env 
        # out = Squiz(input=(op0 + op1), rate=0.7, chunk_size=1)
        # output = out * gain * env

        self.output = output
        self.set_auto_free(True)

if __name__ == "__main__":

    # AUDIO GRAPH
    config = AudioGraphConfig()
    config.output_buffer_size = 256
    config.output_device_name = "Scarlett 2i2 USB"
    config.input_device_name = "Scarlett 2i2 USB"
    graph = AudioGraph(config=config, start=True)

    # GET SYNTH PROPERTIES
    synth = FM()
    param_names = list(synth.inputs.keys())
    param_names.remove("gate")
    param_names.remove("gain")
    N_params = len(param_names)

    # PLAYBACK PROPERTIES
    metro = 0.3
    random_metro = 0.5
    NUM_VOICES = 10
    MAX_DURATION = 4
    voices = [ None ] * NUM_VOICES
    durations = np.zeros(NUM_VOICES)

    # PLAY
    j = 0
    while True:
        voice = FM()
        new_parameters = np.random.rand(N_params).tolist()
        for i, param_name in enumerate(param_names):
            if param_name == "frequency":
                new_parameters[i] = math.log2(new_parameters[i]+1) * 11000 + 50
            voice.set_input(param_name, new_parameters[i]) 
        voice.set_input("gain", 1 / NUM_VOICES)
        voice.set_input("gate", 1)
        voice.auto_free = True
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
