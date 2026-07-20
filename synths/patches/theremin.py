from signalflow import *
import time
import numpy as np
import random


class ADSRTheremin(Patch):
    def __init__(self, parameter_values: list = [0,0,0,0,0,0]):
        super().__init__()

        frequency = self.add_input(f"frequency", parameter_values[0])
        gain = self.add_input(f"gain", parameter_values[1])
        A = self.add_input(f"A", parameter_values[2])
        D = self.add_input(f"D", parameter_values[3])
        S = self.add_input(f"S", parameter_values[4])
        R = self.add_input(f"R", parameter_values[5])
        gate = self.add_input("gate", 1.0)

        frequency = ScaleLinLin(input=frequency, a=0, b=1, c=220, d=660)
        gain = ScaleLinLin(input=gain, a=0, b=1, c=0, d=1)
        A = ScaleLinLin(input=A, a=0, b=1, c=0.001, d=0.01)
        D = ScaleLinLin(input=D, a=0, b=1, c=0.1, d=0.6)
        S = ScaleLinLin(input=S, a=0, b=1, c=0.3, d=1)
        R = ScaleLinLin(input=R, a=0, b=1, c=0.1, d=0.3)

        # create synth
        sine = SineOscillator(frequency)
        env = ADSREnvelope(A, D, S, R, gate=gate)
        output = sine * env * gain

        # forward
        self.set_output(output)
        self.set_auto_free(True)

if __name__ == "__main__":

    # AUDIO GRAPH
    config = AudioGraphConfig()
    config.output_buffer_size = 256
    graph = AudioGraph(config=config, start=True)

    # GET SYNTH PROPERTIES
    synth = ADSRTheremin()
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
        voice = ADSRTheremin()
        new_parameters = np.random.rand(N_params).tolist()
        for i, param_name in enumerate(param_names):
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


