from signalflow import *
import time
import numpy as np
import random
import math

class Snare(Patch):
    def __init__(self, parameter_values: list = [0,0,0,0,0,0,0,0,0]):
        super().__init__()

        cutoff_scale = self.add_input(f"frequency", parameter_values[0])
        center_frequency = self.add_input(f"center_frequency", parameter_values[1])
        modulator_frequency = self.add_input(f"modulator_frequency", parameter_values[2])
        gain = self.add_input(f"gain", parameter_values[3])
        A = self.add_input(f"A", parameter_values[4])
        D = self.add_input(f"D", parameter_values[5])
        S = self.add_input(f"S", parameter_values[6])
        R = self.add_input(f"R", parameter_values[7])
        gate = self.add_input("gate", 1.0)

        cutoff_scale = Clip(cutoff_scale, 50, 10000)
        center_frequency = ScaleLinLin(input=center_frequency, a=0, b=1, c=70, d=200)
        modulator_frequency = ScaleLinLin(input=modulator_frequency, a=0, b=1, c=30, d=60)
        A = ScaleLinLin(input=A, a=0, b=1, c=0.0001, d=0.001)
        D = ScaleLinLin(input=D, a=0, b=1, c=0.01, d=0.3)
        S = ScaleLinLin(input=S, a=0, b=1, c=0.01, d=0.1)
        R = ScaleLinLin(input=R, a=0, b=1, c=0.01, d=0.9)

        # modulator_frequency = 60
        carrier = SineOscillator(modulator_frequency)
        FM = SineOscillator(center_frequency + carrier * 30)
        noise = WhiteNoise(frequency=0.0, min=-1.0, max=1.0)
        noise.set_seed(666)
        env = ADSREnvelope(A, D, S, R, gate=gate)
        # env = ASREnvelope(A, S, R)
        out = noise * env * 0.9
        filter_out = SVFilter(input=out, filter_type='high_pass', cutoff=100, resonance=0.1)
        filter_1 = SVFilter(input=filter_out + FM * env * env, filter_type='band_pass', cutoff=(env**2)*cutoff_scale+100, resonance=0.6)
        filter_2 = SVFilter(input=filter_out + FM * env * env, filter_type='band_pass', cutoff=(env**2)*cutoff_scale*0.6+200, resonance=0.8)
        output = (filter_1 + filter_2) * gain

        # forward
        # self.set_trigger_node(env)
        self.set_output(output)
        self.set_auto_free(True)
        # self.set_auto_free(False)

if __name__ == "__main__":

    # AUDIO GRAPH
    config = AudioGraphConfig()
    config.output_buffer_size = 256
    graph = AudioGraph(config=config, start=True)

    # GET SYNTH PROPERTIES
    synth = Snare()
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
        voice = Snare()
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


