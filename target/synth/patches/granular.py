from signalflow import *
import time
import numpy as np
import random


class Granular(Patch):
    def __init__(self, parameter_values: list = [0,0,0,0,0,0,0,0,0,0,0,0,0]):
        super().__init__()
        frequency = self.add_input("frequency", parameter_values[0])
        duration = self.add_input("grain-duration", parameter_values[1])
        position = self.add_input("start-position", parameter_values[2])
        rate = self.add_input("rate", parameter_values[3])
        position_variance = self.add_input("position_variance", parameter_values[4])
        duration_variance = self.add_input("duration_variance", parameter_values[5])
        rate_variance = self.add_input("rate_variance", parameter_values[6])
        trigger_variance = self.add_input("trigger_variance", parameter_values[7])
        gain = self.add_input("gain", parameter_values[8])
        A = self.add_input(f"A", parameter_values[9])
        # D = self.add_input(f"D", parameter_values[10])
        S = self.add_input(f"S", parameter_values[10])
        R = self.add_input(f"R", parameter_values[11])
        # gate = self.add_input("gate", 1.0)

        # scale parameters
        # buf = Buffer("./samples/bowed-guitar.wav")
        buf = Buffer("./synths/patches/samples/tibetan-ball.wav")
        frequency = ScaleLinExp(input=frequency, a=0, b=1, c=1, d=50)
        duration = ScaleLinExp(input=duration, a=0, b=1, c=0.1, d=2)
        position = ScaleLinLin(input=position, a=0, b=1, c=0, d=buf.duration)
        rate = ScaleLinExp(input=rate, a=0, b=1, c=0.3, d=4)
        position_variance = ScaleLinLin(input=position_variance, a=0, b=1, c=0, d=int(buf.duration/2))
        duration_variance = ScaleLinLin(input=duration_variance, a=0, b=1, c=0, d=1)
        rate_variance = ScaleLinLin(input=rate_variance, a=0, b=1, c=0, d=2)
        trigger_variance = ScaleLinLin(input=trigger_variance, a=0, b=1, c=0, d=1)
        A = ScaleLinLin(input=A, a=0, b=1, c=0.001, d=0.1)
        # D = ScaleLinLin(input=D, a=0, b=1, c=0.01, d=0.3)
        S = ScaleLinLin(input=S, a=0, b=1, c=0.5, d=3)
        R = ScaleLinLin(input=R, a=0, b=1, c=0.6, d=2)

        # define synthesizer
        # env = ADSREnvelope(A, D, S, R, gate=gate)
        env = ASREnvelope(A, S, R)
        grain_frequency = WhiteNoise(frequency=frequency, 
                                     min=frequency-frequency*trigger_variance, 
                                     max=frequency+frequency*trigger_variance)
        clock = Impulse(frequency=grain_frequency)
        granulator = Granulator(buffer = buf,
                                clock = clock,
                                pan = -1,
                                amplitude = 1.0,
                                pos = position + RandomUniform(-1*position_variance, position_variance, clock=clock),
                                rate = rate + RandomUniform(-1*rate_variance, rate_variance, clock=clock),
                                duration = duration + RandomUniform(-1*duration_variance, duration_variance, clock=clock),
                                max_grains = 2048,
                                wrap = True)
        granulator.set_buffer("envelope", EnvelopeBuffer("triangle"))
        output = granulator * env * gain

        # forward
        self.set_trigger_node(env)
        self.set_output(output)
        # self.set_auto_free(True)
        self.set_auto_free(False)

if __name__ == "__main__":

    # AUDIO GRAPH
    config = AudioGraphConfig()
    config.output_buffer_size = 256
    graph = AudioGraph(config=config, start=True)

    # GET SYNTH PROPERTIES
    synth = Granular()
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
        voice = Granular()
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


