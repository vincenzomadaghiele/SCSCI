from signalflow import *
import time
import numpy as np
import random

class Granular(Patch):
    def __init__(self, parameter_values: list = [0,0,0,0,0,0,0,0,0,0,0]):
        super().__init__()

        rate = self.add_input("rate", parameter_values[0])
        duration = self.add_input("grain-duration", parameter_values[1])
        position = self.add_input("start-position", parameter_values[2])
        frequency = self.add_input("frequency", parameter_values[3])
        position_variance = self.add_input("position_variance", parameter_values[4])
        duration_variance = self.add_input("duration_variance", parameter_values[5])
        gain = self.add_input("gain", parameter_values[6])
        A = self.add_input(f"A", parameter_values[7])
        D = self.add_input(f"D", parameter_values[8])
        S = self.add_input(f"S", parameter_values[9])
        R = self.add_input(f"R", parameter_values[10])
        gate = self.add_input("gate", 1.0)

        # rate = 0.5
        # duration = 0.5
        # position = 0.5
        # frequency = 0.5
        # position_variance = 0.5
        # duration_variance = 0.5
        # gain = 0.2
        # A, D, S, R = 0.1, 0.3, 0.4, 0.3

        buf = Buffer("bowed-guitar.wav")
        rate = ScaleLinLin(input=rate, a=0, b=1, c=0.1, d=2.1)
        duration = ScaleLinLin(input=duration, a=0, b=1, c=0.1, d=2)
        position = ScaleLinLin(input=position, a=0, b=1, c=0.01, d=buf.duration-2+0.01)
        frequency = ScaleLinLin(input=frequency, a=0, b=1, c=50, d=200)
        position_variance = ScaleLinLin(input=position_variance, a=0, b=1, c=0, d=0.7)
        duration_variance = ScaleLinLin(input=duration_variance, a=0, b=1, c=0, d=0.7)
        A = ScaleLinLin(input=A, a=0, b=1, c=0.001, d=0.1)
        D = ScaleLinLin(input=D, a=0, b=1, c=0.1, d=0.9)
        S = ScaleLinLin(input=S, a=0, b=1, c=0.3, d=3)
        R = ScaleLinLin(input=R, a=0, b=1, c=0.1, d=0.8)

        clock = Impulse(frequency=frequency)
        granulator = Granulator(buffer=buf,
                                clock=clock,
                                pan=0,
                                pos=RandomUniform(position-position_variance, position+position_variance, clock=clock),
                                rate=rate,
                                duration=RandomUniform(duration-duration_variance, duration+duration_variance, clock=clock),
                                wrap=True)
        granulator.set_buffer("envelope", EnvelopeBuffer("triangle"))
        env = ADSREnvelope(A, D, S, R, gate=gate)
        output = granulator * env * gain

        # forward
        # self.set_trigger_node(env)
        self.set_output(output)
        self.set_auto_free(True)


if __name__ == "__main__":

    config = AudioGraphConfig()
    config.output_buffer_size = 256
    graph = AudioGraph(config=config, start=True)

    voices = [ None ] * 128
    durations = np.zeros(128)

    synth = Granular()
    param_names = list(synth.inputs.keys())
    param_names.remove("gate")
    param_names.remove("gain")
    N_params = len(param_names)

    j = 0
    metro = 0.4
    while True:
        voice = Granular()
        new_parameters = np.random.rand(N_params).tolist()
        for i, param_name in enumerate(param_names):
            voice.set_input(param_name, new_parameters[i]) 
        voice.set_input("gain", 0.4)
        voice.set_input("gate", 1)
        voice.auto_free = True
        voice.play()
        voices[j] = voice
        durations[j] = random.random() * 4
        j += 1
        if j == 128:
            j = 0
        durations[np.argwhere(durations > 0)] = durations[np.argwhere(durations > 0)] - metro
        ended_notes = np.argwhere(durations < 0).T
        for k in ended_notes.tolist()[0]:
            voices[k].set_input("gate", 0)
            durations[k] = 0
        time.sleep(metro)


    # patch = Granular()
    # spec = patch.to_spec()
    # voices = [ None ] * 128
    # durations = np.zeros(128)
    # j=0
    # # while True:
    # voice = Patch(spec)
    # voice.auto_free = True
    # voice.play()
    # voice.set_input("gate", 1)
    # voices[j] = voice
    # time.sleep(1)
    # voice.set_input("gate", 0)
    # # durations[j] = random.random() * 2
    # # j += 1
    # # if j == 128:
    # #     j = 0



    # rate = 0.5
    # duration = 0.5
    # position = 0.5

    # buf = Buffer("bowed-guitar.wav")
    # rate = ScaleLinLin(input=rate, a=0, b=1, c=0.1, d=2.1)
    # duration = ScaleLinLin(input=duration, a=0, b=1, c=0.1, d=2)
    # position = ScaleLinLin(input=position, a=0, b=1, c=0.01, d=buf.duration-2+0.01)

    # frequency = 100
    # position_variance = 0.5
    # duration_variance = 0.5
    # clock = Impulse(frequency=frequency)
    # granulator = Granulator(buffer=buf,
    #                         clock=clock,
    #                         pan=0,
    #                         pos=RandomUniform(position-position_variance, position+position_variance, clock=clock),
    #                         rate=rate,
    #                         duration=RandomUniform(duration-duration_variance, duration+duration_variance, clock=clock),
    #                         wrap=True)
    # granulator.set_buffer("envelope", EnvelopeBuffer("triangle"))
    # env = ADSREnvelope(0.001, 0.5, 0.9, 0.2, gate=0)
    # output = granulator * env
    # output.play()

    # metro = 2
    # while True:
    #     env.set_input("gate", 0)
    #     # time.sleep(0.01)
    #     new_parameters = np.random.rand(10).tolist()
    #     rate = new_parameters[0]
    #     duration = new_parameters[1]
    #     position = new_parameters[2]
    #     frequency = new_parameters[3]
    #     position_variance = new_parameters[4]
    #     duration_variance = new_parameters[5]
    #     A = new_parameters[6]
    #     D = new_parameters[7]
    #     S = new_parameters[8]
    #     R = new_parameters[9]
    #     clock = Impulse(frequency=frequency)
    #     rate = ScaleLinLin(input=rate, a=0, b=1, c=0.1, d=2.1)
    #     duration = ScaleLinLin(input=duration, a=0, b=1, c=0.1, d=2)
    #     position = ScaleLinLin(input=position, a=0, b=1, c=0.01, d=buf.duration-2+0.01)
    #     granulator.set_input("pos", RandomUniform(position-position_variance, position+position_variance, clock=clock))
    #     granulator.set_input("rate", rate)
    #     granulator.set_input("duration", RandomUniform(duration-duration_variance, duration+duration_variance, clock=clock))
    #     A = ScaleLinLin(input=A, a=0, b=1, c=0.001, d=0.01)
    #     D = ScaleLinLin(input=D, a=0, b=1, c=0.1, d=0.6)
    #     S = ScaleLinLin(input=S, a=0, b=1, c=0.3, d=1)
    #     R = ScaleLinLin(input=R, a=0, b=1, c=0.1, d=0.3)
    #     env.set_input("attack", A)
    #     env.set_input("decay", D)
    #     env.set_input("sustain", S)
    #     env.set_input("release", R)
    #     env.set_input("gate", 1)
    #     time.sleep(metro)
