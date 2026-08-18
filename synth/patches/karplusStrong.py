import time
import random
import math
import numpy as np
from signalflow import *


class KarplusStrong(Patch):
    def __init__(self, graph: AudioGraph, parameter_values: list = [0,0,0,0,0,0,0,0]):
        super().__init__()

        frequency = self.add_input(f"frequency", parameter_values[0]) # Hz
        gain = self.add_input(f"gain", parameter_values[1])
        feedback = self.add_input(f"feedback", parameter_values[2])
        damping = self.add_input(f"damping", parameter_values[3])
        A = self.add_input(f"A", parameter_values[4])
        D = self.add_input(f"D", parameter_values[5])
        S = self.add_input(f"S", parameter_values[6])
        R = self.add_input(f"R", parameter_values[7])
        gate = self.add_input("gate", 1.0)

        gain = -1 *(gain - 1)*(gain - 1) + 1 # exponential gain
        feedback = ScaleLinExp(input=feedback, a=0, b=1, c=0.9, d=0.999)
        damping = ScaleLinLin(input=damping, a=0, b=1, c=500, d=5000)
        A = ScaleLinLin(input=A, a=0, b=1, c=0.001, d=0.2)
        D = ScaleLinLin(input=D, a=0, b=1, c=0.001, d=0.1)
        S = ScaleLinLin(input=S, a=0, b=1, c=0.001, d=0.002)
        R = ScaleLinLin(input=R, a=0, b=1, c=0.001, d=0.002)

        feedback_buf = Buffer(1, 2048)
        feedback_reader = FeedbackBufferReader(feedback_buf)
        noise = WhiteNoise(frequency=0.0, min=-1.0, max=1.0)
        noise.set_seed(666)
        env = ADSREnvelope(A, D, S, R, gate=gate)
        player = noise * env
        output = SVFilter(player * 0.5 + feedback_reader, "low_pass", damping)
        feedback_writer = FeedbackBufferWriter(feedback_buf, output * feedback, 1/frequency)
        graph.add_node(feedback_writer)
        out = output * gain

        self.set_output(out)
        self.set_auto_free(True)


if __name__ == "__main__":

    # AUDIO GRAPH
    config = AudioGraphConfig()
    config.output_buffer_size = 128
    graph = AudioGraph(config=config, start=True)

    # GET SYNTH PROPERTIES
    synth = KarplusStrong(graph)
    param_names = list(synth.inputs.keys())
    param_names.remove("gate")
    param_names.remove("gain")
    N_params = len(param_names)

    # PLAYBACK PROPERTIES
    metro = 0.1
    random_metro = 0.5
    NUM_VOICES = 10
    MAX_DURATION = 4
    voices = [ None ] * NUM_VOICES
    durations = np.zeros(NUM_VOICES)

    # PLAY
    j = 0
    while True:
        voice = KarplusStrong(graph)
        new_parameters = np.random.rand(N_params).tolist()
        for i, param_name in enumerate(param_names):
            if param_name == "frequency":
                new_parameters[i] = math.log2(new_parameters[i]+1) * 200 + 50
            voice.set_input(param_name, new_parameters[i]) 
        voice.set_input("gain", 1 / NUM_VOICES)
        voice.set_input("gate", 1)
        # voice.trigger()
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



    # feedback = 0.99
    # note_interval = 0.125
    # frequency = 200    

    # feedback_buf = Buffer(1, 2048)
    # feedback_reader = FeedbackBufferReader(feedback_buf)
    # noise = WhiteNoise(frequency=0.0, min=-1.0, max=1.0)
    # noise.set_seed(666)
    # env = ADSREnvelope(0.001, 0.001, 0.001, 0.001)
    # player = noise * env
    # output = SVFilter(player * 0.5 + feedback_reader, "low_pass", 4000)
    # feedback_writer = FeedbackBufferWriter(feedback_buf, output * feedback, 1/frequency)
    # out = output + feedback_writer
    # graph.play(out)

    # while True:
    #     graph.wait(note_interval)
    #     env.set_input("gate", 1)
    #     time.sleep(0.01)
    #     env.set_input("gate", 0)
    #     # env.trigger()
    #     # random_frequency.trigger()

