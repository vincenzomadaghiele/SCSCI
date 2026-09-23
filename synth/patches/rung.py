from signalflow import *
import time
import numpy as np
import random
import math

class Rungler(Patch):
    def __init__(self, PULSE1, PULSE2, graph):
        super().__init__()
        sh1, sh2, sh3, sh4, sh5, sh6, sh7, sh8 = 0, 0, 0, 0, 0, 0, 0, 0

        buf = Buffer(1, 44100)
        buf_reader = FeedbackBufferReader(buf)
        
        ## RUNGLER
        XOR = Pow(a=Clip(GreaterThan(a=PULSE1, b=0.5), min=0, max=1.0), b=buf_reader)
        clock = GreaterThan(a=PULSE2, b=0)
        sh1 = SampleAndHold(input=XOR, clock=clock)
        sh2 = SampleAndHold(input=sh1, clock=clock)
        sh3 = SampleAndHold(input=sh2, clock=clock)
        sh4 = SampleAndHold(input=sh3, clock=clock)
        sh5 = SampleAndHold(input=sh4, clock=clock)
        sh6 = SampleAndHold(input=sh5, clock=clock)
        sh7 = SampleAndHold(input=sh6, clock=clock)
        sh8 = SampleAndHold(input=sh7, clock=clock)
        graph.add_node(FeedbackBufferWriter(buf, sh8, 0.05))
        RUNGLER = (sh6/8) + (sh7/4) + (sh8/2)
        self.output = RUNGLER
        # self.set_auto_free(True)

class Rung(Patch):
    def __init__(self, graph: AudioGraph, parameter_values: list = [0,0,0,0,0,0,0,0,0,0,0,0,0]):
        super().__init__()

        FRQ01 = self.add_input("frequency", parameter_values[0])
        FRQ02 = self.add_input("FRQ02", parameter_values[1])
        RUN01 = self.add_input("RUN01", parameter_values[2])
        RUN02 = self.add_input("RUN02", parameter_values[3])
        gain = self.add_input("gain", parameter_values[4])
        A = self.add_input(f"A", parameter_values[5])
        D = self.add_input(f"D", parameter_values[6])
        S = self.add_input(f"S", parameter_values[7])
        R = self.add_input(f"R", parameter_values[8])
        gate = self.add_input("gate", 1.0)

        ## PARAMETERS
        FRQ01 = Clip(FRQ01, 50, 11050)
        FRQ01 = FrequencyToMidiNote(FRQ01)
        FRQ02 = ScaleLinLin(input=FRQ02, a=0, b=1, c=0, d=127)
        RUN01 = ScaleLinLin(input=RUN01, a=0, b=1, c=0, d=127)
        RUN02 = ScaleLinLin(input=RUN02, a=0, b=1, c=0, d=127)
        A = ScaleLinLin(input=A, a=0, b=1, c=0.001, d=0.01)
        D = ScaleLinLin(input=D, a=0, b=1, c=0.1, d=0.9)
        S = ScaleLinLin(input=S, a=0, b=1, c=0.5, d=1)
        R = ScaleLinLin(input=R, a=0, b=1, c=0.1, d=1)
        env = ADSREnvelope(A, D, S, R, gate=gate)

        buf_pulse01 = Buffer(1, graph.output_buffer_size)
        buf_pulse02 = Buffer(1, graph.output_buffer_size)
        
        buf_pulse01_reader = FeedbackBufferReader(buf_pulse01)
        buf_pulse02_reader = FeedbackBufferReader(buf_pulse02)
        RUNGLER = Rungler(buf_pulse01_reader, buf_pulse02_reader, graph)

        ## OSCILLATOR 1
        O1 = ScaleLinLin(input=FRQ01, a=0, b=127, c=-61, d=80) + RUNGLER * (RUN01 / 2)
        O1 = MidiNoteToFrequency(a=Clip(input=O1, min=-60, max=127))
        O1_tri = TriangleOscillator(frequency=O1)
        O1_pls = SquareOscillator(frequency=O1)
        graph.add_node(FeedbackBufferWriter(buf_pulse01, O1_pls, graph.output_buffer_size / graph.sample_rate))

        ## OSCILLATOR 2
        O2 = ScaleLinLin(input=FRQ02, a=0, b=127, c=-59, d=80) + RUNGLER * (RUN02 / 2)
        O2 = MidiNoteToFrequency(a=Clip(input=O2, min=-60, max=127))
        O2_tri = TriangleOscillator(frequency=O2)
        O2_pls = SquareOscillator(frequency=O2)
        graph.add_node(FeedbackBufferWriter(buf_pulse02, O2_pls, graph.output_buffer_size / graph.sample_rate))

        PWM = GreaterThan(a=O1_tri, b=O2_tri)
        output = Tanh(RUNGLER + PWM) * 0.2
        output = output * gain * env

        self.set_output(output)
        self.set_auto_free(False)
        # self.set_auto_free(True)

if __name__ == "__main__":

    # AUDIO GRAPH
    config = AudioGraphConfig()
    config.output_buffer_size = 256
    config.output_device_name = "Scarlett 2i2 USB"
    config.input_device_name = "Scarlett 2i2 USB"
    graph = AudioGraph(config=config, start=True)

    # GET SYNTH PROPERTIES
    synth = Rung(graph)
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
        voice = Rung(graph)
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

