from signalflow import *
import time
import numpy as np
import random


class KarplusStrong(Patch):
    def __init__(self, graph, parameter_values: list = [0,0,0,0,0,0,0]):
        super().__init__()

        MIDI = self.add_input(f"MIDI", parameter_values[0])
        damping_MIDI = self.add_input(f"damping_MIDI", parameter_values[1])
        feedback_amt = self.add_input(f"feedback_amt", parameter_values[2])
        A = self.add_input(f"A", parameter_values[3])
        S = self.add_input(f"S", parameter_values[4])
        R = self.add_input(f"R", parameter_values[5])
        gain = self.add_input(f"gain", parameter_values[6])

        # input parameters are all normalized between 0 and 1
        MIDI = ScaleLinLin(MIDI, 0, 1, 40, 50)
        damping_MIDI = ScaleLinLin(damping_MIDI, 0, 1, 90, 127)
        feedback_amt = ScaleLinLin(feedback_amt, 0, 1, 0.7, 0.8)
        A = ScaleLinLin(A, 0, 1, 0.0005, 0.01)
        S = ScaleLinLin(S, 0, 1, 0.0005, 0.001)
        R = ScaleLinLin(R, 0, 1, 0.0005, 5)

        # create synth
        frequency = MidiNoteToFrequency(MIDI)
        damping = MidiNoteToFrequency(damping_MIDI)

        feedback = Buffer(1, 4086)
        delay = Buffer(1, graph.sample_rate * 100)
        noise = WhiteNoise(frequency=0.0, min=-1.0, max=1.0)
        env = ASREnvelope(A, S, R)
        graph.add_node(FeedbackBufferWriter(delay, noise * env + FeedbackBufferReader(feedback), 1/frequency))
        
        delayout = FeedbackBufferReader(delay)
        filter_out = SVFilter(input=delayout, filter_type='low_pass', cutoff=damping, resonance=0.1)
        graph.add_node(FeedbackBufferWriter(feedback, filter_out * feedback_amt, graph.output_buffer_size / graph.sample_rate))
        output = filter_out * gain

        # forward
        self.set_trigger_node(env)
        self.set_output(output)
        self.set_auto_free(True)

if __name__ == "__main__":

    # AUDIO GRAPH
    config = AudioGraphConfig()
    config.output_buffer_size = 256
    graph = AudioGraph(config=config, start=True)

    # GET SYNTH PROPERTIES
    synth = KarplusStrong()
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
        voice = KarplusStrong()
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


