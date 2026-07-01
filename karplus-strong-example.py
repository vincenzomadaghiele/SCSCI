from signalflow import *
import numpy as np
import random
import time
import os
# import soundfile as sf


class KarplusStrongPatch(Patch):
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
        self.set_trigger_node(env)
        self.set_output(output)
        self.set_auto_free(True)


class Granular(Patch):
    def __init__(self, parameter_values: list = [0,0,0,0,0,0,0,0]):
        super().__init__()

        rate = self.add_input("rate", parameter_values[0])
        duration = self.add_input("grain-duration", parameter_values[1])
        position = self.add_input("start-position", parameter_values[2])
        gain = self.add_input("gain", parameter_values[3])
        A = self.add_input(f"A", parameter_values[4])
        D = self.add_input(f"D", parameter_values[5])
        S = self.add_input(f"S", parameter_values[6])
        R = self.add_input(f"R", parameter_values[7])
        gate = self.add_input("gate", 1.0)

        # audio_path = os.path.join(os.path.dirname(__file__), "", "bowed-guitar.wav")
        buf = Buffer("bowed-guitar.wav")
        # buf = Buffer("tibetan-ball.wav")
        rate = ScaleLinLin(input=rate, a=0, b=1, c=0.1, d=2.1)
        duration = ScaleLinLin(input=duration, a=0, b=1, c=0.1, d=2)
        position = ScaleLinLin(input=position, a=0, b=1, c=0.01, d=buf.duration-2+0.01)
        A = ScaleLinLin(input=A, a=0, b=1, c=0.001, d=0.01)
        D = ScaleLinLin(input=D, a=0, b=1, c=0.1, d=0.6)
        S = ScaleLinLin(input=S, a=0, b=1, c=0.3, d=1)
        R = ScaleLinLin(input=R, a=0, b=1, c=0.1, d=0.3)

        # create synth
        frequency = 100
        position_variance = 0
        duration_variance = 0
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
        output = granulator * gain * env

        # forward
        self.set_trigger_node(env)
        self.set_output(output)
        self.set_auto_free(True)


if __name__ == "__main__":

    config = AudioGraphConfig()
    config.output_buffer_size = 128
    graph = AudioGraph(config=config, start=True)

    patch = Granular()
    spec = patch.to_spec()
    voices = [ None ] * 128
    durations = np.zeros(128)
    # print(spec.to_json())

    j = 0
    metro = 0.5
    while True:
        voice = Patch(spec)
        N_params = len(list(voice.inputs.keys()))
        new_parameters = np.random.rand(N_params).tolist()
        new_parameters[-2] = 0.8
        new_parameters[-1] = 1
        print([f'{p:.2f}' for p in new_parameters])
        for i, param_name in enumerate(list(voice.inputs.keys())):
            voice.set_input(param_name, new_parameters[i]) 
        # voice.set_input("gate", 1)
        voice.auto_free = True
        voice.play()
        voices[j] = voice
        durations[j] = random.random() * 2
        j += 1
        if j == 128:
            j = 0
        durations[np.argwhere(durations > 0)] = durations[np.argwhere(durations > 0)] - metro
        ended_notes = np.argwhere(durations < 0).T
        for k in ended_notes.tolist()[0]:
            voices[k].set_input("gate", 0)
            durations[k] = 0
        time.sleep(metro)
