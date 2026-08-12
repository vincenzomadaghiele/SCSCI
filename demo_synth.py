from signalflow import *
import time
import numpy as np
import random
import soundfile as sf


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
        output = (output1 + output2) * env * gain

        # forward
        self.set_output(output)
        self.set_auto_free(True)
        # self.set_trigger_node(env)


if __name__ == "__main__":

    # offline graph
    graph = AudioGraph(output_device=AudioOut_Dummy(2), start=False)

    # SYNTH
    synth = TimbreTheremin() 
    parameters_names = list(synth.inputs.keys())
    # THESE ARE THE SAME FOR ALL SYNTHS
    frequency_param_name = "frequency"
    gain_param_name = "gain"
    gate_param_name = "gate"
    duration_param_name = "D"
    envelope_param_names = ['A','S','R']
    # THESE CAN CHANGE AMONG SYNTHS
    non_timbre_params_names = [frequency_param_name, gain_param_name, gate_param_name, duration_param_name]
    non_timbre_params_names += envelope_param_names
    print(non_timbre_params_names)
    timbre_param_names = [p for p in parameters_names if p not in non_timbre_params_names]
    print(timbre_param_names)
    # SET CONSTANT NON-TIMBRE PARAMS
    frequency = 440
    gain = 0.8
    event_duration = 0.1 # seconds

    buffer_size = 1
    buffer_event = Buffer(1, int(graph.sample_rate * event_duration))
    buffer_decay = Buffer(1, int(graph.sample_rate * (buffer_size-event_duration)))

    # set non-timbre parameters
    synth.set_input(frequency_param_name, frequency) 
    synth.set_input(gain_param_name, gain) 
    synth.set_input(duration_param_name, event_duration) 
    synth.set_input(envelope_param_names[0], 0.2) 
    synth.set_input(envelope_param_names[1], 0.2) 
    synth.set_input(envelope_param_names[2], 0.2) 
    # set timbre parameters
    timbre_parameters_values = np.random.rand(len(timbre_param_names)).tolist()
    for i, param_name in enumerate(timbre_param_names):
        synth.set_input(param_name, timbre_parameters_values[i]) 
    synth.set_input("gate", 1)
    synth.play()
    graph.render_to_buffer(buffer_event)
    rendered_event = buffer_event.data[0,:]
    synth.set_input("gate", 0) # record decay
    graph.render_to_buffer(buffer_decay)
    rendered_decay = buffer_decay.data[0,:]
    rendered_array = np.concatenate((rendered_event, rendered_decay), axis=0)
    sf.write(f'timbre_theremin.wav', rendered_array, 44100, subtype='PCM_24')
