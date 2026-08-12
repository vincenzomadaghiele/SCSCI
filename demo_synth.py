from signalflow import *
import time
import numpy as np
import random


class TimbreTheremin(Patch):
    def __init__(self, parameter_values: list = [0,0,0,0,0,0]):
        super().__init__()

        frequency = self.add_input(f"frequency", parameter_values[0])
        gain = self.add_input(f"gain", parameter_values[1])
        timbre = self.add_input(f"timbre", parameter_values[2])
        A = self.add_input(f"A", parameter_values[3])
        D = self.add_input(f"D", parameter_values[3])
        S = self.add_input(f"S", parameter_values[4])
        R = self.add_input(f"R", parameter_values[5])
        gate = self.add_input("gate", 1.0)

        frequency = ScaleLinExp(input=frequency, a=0, b=1, c=50, d=11050)
        gain = ScaleLinLin(input=gain, a=0, b=1, c=0, d=1)
        timbre = ScaleLinLin(input=gain, a=0, b=1, c=0, d=1)
        A = ScaleLinLin(input=A, a=0, b=1, c=0.001, d=0.01)
        D = ScaleLinLin(input=D, a=0, b=1, c=0.1, d=0.6)
        S = ScaleLinLin(input=S, a=0, b=1, c=0.3, d=1)
        R = ScaleLinLin(input=R, a=0, b=1, c=0.1, d=0.3)

        # create synth
        sine = SineOscillator(frequency)
        tri = TriangleOscillator(frequency)
        # square = SquareOscillator(frequency)
        env = ADSREnvelope(A, D, S, R, gate=gate)
        # env = ASREnvelope(A, S, R)
        # output = sine * env * gain
        output = (tri * timbre + sine * (1-timbre)) * env * gain

        # forward
        # self.set_trigger_node(env)
        self.set_output(output)
        self.set_auto_free(True)
        # self.set_auto_free(False)


if __name__ == "__main__":

    # offline graph
    graph = AudioGraph(output_device=AudioOut_Dummy(2), start=False)

    # SYNTH
    synth = TimbreTheremin() 
    new_parameters_names = list(synth.inputs.keys())
    frequency_param_name = "frequency"
    gain_param_name = "gain"
    gate_param_name = "gate"
    duration_param_name = "D"
    envelope_param_names = ['A','S','R']
    # REMOVE THESE FROM PARAMS AND REMAINING ARE TIMBRE
    # SAVE NAMES OF TIMBRE PARAMS INTO TXT FILE
    # SET CONSTANT NON-TIMBRE PARAMS
    frequency = 440
    gain = 0.8

    # event_duration = 0.1
    # buffer_size = 1
    # buffer1 = Buffer(1, int(graph.sample_rate * event_duration))
    # buffer2 = Buffer(1, int(graph.sample_rate * (buffer_size-event_duration)))

    # new_parameters_values = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    # for i, param_name in enumerate(new_parameters_names):
    #     synth.set_input(param_name, new_parameters_values[i]) 

    # synth.set_input("gate", 1)
    # graph.render_to_buffer(buffer1)
    # rendered_array1 = buffer1.data[0,:]

    # synth.set_input("gate", 0) # record decay
    # graph.render_to_buffer(buffer2)
    # rendered_array2 = buffer2.data[0,:]

    # rendered_array = np.concatenate((rendered_array1, rendered_array2), axis=0)
    # print(rendered_array.shape)

