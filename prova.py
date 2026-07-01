from signalflow import *
import time
import numpy as np
import random


class Snare(Patch):
    def __init__(self, parameter_values: list = [0,0,0,0,0,0,0,0,0]):
        super().__init__()

        decay = self.add_input(f"decay", parameter_values[0])
        cutoff_scale = self.add_input(f"cutoff_scale", parameter_values[1])
        center_frequency = self.add_input(f"center_frequency", parameter_values[2])
        modulator_frequency = self.add_input(f"modulator_frequency", parameter_values[3])
        gain = self.add_input(f"gain", parameter_values[4])
        A = self.add_input(f"A", parameter_values[5])
        D = self.add_input(f"D", parameter_values[6])
        S = self.add_input(f"S", parameter_values[7])
        R = self.add_input(f"R", parameter_values[8])
        gate = self.add_input("gate", 1.0)

        cutoff_scale = ScaleLinLin(input=cutoff_scale, a=0, b=1, c=100, d=10100)
        # decay = ScaleLinLin(input=decay, a=0, b=1, c=0.1, d=0.99)
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
        out = noise * env * 0.15
        filter_out = SVFilter(input=out, filter_type='high_pass', cutoff=100, resonance=0.1)
        filter_1 = SVFilter(input=filter_out + FM * env * env, filter_type='band_pass', cutoff=(env**2)*cutoff_scale+100, resonance=0.6)
        filter_2 = SVFilter(input=filter_out + FM * env * env, filter_type='band_pass', cutoff=(env**2)*cutoff_scale*0.6+200, resonance=0.8)
        output = (filter_1 + filter_2) * gain

        # forward
        self.set_output(output)
        self.set_auto_free(False)


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
        self.set_auto_free(False)

class Benjolin(Patch):
    def __init__(self, graph: AudioGraph, parameter_values: list = [0,0,0,0,0,0,0,0,0,0,0,0,0]):
        super().__init__()

        FRQ01 = self.add_input("FRQ01", parameter_values[0])
        FRQ02 = self.add_input("FRQ02", parameter_values[1])
        RUN01 = self.add_input("RUN01", parameter_values[2])
        RUN02 = self.add_input("RUN02", parameter_values[3])
        FIL_FRQ = self.add_input("FIL_FRQ", parameter_values[4])
        FIL_RES = self.add_input("FIL_RES", parameter_values[5])
        FIL_RUN = self.add_input("FIL_RUN", parameter_values[6])
        FIL_SWP = self.add_input("FIL_SWP", parameter_values[7])
        gain = self.add_input("gain", parameter_values[8])
        A = self.add_input(f"A", parameter_values[9])
        D = self.add_input(f"D", parameter_values[10])
        S = self.add_input(f"S", parameter_values[11])
        R = self.add_input(f"R", parameter_values[12])
        gate = self.add_input("gate", 1.0)

        # RUN01 = 0.1
        # FIL_RES = 0.2
        # # FIL_SWP = 0.3
        # FIL_RUN = 0.5

        ## PARAMETERS
        FRQ01 = ScaleLinLin(input=FRQ01, a=0, b=1, c=0, d=127)
        FRQ02 = ScaleLinLin(input=FRQ02, a=0, b=1, c=0, d=127)
        RUN01 = ScaleLinLin(input=RUN01, a=0, b=1, c=0, d=127)
        RUN02 = ScaleLinLin(input=RUN02, a=0, b=1, c=0, d=127)
        FIL_FRQ = ScaleLinLin(input=FIL_FRQ, a=0, b=1, c=0, d=127)
        FIL_RES = ScaleLinLin(input=FIL_RES, a=0, b=1, c=0, d=127)
        FIL_RUN = ScaleLinLin(input=FIL_RUN, a=0, b=1, c=0, d=127)
        FIL_SWP = ScaleLinLin(input=FIL_SWP, a=0, b=1, c=0, d=127)
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
        output = (RUNGLER + PWM) * 0.5

        ## FILTER
        filter_f0 = FIL_FRQ + (FIL_RUN * RUNGLER) + (FIL_SWP * O2_tri)
        filter_f0 = MidiNoteToFrequency(a = Clip(input=filter_f0, min=0, max=127))
        filter_res = ScaleLinLin(input=FIL_RES, a=0, b=127, c=0.1, d=0.9)
        filter_out = SVFilter(input=output, filter_type='low_pass', cutoff=filter_f0, resonance=filter_res)
        output = filter_out * ScaleLinLin(input=FIL_RES, a=0, b=127, c=2, d=12)
        output = SVFilter(input=output, filter_type='high_pass', cutoff=10, resonance=0.0)
        output = Compressor(input=output, threshold=0.2, ratio=4, attack_time=0.01, release_time=0.1)
        output = output * gain * env

        self.output = output
        self.set_auto_free(False)


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

        # scaling parameters
        buf = Buffer("bowed-guitar.wav")
        rate = ScaleLinLin(input=rate, a=0, b=1, c=0.1, d=2.1)
        duration = ScaleLinLin(input=duration, a=0, b=1, c=0.1, d=2)
        position = ScaleLinLin(input=position, a=0, b=1, c=0.01, d=buf.duration-2+0.01)
        frequency = ScaleLinLin(input=frequency, a=0, b=1, c=50, d=100)
        position_variance = ScaleLinLin(input=position_variance, a=0, b=1, c=0, d=0.7)
        duration_variance = ScaleLinLin(input=duration_variance, a=0, b=1, c=0, d=0.7)
        A = ScaleLinLin(input=A, a=0, b=1, c=0.001, d=0.1)
        D = ScaleLinLin(input=D, a=0, b=1, c=0.1, d=0.9)
        S = ScaleLinLin(input=S, a=0, b=1, c=0.5, d=3)
        R = ScaleLinLin(input=R, a=0, b=1, c=0.6, d=2)

        # define synthesizer
        env = ADSREnvelope(A, D, S, R, gate=gate)
        clock = Impulse(frequency=frequency/2+frequency*env)
        granulator = Granulator(buffer=buf,
                                clock=clock,
                                pan=0,
                                pos=RandomUniform(position-position_variance, position+position_variance, clock=clock),
                                rate=rate,
                                duration=RandomUniform(duration-duration_variance, duration+duration_variance, clock=clock),
                                wrap=True)
        granulator.set_buffer("envelope", EnvelopeBuffer("triangle"))
        output = granulator * env * gain

        # forward
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
        self.set_output(output)
        self.set_auto_free(True)


if __name__ == "__main__":

    # AUDIO GRAPH
    config = AudioGraphConfig()
    config.output_buffer_size = 256
    graph = AudioGraph(config=config, start=True)

    # GET SYNTH PROPERTIES
    synth_name = 'Granular'
    if synth_name == 'Granular':
        synth = Granular()
    elif synth_name == 'Benjolin':
        synth = Benjolin(graph)
    elif synth_name == 'Snare':
        synth = Snare()
    else:
        synth = ADSRTheremin()
    param_names = list(synth.inputs.keys())
    param_names.remove("gate")
    param_names.remove("gain")
    N_params = len(param_names)

    # PLAYBACK PROPERTIES
    metro = 0.6
    random_metro = 0.5
    NUM_VOICES = 30
    MAX_DURATION = 2
    voices = [ None ] * NUM_VOICES
    durations = np.zeros(NUM_VOICES)

    # PLAY
    j = 0
    while True:
        if synth_name == 'Granular':
            voice = Granular()
        elif synth_name == 'Benjolin':
            voice = Benjolin(graph)
        elif synth_name == 'Snare':
            voice = Snare()
        else:
            voice = ADSRTheremin()
        new_parameters = np.random.rand(N_params).tolist()
        for i, param_name in enumerate(param_names):
            voice.set_input(param_name, new_parameters[i]) 
        voice.set_input("gain", 1/NUM_VOICES)
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


