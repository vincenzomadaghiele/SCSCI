from signalflow import *
import time
import numpy as np
import random
import math

class Granular(Patch):
    def __init__(self, sample_path: str, base_pitch_hz: float = None, 
                 parameter_values: list = [0,0,0,0,0,0,0,0,0,0,0,0,0]):
        super().__init__()

        density = self.add_input("density", parameter_values[0])
        duration = self.add_input("grain-duration", parameter_values[1])
        position = self.add_input("start-position", parameter_values[2])
        rate = self.add_input("frequency", parameter_values[3]) # in Hz
        position_variance = self.add_input("position_variance", parameter_values[4])
        duration_variance = self.add_input("duration_variance", parameter_values[5])
        rate_variance = self.add_input("rate_variance", parameter_values[6])
        trigger_variance = self.add_input("trigger_variance", parameter_values[7])
        gain = self.add_input("gain", parameter_values[8])
        A = self.add_input(f"A", parameter_values[9])
        D = self.add_input(f"D", parameter_values[10])
        S = self.add_input(f"S", parameter_values[11])
        R = self.add_input(f"R", parameter_values[12])
        gate = self.add_input("gate", 1.0)

        # scale parameters
        # buf = Buffer("./synth/patches/samples/bowed-guitar.wav")
        # buf = Buffer("./synth/patches/samples/tibetan-ball.wav")
        buf = Buffer(sample_path)
        # frequency = ScaleLinExp(input=frequency, a=50, b=11050, c=1, d=50)
        # gain = -1 *(gain - 1)*(gain - 1) + 1 # exponential gain
        rate = Clip(rate, 50, 11050)
        density = ScaleLinExp(input=density, a=0, b=1, c=1, d=200)
        duration = ScaleLinExp(input=duration, a=0, b=1, c=0.1, d=2)
        position = ScaleLinLin(input=position, a=0, b=1, c=0, d=int(buf.duration - (buf.duration/2)))
        position_variance = ScaleLinLin(input=position_variance, a=0, b=1, c=0, d=int(buf.duration/4))
        duration_variance = ScaleLinLin(input=duration_variance, a=0, b=1, c=0, d=1)
        rate_variance = ScaleLinLin(input=rate_variance, a=0, b=1, c=0, d=0.1)
        trigger_variance = ScaleLinLin(input=trigger_variance, a=0, b=1, c=0, d=1)
        A = ScaleLinLin(input=A, a=0, b=1, c=0.001, d=0.1)
        D = ScaleLinLin(input=D, a=0, b=1, c=0.01, d=0.3)
        S = ScaleLinLin(input=S, a=0, b=1, c=0.5, d=3)
        R = ScaleLinLin(input=R, a=0, b=1, c=0.6, d=2)

        # define synthesizer
        env = ADSREnvelope(A, D, S, R, gate=gate)
        grain_frequency = WhiteNoise(frequency=density, 
                                     min=density-density*trigger_variance, 
                                     max=density+density*trigger_variance)
        grain_frequency.set_seed(666)

        base_pitch_hz = 388
        if base_pitch_hz:
            rate = rate / base_pitch_hz # rate [50-11050]
        else:
            rate = ScaleLinExp(input=rate, a=0, b=1, c=0.3, d=4)
        clock = Impulse(frequency=grain_frequency) * env
        granulator = Granulator(buffer = buf,
                                clock = clock,
                                pan = -1,
                                amplitude = 1.0,
                                pos = position + RandomUniform(-1*position_variance, position_variance, clock=clock) * env,
                                rate = rate + RandomUniform(-1*rate_variance, rate_variance, clock=clock) * env,
                                duration = duration + RandomUniform(-1*duration_variance, duration_variance, clock=clock) * env,
                                max_grains = 8,
                                wrap = True)
        granulator.set_buffer("envelope", EnvelopeBuffer("triangle"))
        output = granulator * env * gain * 0.5

        # forward
        self.set_output(output)
        self.set_auto_free(True)

if __name__ == "__main__":

    # AUDIO GRAPH
    config = AudioGraphConfig()
    config.output_buffer_size = 256
    config.output_device_name = "Scarlett 2i2 USB"
    config.input_device_name = "Scarlett 2i2 USB"
    graph = AudioGraph(config=config, start=True)

    # open sample and find base pitch
    import librosa
    # sample_path = "./synth/patches/samples/bowed-guitar.wav"
    sample_path = "./synth/patches/samples/tibetan-ball.wav"
    sample, _ = librosa.load(sample_path, sr=graph.sample_rate, mono=True)
    pitch = librosa.yin(sample, fmin=200, fmax=graph.sample_rate/2, sr=graph.sample_rate, frame_length=config.output_buffer_size*2, hop_length=config.output_buffer_size)
    pitch = [p for p in pitch.tolist() if p != graph.sample_rate]
    base_pitch_hz = np.array(pitch).mean()

    # GET SYNTH PROPERTIES
    synth = Granular(sample_path, base_pitch_hz)
    param_names = list(synth.inputs.keys())
    param_names.remove("gate")
    param_names.remove("gain")
    N_params = len(param_names)

    # PLAYBACK PROPERTIES
    metro = 1
    random_metro = 0.5
    NUM_VOICES = 10
    MAX_DURATION = 4
    voices = [ None ] * NUM_VOICES
    durations = np.zeros(NUM_VOICES)

    # PLAY
    j = 0
    while True:
        voice = Granular(sample_path, base_pitch_hz)
        new_parameters = np.random.rand(N_params).tolist()
        for i, param_name in enumerate(param_names):
            if param_name == "frequency":
                new_parameters[i] = math.log2(new_parameters[i]+1) * 1000 + 50
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
