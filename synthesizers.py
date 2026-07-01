from signalflow import *
import numpy as np
import soundfile as sf
import os


class AbstractTriggeredSynth:
	'''A synthesizer that loads a synthesis patch'''
	def __init__(self, 
				synth_name: str, 
				sample_rate=44100,
				window_size = 2048,
				hop_size = 512, 
				buffer_size = 0.8
				):

		self.config = AudioGraphConfig()
		self.config.sample_rate = sample_rate
		self.window_size = window_size
		self.hop_size = hop_size
		self.buffer_size = buffer_size
		self.synth_name = synth_name
		self.fade_time = 0.005
		self.connectGraph()

	def connectGraph(self):
		self.graph = AudioGraph(self.config, output_device=AudioOut_Dummy(2), start=False)
		if self.synth_name == 'Theremin':
			self.synth = ASRTheremin()
		elif self.synth_name == 'Snare':
			self.synth = Snare()
		self.graph.play(self.synth)
		self.buffer = Buffer(1, int(self.graph.sample_rate * self.buffer_size))

	def resetGraph(self):
		if self.graph:
			self.graph.destroy()
		self.connectGraph()

	def close(self):
		self.graph.destroy()

	def forward(self):
		# render audio producing one onset
		# self.synth.set_input("gate", Impulse(frequency=0)) 
		self.synth.trigger()
		self.graph.render_to_buffer(self.buffer)
		rendered_array = self.buffer.data[0,:]
		return rendered_array

	def fadeParameters(self, new_parameters: list):
		for i, param_name in enumerate(list(self.synth.inputs.keys())):
			current_p = self.synth.inputs[param_name].value
			p_fade = Line(current_p, new_parameters[i], self.fade_time)
			self.synth.set_input(param_name, p_fade) 

	def resetParameters(self, new_parameters: list):
		for i, param_name in enumerate(list(self.synth.inputs.keys())):
			self.synth.set_input(param_name, new_parameters[i]) 

	def getInputs(self):
		return self.synth.inputs



class ASRTheremin(Patch):
	def __init__(self, parameter_values: list = [0,0]):
		super().__init__()

		frequency = self.add_input(f"parameter-0", parameter_values[0])
		gain = self.add_input(f"parameter-1", parameter_values[1])
		# gate = self.add_input(f"gate", 0)

		# input parameters are all normalized between 0 and 1
		frequency = ScaleLinLin(input=frequency, a=0, b=1, c=220, d=660)
		# gain = ScaleLinLin(input=gain, a=0, b=1, c=0, d=1)

		# create synth
		sine = SineOscillator(frequency)
		# env = ASREnvelope(0.01, 0.1, 0.5, clock=gate)
		env = ASREnvelope(0.01, 0.1, 0.5)
		# output = sine * env * DecibelsToAmplitude(gain)
		output = sine * env * gain

		# forward
		self.set_trigger_node(env)
		self.set_output(output)
		self.set_auto_free(True)



class KarplusStrong(Patch): # BUGGS !!!
	'''A synthesis patch with N continuous parameters between 0 and 1'''
	def __init__(self, parameter_values, graph):
		super().__init__()

		MIDI = self.add_input("parameter-0", parameter_values[0])
		damping_MIDI = self.add_input("parameter-1", parameter_values[1])
		feedback_amt = self.add_input("parameter-2", parameter_values[2])
		gain = self.add_input("parameter-3", parameter_values[3])
		# gate = self.add_input("gate", 0)

		MIDI = ScaleLinLin(input=MIDI, a=0, b=1, c=20, d=50)
		damping_MIDI = ScaleLinLin(input=damping_MIDI, a=0, b=1, c=0, d=127)
		feedback_amt = ScaleLinLin(input=feedback_amt, a=0, b=1, c=0.5, d=0.99)
		# gain = ScaleLinLin(input=gain, a=0, b=1, c=0, d=1)

		# create synth
		frequency = MidiNoteToFrequency(MIDI)
		damping = MidiNoteToFrequency(damping_MIDI)

		feedback = Buffer(1, 4086)
		delay = Buffer(1, graph.sample_rate * 100)
		noise = WhiteNoise(frequency=0.0, min=-1.0, max=1.0)
		noise.set_seed(666)
		env = ASREnvelope(0.005, 0.01, 0.005)
		graph.add_node(FeedbackBufferWriter(delay, noise * env + FeedbackBufferReader(feedback), 1/frequency)) # graph.output_buffer_size / graph.sample_rate

		delayout = FeedbackBufferReader(delay)
		filter_out = SVFilter(input=delayout, filter_type='low_pass', cutoff=damping, resonance=0.1)
		graph.add_node(FeedbackBufferWriter(feedback, filter_out * feedback_amt, graph.output_buffer_size / graph.sample_rate))
		output = filter_out * gain

		# forward
		self.set_trigger_node(env)
		self.set_output(output)
		self.set_auto_free(True)



class Snare(Patch):
	def __init__(self, parameter_values: list = [0,0,0,0]):
		super().__init__()

		decay = self.add_input(f"parameter-0", parameter_values[0])
		cutoff_scale = self.add_input(f"parameter-1", parameter_values[1])
		center_frequency = self.add_input(f"parameter-2", parameter_values[2])
		gain = self.add_input(f"parameter-3", parameter_values[3])

		cutoff_scale = ScaleLinLin(input=cutoff_scale, a=0, b=1, c=100, d=10100)
		decay = ScaleLinLin(input=decay, a=0, b=1, c=0.1, d=0.99)
		center_frequency = ScaleLinLin(input=center_frequency, a=0, b=1, c=70, d=200)
		# frequency = ScaleLinLin(input=frequency, a=0, b=1, c=0, d=1)

		# center_frequency = 150
		modulator_frequency = 60
		carrier = SineOscillator(modulator_frequency)
		FM = SineOscillator(center_frequency + carrier * 30)
		noise = WhiteNoise(frequency=0.0, min=-1.0, max=1.0)
		noise.set_seed(666)
		env = ASREnvelope(0.0005, 0, decay)
		# cutoff_env = ASREnvelope(0.0005, 0, decay*2, clock=gate)
		out = noise * env * 0.15
		filter_out = SVFilter(input=out, filter_type='high_pass', cutoff=100, resonance=0.1)
		filter_1 = SVFilter(input=filter_out + FM * env * env, filter_type='band_pass', cutoff=(env**2)*cutoff_scale+100, resonance=0.6)
		filter_2 = SVFilter(input=filter_out + FM * env * env, filter_type='band_pass', cutoff=(env**2)*cutoff_scale*0.6+200, resonance=0.8)
		output = (filter_1 + filter_2) * gain

		# forward
		self.set_trigger_node(env)
		# self.set_trigger_node(cutoff_env)
		self.set_output(output)
		self.set_auto_free(False)


if __name__ == "__main__":

	sr, window_size, hop_size = 44100, 2048, 1024
	N_params = 4
	synth_parameters = np.random.rand(N_params).tolist()
	print(synth_parameters)
	synth = ADSRTheremin(synth_parameters, sr, window_size, hop_size)
	# synth = KarplusStrong(synth_parameters, sr, window_size, hop_size)
	# synth = SnareSynth(synth_parameters, sr, window_size, hop_size)

	synth = AbstractTriggeredSynth('Snare', sr, window_size, hop_size)
	sf.write('prova_synth1.wav', synth.forward(), sr, subtype='PCM_24')
	synth_parameters = np.random.rand(N_params).tolist()
	print(synth_parameters)
	synth.resetGraph()
	synth.resetParameters(synth_parameters)
	sf.write('prova_synth2.wav', synth.forward(), sr, subtype='PCM_24')
	synth.close()

