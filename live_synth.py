from signalflow import *

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

if __name__ == "__main__":

    sr = 44100
    config = AudioGraphConfig()
    config.sample_rate = sr
    config.output_buffer_size = 256
    config.input_device_name = "MacBook Pro Microphone"
    config.output_device_name = "External Headphones"
    graph = AudioGraph(config)
