from signalflow import *

if __name__ == "__main__":

    # AUDIO GRAPH
    config = AudioGraphConfig()
    config.output_buffer_size = 256
    graph = AudioGraph(config=config, start=True)

    f0 = RandomExponential(40, 2000, clock=RandomImpulse(1))

    buf = Buffer(1, graph.sample_rate)
    feedback = FeedbackBufferReader(buf)
    op0 = SineOscillator(f0 + f0 * feedback * 14)

    level = Smooth(RandomUniform(0, 1, clock=RandomImpulse([0.5, 0.5])))
    level = ScaleLinExp(level, 0, 1, 0.0001, 1.0)

    op0 = op0 * level
    graph.add_node(FeedbackBufferWriter(buf, op0, 0.5))

    op1 = SineOscillator(f0 + f0 * op0 * 14)

    graph.play(op1)
    graph.wait()