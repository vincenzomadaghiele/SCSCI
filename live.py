from signalflow import * 

import os
import time
import soundfile as sf
import numpy as np

if __name__ == "__main__":

    # SAMPLE RATE
    sr = 44100
    FFT_window = 2048
    hop_size = 256

    # START AUDIO GRAPH
    config = AudioGraphConfig()
    config.sample_rate = sr
    config.output_buffer_size = hop_size
    config.output_device_name = "UltraLite-mk5"
    config.input_device_name = "UltraLite-mk5"
    graph = AudioGraph(config)
    audio_in = AudioIn(1)
    audio_in = audio_in[0]
    graph.play(audio_in) 

    # BUFFER FOR RECORDING SOUND
    update_interval_s = hop_size / sr
    input_buffer = Buffer(1, FFT_window)
    min_feedback_writer_delay = graph.output_buffer_size / graph.sample_rate # samples
    graph.add_node(HistoryBufferWriter(input_buffer, audio_in))

    tot_src_signal = []
    latest_interval = 0
    timestamps = 0
    try:
        # save files
        timestamp = int(time.time())
        save_dir = f'./prova/{timestamp}'
        os.makedirs(save_dir, exist_ok=True)
        graph.start_recording(f'{save_dir}/performance.wav', num_channels=2)
        start_time = time.time()
        count_frames = 0

        while True:
            start_time_this = time.time()
            signal = input_buffer.data[0,:].copy()
            count_frames += hop_size
            tot_src_signal.append(signal[-hop_size:])
            end_time_this = time.time()
            latest_interval = end_time_this - start_time_this

            timestamps += update_interval_s
            time.sleep( update_interval_s - update_interval_s*0.19)

    except KeyboardInterrupt:
        graph.stop_recording()
        end_time = time.time()
        print(f'total performance time: {end_time-start_time}')
        print(f'total performance time2: {timestamps}')
        sf.write(f'{save_dir}/performance_src_all.wav', np.concatenate(tot_src_signal), sr)
