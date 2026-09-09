import os
import librosa
import numpy as np
import pandas as pd
import soundfile as sf

from CONSTANTS import *
from corpus.extract_events import plotEvents, plotPairedEvents


def slicePerformanceRecording(START_SLICE_INDEX_S, STOP_SLICE_INDEX_S, corpus_name, SYNTH_NAME, transformer_name, performance_name):


    # LOAD DATABASE
    model_dir = f'./transformer/trained/{corpus_name}_{SYNTH_NAME}/{transformer_name}'
    performance_dir = f'{model_dir}/01_live_recordings/{performance_name}'
    df_src = pd.read_csv(f"{performance_dir}/src_events.csv", index_col=0)
    df_tgt = pd.read_csv(f"{performance_dir}/tgt_events.csv", index_col=0)

    if START_SLICE_INDEX_S is None:
        START_SLICE_INDEX_S = 0

    if STOP_SLICE_INDEX_S is None:
        src_max = (df_src.iloc[-1]["event_start"] + df_src.iloc[-1]["event_duration"]) / sr
        tgt_max = (df_tgt.iloc[-1]["event_start"] + df_tgt.iloc[-1]["event_duration"]) / sr
        STOP_SLICE_INDEX_S = int(max(src_max, tgt_max))

    # MAKE DIR TO SAVE SLICED FILE
    sliced_dir = f'{performance_dir}/sliced/sliced-{START_SLICE_INDEX_S}_{STOP_SLICE_INDEX_S}'
    os.makedirs(sliced_dir, exist_ok=True)

    # SLICE PERFORMANCES
    df_src_sliced = df_src.loc[df_src['event_start'] > START_SLICE_INDEX_S * sr]
    df_src_sliced = df_src_sliced.loc[df_src_sliced['event_start'] < STOP_SLICE_INDEX_S * sr]
    df_src_sliced['event_start'] = df_src_sliced['event_start'].values - START_SLICE_INDEX_S * sr
    df_tgt_sliced = df_tgt.loc[df_tgt['event_start'] > START_SLICE_INDEX_S * sr]
    df_tgt_sliced = df_tgt_sliced.loc[df_tgt_sliced['event_start'] < STOP_SLICE_INDEX_S * sr]
    df_tgt_sliced['event_start'] = df_tgt_sliced['event_start'].values - START_SLICE_INDEX_S * sr

    # SAVE SLICED PERFORMANCES
    df_src_sliced.to_csv(f"{sliced_dir}/src_events.csv")
    df_tgt_sliced.to_csv(f"{sliced_dir}/tgt_events.csv")

    # SAVE AUDIO FILES
    signal_src, _ = librosa.load(f'{performance_dir}/performance_src.wav', mono=False)
    sf.write(f'{sliced_dir}/performance_src-{START_SLICE_INDEX_S}_{STOP_SLICE_INDEX_S}.wav', signal_src[START_SLICE_INDEX_S * sr : STOP_SLICE_INDEX_S * sr], sr)
    signal_tgt, _ = librosa.load(f'{performance_dir}/performance_tgt.wav', mono=False)
    sf.write(f'{sliced_dir}/performance_tgt-{START_SLICE_INDEX_S}_{STOP_SLICE_INDEX_S}.wav', signal_tgt[START_SLICE_INDEX_S * sr : STOP_SLICE_INDEX_S * sr], sr)
    stereo_data = np.column_stack((signal_src[START_SLICE_INDEX_S * sr : STOP_SLICE_INDEX_S * sr], signal_tgt[START_SLICE_INDEX_S * sr : STOP_SLICE_INDEX_S * sr]))
    sf.write(f'{sliced_dir}/performance-{START_SLICE_INDEX_S}_{STOP_SLICE_INDEX_S}.wav', stereo_data, sr)

    # SAVE PLOTS
    df_src_sliced['filename'] = [f'{sliced_dir}/performance_src-{START_SLICE_INDEX_S}_{STOP_SLICE_INDEX_S}.wav' for _ in range(df_src_sliced.values.shape[0])]
    plotEvents(df_src_sliced, sr, save_dir=sliced_dir, isDfin=True)
    df_tgt_sliced['filename'] = [f'{sliced_dir}/performance_tgt-{START_SLICE_INDEX_S}_{STOP_SLICE_INDEX_S}.wav' for _ in range(df_tgt_sliced.values.shape[0])]
    plotEvents(df_tgt_sliced, sr, save_dir=sliced_dir, isDfin=True)
    plotPairedEvents(df_src_sliced, df_tgt_sliced, sr, save_dir=sliced_dir, isDfin=True)


if __name__ == '__main__': 

    corpus_name = "petter-study"
    SYNTH_NAME = "FM"
    transformer_name = '1787764778-transformerDiscreteT2V-BEST'
    performance_name = '1787905202'
    START_SLICE_INDEX_S = 45
    STOP_SLICE_INDEX_S = None
    
    slicePerformanceRecording(START_SLICE_INDEX_S, 
                              STOP_SLICE_INDEX_S, 
                              corpus_name, 
                              SYNTH_NAME, 
                              transformer_name, 
                              performance_name)
