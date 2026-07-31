import os
import subprocess
import itertools
import numpy as np
from concurrent.futures import ThreadPoolExecutor

def task(args):
    print(f"Starting task {args}")
    subprocess.call(args.split(' '))
    return 

if __name__ == '__main__': 

    # CONFIG
    SAVE_FILEPATH = './datasets'
    RECORDING_TIME = 10
    sr = 44100
    hop_size = 256
    SYNTH_NAME = 'Snare'
    SAVE_FILEPATH = f'{SAVE_FILEPATH}/{SYNTH_NAME}'
    os.makedirs(SAVE_FILEPATH, exist_ok=True)

    RANDOM_PERMUTATIONS = True
    N_synth_parameters = 8
    if RANDOM_PERMUTATIONS:
        NUM_SAMPLES = 10000
        permutations = []
        for _ in range(NUM_SAMPLES):
            pp = list(np.random.rand(N_synth_parameters))
            permutations.append('-'.join([str(p)[:5] for p in pp]))
    else:
        # COMPUTE ALL PARAMETER COMBINATIONS
        param_step_for_corpus_gen = 4
        elements = np.array(range(0, param_step_for_corpus_gen)) / param_step_for_corpus_gen * 0.85 + 0.1
        permutations = [p for p in itertools.product(elements, repeat=N_synth_parameters)] 
        ppp = []
        for pp in permutations:
            pp = list(pp)
            ppp.append('-'.join([str(p)[:5] for p in pp]))
        permutations = ppp

    # COMPUTE COMMANDS TO EXTRACT SOUND
    commands = []
    for params in permutations:
        command = f'python3 synths.py --SYNTH_NAME {SYNTH_NAME} --SAVE_FILEPATH {SAVE_FILEPATH} --SYNTH_PARAMETERS {params} --RECORDING_TIME {RECORDING_TIME} --SAMPLE_RATE {sr} --HOP_SIZE {hop_size}'
        commands.append(command)

    # EXECUTE WITH MULTI-THREADING
    MAX_WORKERS = 5
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(task, commands))
