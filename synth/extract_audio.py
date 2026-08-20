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
    SYNTH_NAME = 'Granular'
    SAVE_FILEPATH = f'{SAVE_FILEPATH}/{SYNTH_NAME}'
    os.makedirs(SAVE_FILEPATH, exist_ok=True)

    RANDOM_PERMUTATIONS = True
    N_timbre_parameters = 7
    if RANDOM_PERMUTATIONS:
        NUM_SAMPLES = 5000
        permutations = []
        for _ in range(NUM_SAMPLES):
            pp = list(np.random.rand(N_timbre_parameters))
            permutations.append('-'.join([str(np.clip(p, 0.001, 0.999))[:5] for p in pp]))
    else:
        # COMPUTE ALL PARAMETER COMBINATIONS
        param_step_for_corpus_gen = 4
        elements = np.array(range(0, param_step_for_corpus_gen)) / param_step_for_corpus_gen * 0.85 + 0.1
        permutations = [p for p in itertools.product(elements, repeat=N_timbre_parameters)] 
        ppp = []
        for pp in permutations:
            pp = list(pp)
            ppp.append('-'.join([str(p)[:5] for p in pp]))
        permutations = ppp

    # COMPUTE COMMANDS TO EXTRACT SOUND
    GAIN = 2
    FREQUENCY = 220
    DURATION = RECORDING_TIME / 2
    ADSR = '0.5-0.5-0.5-0.5'
    commands = []
    for params in permutations:
        command = 'python3 synths.py '
        command += f'--SYNTH_NAME {SYNTH_NAME} '
        command += f'--SAVE_FILEPATH {SAVE_FILEPATH} '
        command += f'--GAIN {GAIN} '
        command += f'--FREQUENCY {FREQUENCY} '
        command += f'--DURATION {DURATION} '
        command += f'--ADSR {ADSR} '
        command += f'--TIMBRE_PARAMETERS {params} '
        command += f'--RECORDING_TIME {RECORDING_TIME} '
        command += f'--SAMPLE_RATE {sr} '
        command += f'--HOP_SIZE {hop_size}'
        commands.append(command)

    # EXECUTE WITH MULTI-THREADING
    MAX_WORKERS = 5
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(task, commands))
