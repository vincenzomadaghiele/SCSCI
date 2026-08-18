# SCSCI: synthesizer control system for co-improvisation

1. freq param (if available) is not included in timbre [input in Hz]
2. duration param is not included in timbre [input in ms]
3. gain param is not included in timbre [input 0-1]
4. A, S, R envelope params are NOT in timbre and they are fixed [0-1]
5. all other params are TIMBRE, used to train TIMBRE SPACE [0-1]

transformer gets as inputs two sequences of:
[ RMS, duration, pitch (if available), latents-timbre ]

and outputs regression of:
[ RMS, duration, pitch (if available), latents-timbre ]

these are mapped:
- RMS --> gain
- pitch --> frequency
- duration --> gate on/off
- latents-timbre --> NF regressor --> timbre parameters

SYNTHS provided:
- timbre theremin (with pitch)
- snare (no pitch)
- risset bell (pitch)
- karplus-strong (with pitch)
- FM (with pitch)
- event-granular (with pitch)
- chaotic (no pitch)

TODO:
2. make all synths and train NF model for each
3. send demo synths to musicians (make them choose)
4. train demo models
5. workshops (process?)
6. fine-tune (more than one model and change with MIDI or automatic?)

To train flow:
```
python3 flow_train.py --path synth/datasets --dataset TimbreTheremin --data mel --model vae_flow --regressor flow_post --flow planar --epochs 200 --output synth/trained_flows/TimbreTheremin/ --start_regress 50
```