import os
import time
import json
import joblib
import numpy as np
import pandas as pd

from datetime import date
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from CONSTANTS import * 
from transformer.transformerDiscreteT2V import TransformerDiscreteT2V


def batchify(data, batch_size=16):
    # return batches of dim (batch_size, seq_length, num_feats)
    batches = []
    for i in range(0, len(data)-batch_size, batch_size):
        this_batch_X = []
        this_batch_y = []
        this_batch_y_tgt = []
        for k in range(batch_size):
            this_batch_X.append(data[i+k][0])
            this_batch_y.append(data[i+k][1])
            this_batch_y_tgt.append(data[i+k][2])
        batches.append([np.array(this_batch_X),np.array(this_batch_y),np.array(this_batch_y_tgt)])
    print(f"{len(batches)} batches of size {batch_size}")
    return batches


# train and validation loops
def trainLoop(model, train_dataloader, loss_fn, opt, device, PAD_TOKEN):
    model.train()
    total_loss = 0
    for batch in train_dataloader:
        X = torch.tensor(batch[0]).type(torch.FloatTensor).to(device)
        y = torch.tensor(batch[1]).type(torch.FloatTensor).to(device)
        y_expected = torch.tensor(batch[2]).type(torch.FloatTensor).to(device).reshape(1, batch[2].shape[0],batch[2].shape[1])
        # y_expected = torch.tensor(batch[2]).type(torch.FloatTensor).to(device).reshape(1, batch[2].shape[0], batch[2].shape[1])
        # generate pad mask before t2v (t2v is non-linear and padding has to be applied before)
        src_pad_mask = model.create_pad_mask(X, pad_token=PAD_TOKEN).to(device).to(torch.bool)
        tgt_pad_mask = model.create_pad_mask(y, pad_token=PAD_TOKEN).to(device).to(torch.bool)
        # tgt_mask = model.get_tgt_mask(y.size(1)).to(device)
        # src_mask = model.transformer.generate_square_subsequent_mask(X.size(1), device=device).to(torch.bool)
        # tgt_mask = model.transformer.generate_square_subsequent_mask(y.size(1), device=device).to(torch.bool)
        pred = model(X, y, 
                    #  src_mask=src_mask, tgt_mask=tgt_mask, 
                     src_pad_mask=src_pad_mask[:,:,0], tgt_pad_mask=tgt_pad_mask[:,:,0])
        loss = loss_fn(pred, y_expected)
        opt.zero_grad()
        loss.backward()
        opt.step()
        total_loss += loss.detach().item()
    return total_loss / len(train_dataloader)

def validationLoop(model, val_dataloader, loss_fn, opt, device, PAD_TOKEN):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            X = torch.tensor(batch[0]).type(torch.FloatTensor).to(device)
            y = torch.tensor(batch[1]).type(torch.FloatTensor).to(device)
            y_expected = torch.tensor(batch[2]).type(torch.FloatTensor).to(device).reshape(1, batch[2].shape[0], batch[2].shape[1])
            # y_expected = torch.tensor(batch[2]).type(torch.FloatTensor).to(device).reshape(1, batch[2].shape[0], batch[2].shape[1])
            # generate pad mask before t2v (t2v is non-linear and padding has to be applied before)
            src_pad_mask = model.create_pad_mask(X, pad_token=PAD_TOKEN).to(device).to(torch.bool)
            tgt_pad_mask = model.create_pad_mask(y, pad_token=PAD_TOKEN).to(device).to(torch.bool)
            # tgt_mask = model.get_tgt_mask(y.size(1)).to(device)
            # src_mask = model.transformer.generate_square_subsequent_mask(X.size(1), device=device).to(torch.bool)
            # tgt_mask = model.transformer.generate_square_subsequent_mask(y.size(1), device=device).to(torch.bool)

            pred = model(X, y, 
                        #  src_mask=src_mask, tgt_mask=tgt_mask, 
                         src_pad_mask=src_pad_mask[:,:,0], tgt_pad_mask=tgt_pad_mask[:,:,0])
            loss = loss_fn(pred, y_expected)
            total_loss += loss.detach().item()
    return total_loss / len(val_dataloader)


if __name__ == '__main__': 

    # device 
    DEVICE = torch.device("cpu")
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")

    corpus_name = 'guitar-duo'
    corpus_path = f'corpus/{corpus_name}'
    src_track_name = 'guitar1'
    tgt_track_name = 'guitar2'
    latent_dim = 8
    N_latents_tgt = 2
    SYNTH_NAME = 'TimbreTheremin'

    # TRAIN SEQ2SEQ ON LATENTS
    training_parameters = {}
    training_parameters['sound_corpus_path'] = corpus_path
    training_parameters['source_track_name'] = src_track_name
    training_parameters['target_track_name'] = tgt_track_name
    training_parameters['batch_size'] = 64
    training_parameters['num_frequency'] = 8
    training_parameters['dataset_training_percentage'] = 0.8
    training_parameters['dataset_validation_percentage'] = 0.1
    training_parameters['dim_model'] = 64
    training_parameters['num_heads'] = 8
    training_parameters['num_encoder_layers'] = 8
    training_parameters['num_decoder_layers'] = 8
    training_parameters['dim_feedforward'] = 1024
    training_parameters['dropout_p'] = 0.3
    training_parameters['window_size_s'] = 12
    training_parameters['pred_time_s'] = 0
    training_parameters['pad_token'] = -1000
    training_parameters['loss_fn'] = 'MSE'
    training_parameters['learning_rate'] = 0.001
    training_parameters['epochs'] = 1000
    training_parameters['seed'] = 666
    model_load_path = None

    features_source = [f'latent-{j}' for j in range(latent_dim)]
    features_source.append('rms')
    features_source.append('pitch')
    features_source.append('event_duration')
    features_target = [f'latent-{j}' for j in range(N_latents_tgt)]
    features_target.append('rms')
    features_target.append('pitch')
    features_target.append('event_duration')

    features_source = ['rms', 'pitch', 'event_duration']
    features_target = ['rms', 'pitch', 'event_duration']


    # MAKE SAVE DIRECTORY
    today = date.today()
    model_name = f'{int(time.time())}-transformerDiscreteT2V'
    logdir = f'transformer/trained/{corpus_name}_{SYNTH_NAME}/{model_name}'
    os.makedirs(logdir, exist_ok=True)
    with open(f'{logdir}/training_config.json', 'w', encoding='utf-8') as f:
        json.dump(training_parameters, f, ensure_ascii=False, indent=4)

    # LOAD CORPUS
    source_corpus_df = pd.read_csv(f"{corpus_path}/src_events.csv", index_col=0)
    target_corpus_df = pd.read_csv(f"{corpus_path}/tgt_events.csv", index_col=0)
    src_signals_dfs = []
    tgt_signals_dfs = []
    unique_src_filenames = source_corpus_df['filename'].unique()
    unique_tgt_filenames = target_corpus_df['filename'].unique()
    unique_src_filenames = [name.split('/')[-3] for name in unique_src_filenames]
    unique_tgt_filenames = [name.split('/')[-3] for name in unique_tgt_filenames]
    common_filenames = list(set(unique_src_filenames) & set(unique_tgt_filenames))
    # common_filenames = common_filenames[:4] # select a subset of data
    for filename in common_filenames:
        # assume that audio tracks are located one folder before datasets
        partial_src_df = source_corpus_df[source_corpus_df['filename'] == f"{training_parameters['sound_corpus_path']}/{filename}/mel/{training_parameters['source_track_name']}.npy"]
        partial_tgt_df = target_corpus_df[target_corpus_df['filename'] == f"{training_parameters['sound_corpus_path']}/{filename}/mel/{training_parameters['target_track_name']}.npy"]
        # check that both are non-empty
        if partial_src_df.shape[0] > 0 and partial_tgt_df.shape[0] > 0:
            src_signals_dfs.append(partial_src_df)
            tgt_signals_dfs.append(partial_tgt_df)

    # print(source_corpus_df.head())

    # normalize source features
    src_features = source_corpus_df[features_source].values
    src_scaler = StandardScaler()
    src_scaler.fit(src_features)
    src_normalized_sequences = []
    for seq_df in src_signals_dfs:
        sequence_df = pd.DataFrame(data=src_scaler.transform(seq_df[features_source].values), columns=features_source)
        sequence_df['event_start'] = seq_df['event_start'].values
        src_normalized_sequences.append(sequence_df)
    joblib.dump(src_scaler, f'{logdir}/source_scaler.pkl')

    print(src_normalized_sequences[0].shape)

    # normalize target features
    tgt_features = target_corpus_df[features_target].values
    tgt_scaler = StandardScaler()
    tgt_scaler.fit(tgt_features)
    tgt_normalized_sequences = []
    for seq_df in tgt_signals_dfs:
        sequence_df = pd.DataFrame(data=tgt_scaler.transform(seq_df[features_target].values), columns=features_target)
        sequence_df['event_start'] = seq_df['event_start'].values
        tgt_normalized_sequences.append(sequence_df)
    joblib.dump(tgt_scaler, f'{logdir}/target_scaler.pkl')

    # make dataloader
    window_size_s = training_parameters['window_size_s']
    pred_time_s = training_parameters['pred_time_s']
    WINDOW_N_SAMPLES = window_size_s * sr
    PRED_TIME_SAMPLES = pred_time_s * sr
    data = []
    # max_seq_len = 0
    seq_lens = []
    all_src_times = []
    all_tgt_times = []
    for k in range(len(src_normalized_sequences)):
        X_values = src_normalized_sequences[k][features_source].values
        y_values = tgt_normalized_sequences[k][features_target].values
        X_times = src_normalized_sequences[k]['event_start'].values
        y_times = tgt_normalized_sequences[k]['event_start'].values
        for j in range(y_times.shape[0]-1):
            y_tgt = y_values[j+1,:]
            # we subtract inference time 
            # so that we can call inference some time before the end of an onset
            window_start_idx = (y_times[j+1] - PRED_TIME_SAMPLES) - WINDOW_N_SAMPLES
            indices = np.where((X_times >= window_start_idx) & (X_times < y_times[j+1]))
            # add time difference index
            x_src_window_times = (y_times[j+1] - PRED_TIME_SAMPLES) - X_times[indices]
            x_src = np.concatenate((X_values[indices], x_src_window_times.reshape(-1,1)), axis=1)
            all_src_times += x_src_window_times.tolist()
            indices = np.where((y_times >= window_start_idx) & (y_times < y_times[j+1]))
            # add time difference index
            x_tgt_window_times = (y_times[j+1] - PRED_TIME_SAMPLES) - y_times[indices]
            x_tgt = np.concatenate((y_values[indices], x_tgt_window_times.reshape(-1,1)), axis=1)
            all_tgt_times += x_tgt_window_times.tolist()
            data.append([x_src, x_tgt, y_tgt])
            seq_lens.append(x_src.shape[0])
            seq_lens.append(x_tgt.shape[0])

    # DATA AUGMENTATION
    rmsTranspose = True
    pitchTranspose = True
    durationTranspose = True
    maskAugment = False
    randomSubstitute = True
    NUM_AUGMENT = 10
    rmsIndexSrc = features_source.index("rms")
    rmsIndexTgt = features_source.index("rms")
    pitchIndexSrc = features_source.index("pitch")
    pitchIndexTgt = features_source.index("pitch")
    durationIndexSrc = features_source.index("event_duration")
    durationIndexTgt = features_source.index("event_duration")
    new_data = []
    import random
    for datum in data:
        if datum[0].shape[0] > 0 and datum[1].shape[0] > 0:
            x_src = datum[0].copy()
            x_tgt = datum[1].copy()
            y_tgt = datum[2].copy()
            new_data.append([x_src, x_tgt, y_tgt])
            for _ in range(NUM_AUGMENT):
                if rmsTranspose:
                    # increase or decrease rms by max 10 %
                    RMS_TRANSP = 0.3
                    coefficient = RMS_TRANSP * (np.random.rand() * 2 -1)
                    x_src[:,rmsIndexSrc] = x_src[:,rmsIndexSrc] + x_src[:,rmsIndexSrc] * coefficient
                    x_tgt[:,rmsIndexTgt] = x_tgt[:,rmsIndexTgt] + x_tgt[:,rmsIndexTgt] * coefficient
                    y_tgt[rmsIndexTgt] = y_tgt[rmsIndexTgt] + y_tgt[rmsIndexTgt] * coefficient
                    # new_data.append([x_src, x_tgt, y_tgt])
                if pitchTranspose:
                    # transpose pitch by max 10 %
                    PITCH_TRANSP = 0.5
                    coefficient = PITCH_TRANSP * (np.random.rand() * 2 -1)
                    x_src[:,pitchIndexSrc] = x_src[:,pitchIndexSrc] + x_src[:,pitchIndexSrc] * coefficient
                    x_tgt[:,pitchIndexTgt] = x_tgt[:,pitchIndexTgt] + x_tgt[:,pitchIndexTgt] * coefficient
                    y_tgt[pitchIndexTgt] = y_tgt[pitchIndexTgt] + y_tgt[pitchIndexTgt] * coefficient
                    # new_data.append([x_src, x_tgt, datum[2]])
                if durationTranspose:
                    # increase or decrease durations by max 10 %
                    DUR_TRANSP = 0.2
                    coefficient = DUR_TRANSP * (np.random.rand() * 2 -1)
                    x_src[:,durationIndexSrc] = x_src[:,durationIndexSrc] + x_src[:,durationIndexSrc] * coefficient
                    x_tgt[:,durationIndexTgt] = x_tgt[:,durationIndexTgt] + x_tgt[:,durationIndexTgt] * coefficient
                    y_tgt[durationIndexTgt] = y_tgt[durationIndexTgt] + y_tgt[durationIndexTgt] * coefficient
                if maskAugment:
                    MASK_TOKEN = np.ones_like(x_src[0,:-1]) * -10
                    src_len, tgt_len = x_src.shape[0], x_tgt.shape[0]
                    if np.random.rand() > 0.8:
                        x_src[random.randint(0, src_len-1), :-1] = MASK_TOKEN
                        x_tgt[random.randint(0, tgt_len-1), :-1] = MASK_TOKEN
                if randomSubstitute:
                    src_len, tgt_len = x_src.shape[0]-1, x_tgt.shape[0]-1
                    if np.random.rand() > 0.8:
                        x_src[random.randint(0, src_len), :-1] = x_src[random.randint(0, src_len), :-1]
                        x_tgt[random.randint(0, tgt_len), :-1] = x_tgt[random.randint(0, tgt_len), :-1]
                new_data.append([x_src, x_tgt, y_tgt])
    data = new_data

    # CUT SEQUENCE LENGTH
    MAX_SEQ_LEN = int(np.array(seq_lens).mean() + np.array(seq_lens).std())
    if model_load_path:
        MAX_SEQ_LEN = training_parameters['max_sequence_length']
    # remove rows where time is less than min or 
    # normalize time feature
    src_time_scaler = StandardScaler()
    src_time_scaler.fit(np.array(all_src_times).reshape(-1,1))
    tgt_time_scaler = StandardScaler()
    tgt_time_scaler.fit(np.array(all_tgt_times).reshape(-1,1))
    joblib.dump(src_time_scaler, f'{logdir}/source_time_scaler.pkl')
    joblib.dump(tgt_time_scaler, f'{logdir}/target_time_scaler.pkl')
    new_data = []
    for datum in data:
        if datum[0].shape[0] > 0 and datum[1].shape[0] > 0:
            # print(datum[0].shape)
            x_src = datum[0]
            x_src[:,-1] = src_time_scaler.transform(x_src[:,-1].reshape(-1,1)).reshape(-1)
            x_tgt = datum[1]
            x_tgt[:,-1] = src_time_scaler.transform(x_tgt[:,-1].reshape(-1,1)).reshape(-1)
            x_src = x_src[-MAX_SEQ_LEN:,:]
            x_tgt = x_tgt[-MAX_SEQ_LEN:,:]
            new_data.append([x_src, x_tgt, datum[2]])
    data = new_data


    # PAD SEQUENCES
    PAD_TOKEN = training_parameters['pad_token']
    new_data = []
    for datum in data:
        x_src = np.ones((MAX_SEQ_LEN, len(features_source)+1)) * PAD_TOKEN
        x_tgt = np.ones((MAX_SEQ_LEN, len(features_target)+1)) * PAD_TOKEN
        x_src[:datum[0].shape[0],:] = datum[0]
        x_tgt[:datum[1].shape[0],:] = datum[1]
        new_data.append([x_src, x_tgt, datum[2]])
    data = new_data

    with open(f'{logdir}/training_config.json', 'r') as f:
        training_log = json.load(f)
    training_log['max_sequence_length'] = MAX_SEQ_LEN
    with open(f'{logdir}/training_config.json', 'w', encoding='utf-8') as f:
        json.dump(training_log, f, ensure_ascii=False, indent=4)

    # shuffle sequences
    np.random.shuffle(data)

    # train-validation split
    train_perc = training_parameters['dataset_training_percentage']
    val_perc = training_parameters['dataset_validation_percentage']
    train_data = data[:int(len(data)*train_perc)]
    test_data = data[int(len(data)*train_perc):]
    val_data = train_data[int(len(train_data)*(1-val_perc)):]
    train_data = train_data[:int(len(train_data)*(1-val_perc))]

    # make training batches
    batch_size = training_parameters['batch_size']
    train_dataloader = batchify(train_data, batch_size)
    val_dataloader = batchify(val_data, batch_size)
    test_dataloader = batchify(test_data, batch_size)

    # find device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    # create model
    model = TransformerDiscreteT2V(
            num_featuresin=len(features_source)+1,
            num_featuresout=len(features_target)+1,
            num_frequency=training_parameters['num_frequency'],
            max_seq_len=MAX_SEQ_LEN,
            dim_model=training_parameters['dim_model'], 
            num_heads=training_parameters['num_heads'], 
            num_encoder_layers=training_parameters['num_encoder_layers'], 
            num_decoder_layers=training_parameters['num_decoder_layers'], 
            dim_feedforward=training_parameters['dim_feedforward'], 
            dropout_p=training_parameters['dropout_p']
    ).to(device)
    if model_load_path:
        model.load_state_dict(torch.load(model_load_path, weights_only=True, map_location=device))
    opt = torch.optim.SGD(model.parameters(), lr=training_parameters['learning_rate'])
    # loss_fn = nn.CrossEntropyLoss()
    if training_parameters['loss_fn'] == 'L1':
        loss_fn = nn.L1Loss()
    else:
        loss_fn = nn.MSELoss()


    # Used for plotting later on
    train_loss_list, validation_loss_list = [], []

    writer = SummaryWriter(logdir)
    epochs = training_parameters['epochs']
    best_loss = float('inf')
    print("Training and validating model")
    for epoch in range(epochs):
        print("-"*25, f"Epoch {epoch + 1}","-"*25)

        train_loss = trainLoop(model, train_dataloader, loss_fn, opt, device, training_parameters['pad_token'])
        train_loss_list += [train_loss]
        writer.add_scalar("Loss/train", train_loss, epoch)

        validation_loss = validationLoop(model, val_dataloader, loss_fn, opt, device, training_parameters['pad_token'])
        validation_loss_list += [validation_loss]
        writer.add_scalar("Loss/validation", validation_loss, epoch)

        print(f"Training loss: {train_loss:.4f}")
        print(f"Validation loss: {validation_loss:.4f}")
        print()

        if validation_loss < best_loss:
            best_loss = validation_loss
            torch.save(model.state_dict(), f'{logdir}/best_model.pth')

    writer.flush()
    writer.close()


    evaluation_metrics = {}

    # load best model weights
    model_load_path = f'{logdir}/best_model.pth'
    model.load_state_dict(torch.load(model_load_path, weights_only=True))
    train_loss = validationLoop(model, train_dataloader, loss_fn, opt, device, training_parameters['pad_token'])
    evaluation_metrics['training_MSE'] = train_loss
    validation_loss = validationLoop(model, val_dataloader, loss_fn, opt, device, training_parameters['pad_token'])
    evaluation_metrics['validation_MSE'] = validation_loss
    test_loss = validationLoop(model, test_dataloader, loss_fn, opt, device, training_parameters['pad_token'])
    evaluation_metrics['test_MSE'] = test_loss

    with open(f'{logdir}/evaluation.json', 'w', encoding='utf-8') as f:
        json.dump(evaluation_metrics, f, ensure_ascii=False, indent=4)

    # GENERATE OUTPUTS

    # GET SOURCE AUDIOFILE
    # EXTRACT EVENTS
    # ENCODE SOURCE LATENTS
    # PREPARE DATA SEQUENCE
    # LOAD SEQ2SEQ TRAINED MODEL
    # GENERATE SOUND RESPONSES
    # SAVE AUDIOFILE
