import os
import time
import json
import joblib
import random
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
        src_pad_mask = model.create_pad_mask(X, pad_token=PAD_TOKEN).to(device).to(torch.bool)
        tgt_pad_mask = model.create_pad_mask(y, pad_token=PAD_TOKEN).to(device).to(torch.bool)
        pred = model(X, y, src_pad_mask=src_pad_mask[:,:,0], tgt_pad_mask=tgt_pad_mask[:,:,0])
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
            src_pad_mask = model.create_pad_mask(X, pad_token=PAD_TOKEN).to(device).to(torch.bool)
            tgt_pad_mask = model.create_pad_mask(y, pad_token=PAD_TOKEN).to(device).to(torch.bool)
            pred = model(X, y, src_pad_mask=src_pad_mask[:,:,0], tgt_pad_mask=tgt_pad_mask[:,:,0])
            loss = loss_fn(pred, y_expected)
            total_loss += loss.detach().item()
    return total_loss / len(val_dataloader)

def evaluateR2Loop(model, val_dataloader, device, PAD_TOKEN, tgt_scaler):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch in val_dataloader:
            X = torch.tensor(batch[0]).type(torch.FloatTensor).to(device)
            y = torch.tensor(batch[1]).type(torch.FloatTensor).to(device)
            y_expected = torch.tensor(batch[2]).type(torch.FloatTensor).to(device).reshape(1, batch[2].shape[0], batch[2].shape[1])
            src_pad_mask = model.create_pad_mask(X, pad_token=PAD_TOKEN).to(device).to(torch.bool)
            tgt_pad_mask = model.create_pad_mask(y, pad_token=PAD_TOKEN).to(device).to(torch.bool)
            pred = model(X, y, src_pad_mask=src_pad_mask[:,:,0], tgt_pad_mask=tgt_pad_mask[:,:,0])
            y_expected_denorm = tgt_scaler.inverse_transform(y_expected.detach().cpu().numpy().reshape(pred.shape[1], pred.shape[2]))
            pred_denorm = tgt_scaler.inverse_transform(pred.detach().cpu().numpy().reshape(pred.shape[1], pred.shape[2]))
            r2 = r2_score(y_expected_denorm, pred_denorm)
            total_loss += r2
    return total_loss / len(val_dataloader)


if __name__ == '__main__': 

    # find device
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    corpus_name = 'guitar-duo'
    corpus_path = f'corpus/{corpus_name}'
    src_track_name = 'guitar1'
    tgt_track_name = 'guitar2'
    SYNTH_NAME = 'FM'

    with open(f"synth/params/{SYNTH_NAME}.txt", "r") as file:
        params = file.readlines()
    N_latents_tgt = len(params) # num of target latents is the same as the number of parameters

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
    training_parameters['dim_feedforward'] = 512
    training_parameters['dropout_p'] = 0.3
    training_parameters['window_size_s'] = 12
    training_parameters['pred_time_s'] = 0
    training_parameters['pad_token'] = -1000
    training_parameters['loss_fn'] = 'MSE'
    training_parameters['learning_rate'] = 0.001
    training_parameters['epochs'] = 4000
    training_parameters['seed'] = 666
    training_parameters['use_timbre'] = True
    training_parameters['N_latents_src'] = 8
    training_parameters['N_latents_tgt'] = N_latents_tgt
    training_parameters['synth_name'] = SYNTH_NAME
    model_load_path = None

    random.seed(training_parameters['seed'])
    torch.manual_seed(training_parameters['seed'])
    np.random.seed(training_parameters['seed'])

    if training_parameters['use_timbre']:
        features_source = [f'latent-{j}' for j in range(training_parameters['N_latents_src'])]
        features_source += ['rms', 'pitch', 'event_duration']
        features_target = [f'latent-{j}' for j in range(training_parameters['N_latents_tgt'])]
        features_target += ['rms', 'pitch', 'event_duration']
    else:
        features_source = ['rms', 'pitch', 'event_duration']
        features_target = ['rms', 'pitch', 'event_duration']

    training_parameters['features_source'] = features_source
    training_parameters['features_target'] = features_target


    # MAKE SAVE DIRECTORY
    today = date.today()
    model_name = f'{int(time.time())}-transformerDiscreteT2V'
    logdir = f'transformer/trained/{corpus_name}_{SYNTH_NAME}/{model_name}'
    os.makedirs(logdir, exist_ok=True)
    with open(f'{logdir}/training_config.json', 'w', encoding='utf-8') as f:
        json.dump(training_parameters, f, ensure_ascii=False, indent=4)

    # LOAD CORPUS
    source_corpus_df = pd.read_csv(f"{corpus_path}/src_events.csv", index_col=0)
    target_corpus_df = pd.read_csv(f"{corpus_path}/tgt_events_{SYNTH_NAME}.csv", index_col=0)
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
            # add time difference index
            indices = np.where((X_times >= window_start_idx) & (X_times < y_times[j+1]))
            x_src_window_times = (y_times[j+1] - PRED_TIME_SAMPLES) - X_times[indices]
            x_src = np.concatenate((X_values[indices], x_src_window_times.reshape(-1,1)), axis=1)
            all_src_times += x_src_window_times.tolist()
            # add time difference index
            indices = np.where((y_times >= window_start_idx) & (y_times < y_times[j+1]))
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
    latentsTranspose = False
    maskAugment = False
    randomSubstitute = True
    NUM_AUGMENT = 10
    rmsIndexSrc = features_source.index("rms")
    rmsIndexTgt = features_target.index("rms")
    pitchIndexSrc = features_source.index("pitch")
    pitchIndexTgt = features_target.index("pitch")
    durationIndexSrc = features_source.index("event_duration")
    durationIndexTgt = features_target.index("event_duration")
    latentIndexSrc = [features_source.index(f"latent-{j}") for j in range(training_parameters['N_latents_src'])]
    latentIndexTgt = [features_source.index(f"latent-{j}") for j in range(training_parameters['N_latents_tgt'])]
    new_data = []
    for datum in data:
        if datum[0].shape[0] > 0 and datum[1].shape[0] > 0:
            x_src = datum[0].copy()
            x_tgt = datum[1].copy()
            y_tgt = datum[2].copy()
            new_data.append([x_src, x_tgt, y_tgt])
            for _ in range(NUM_AUGMENT):
                if rmsTranspose:
                    # increase or decrease rms by max 10 %
                    RMS_TRANSP = 0.2
                    coefficient = RMS_TRANSP * (np.random.rand() * 2 -1)
                    x_src[:,rmsIndexSrc] = x_src[:,rmsIndexSrc] + x_src[:,rmsIndexSrc] * coefficient
                    x_tgt[:,rmsIndexTgt] = x_tgt[:,rmsIndexTgt] + x_tgt[:,rmsIndexTgt] * coefficient
                    y_tgt[rmsIndexTgt] = y_tgt[rmsIndexTgt] + y_tgt[rmsIndexTgt] * coefficient
                    # new_data.append([x_src, x_tgt, y_tgt])
                if pitchTranspose:
                    # transpose pitch by max 10 %
                    PITCH_TRANSP = 0.3
                    coefficient = PITCH_TRANSP * (np.random.rand() * 2 -1)
                    x_src[:,pitchIndexSrc] = x_src[:,pitchIndexSrc] + x_src[:,pitchIndexSrc] * coefficient
                    x_tgt[:,pitchIndexTgt] = x_tgt[:,pitchIndexTgt] + x_tgt[:,pitchIndexTgt] * coefficient
                    y_tgt[pitchIndexTgt] = y_tgt[pitchIndexTgt] + y_tgt[pitchIndexTgt] * coefficient
                    # new_data.append([x_src, x_tgt, datum[2]])
                if durationTranspose:
                    # increase or decrease durations by max 10 %
                    DUR_TRANSP = 0.3
                    coefficient = DUR_TRANSP * (np.random.rand() * 2 -1)
                    x_src[:,durationIndexSrc] = x_src[:,durationIndexSrc] + x_src[:,durationIndexSrc] * coefficient
                    x_tgt[:,durationIndexTgt] = x_tgt[:,durationIndexTgt] + x_tgt[:,durationIndexTgt] * coefficient
                    y_tgt[durationIndexTgt] = y_tgt[durationIndexTgt] + y_tgt[durationIndexTgt] * coefficient
                if latentsTranspose:
                    # increase or decrease durations by max 10 %
                    LATENTS_TRANSP = 0.2
                    coefficient = LATENTS_TRANSP * (np.random.rand(len(latentIndexSrc)) * 2 -1)
                    x_src[:,latentIndexSrc] = x_src[:,latentIndexSrc] + x_src[:,latentIndexSrc] * coefficient
                    coefficient = LATENTS_TRANSP * (np.random.rand(len(latentIndexTgt)) * 2 -1)
                    x_tgt[:,latentIndexTgt] = x_tgt[:,latentIndexTgt] + x_tgt[:,latentIndexTgt] * coefficient
                    y_tgt[latentIndexTgt] = y_tgt[latentIndexTgt] + y_tgt[latentIndexTgt] * coefficient
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
    non_augmented_data = data.copy()
    data = new_data
    # save non-augmented data for later testing

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

    # ----------------------------------------#
    # PERFORM EVALUATION ON NON-AUGMENTED DATA
    # ----------------------------------------#

    # load best model weights
    evaluation_metrics = {}
    model_load_path = f'{logdir}/best_model.pth'
    model.load_state_dict(torch.load(model_load_path, weights_only=True))
    # CUT SEQUENCE LENGTH
    MAX_SEQ_LEN = training_log['max_sequence_length']
    new_data = []
    for datum in non_augmented_data:
        if datum[0].shape[0] > 0 and datum[1].shape[0] > 0:
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
    batch_size = 8
    train_dataloader = batchify(train_data, batch_size)
    val_dataloader = batchify(val_data, batch_size)
    test_dataloader = batchify(test_data, batch_size)

    # compute R2 score
    train_loss = validationLoop(model, train_dataloader, loss_fn, opt, device, training_parameters['pad_token'])
    train_r2 = evaluateR2Loop(model, train_dataloader, device, training_parameters['pad_token'], tgt_scaler)
    evaluation_metrics['training_MSE'] = train_loss
    evaluation_metrics['training_R2'] = train_r2
    validation_loss = validationLoop(model, val_dataloader, loss_fn, opt, device, training_parameters['pad_token'])
    validation_r2 = evaluateR2Loop(model, val_dataloader, device, training_parameters['pad_token'], tgt_scaler)
    evaluation_metrics['validation_MSE'] = validation_loss
    evaluation_metrics['validation_R2'] = validation_r2
    test_loss = validationLoop(model, test_dataloader, loss_fn, opt, device, training_parameters['pad_token'])
    test_r2 = evaluateR2Loop(model, test_dataloader, device, training_parameters['pad_token'], tgt_scaler)
    evaluation_metrics['test_MSE'] = test_loss
    evaluation_metrics['test_R2'] = test_r2
    with open(f'{logdir}/evaluation.json', 'w', encoding='utf-8') as f:
        json.dump(evaluation_metrics, f, ensure_ascii=False, indent=4)

    # ----------------------------------------#
    # GENERATE CO-IMPROVISATIONS OFFLINE
    # ----------------------------------------#
    
    from VAE_train import Encoder, Decoder, Model
    from seq2seq_generate_coimpro import generateCoImprovisation
    from seq2seq_resynthesize import plotFlowTimbre

    training_parameters['max_sequence_length'] = MAX_SEQ_LEN 

    device = torch.device("cpu") 

    # LOAD VAE FOR SRC ENCODING
    corpus_path = f'corpus/{corpus_name}'
    VAE_model_name = f'{corpus_path}/trained_VAE/VAE.pth'
    hidden_dim = 128 # load these from pre-saved JSON
    latent_dim = 8
    image_size = 64
    image_size_flat = image_size * image_size
    encoder = Encoder(input_dim=image_size_flat, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = image_size_flat)
    VAE_model = Model(Encoder=encoder, Decoder=decoder).to(device)
    VAE_model.load_state_dict(torch.load(VAE_model_name, weights_only=True))
    VAE_model.eval()

    # LOAD FLOW MODEL FOR TGT COMPUTATION
    model_path = f'./synth/trained_flows/{SYNTH_NAME}/models'
    flow_model_name = 'vae_flow_mel_mse_0_gated_cnn_planar_flow_post_maf_3'
    flow_model = torch.load(f'{model_path}/{flow_model_name}.model', weights_only=False, map_location=device).to(device)
    flow_model.eval()

    N_GEN_COIMPRO = 4
    for filename in common_filenames[:N_GEN_COIMPRO]:
        src_audio_filepath = f"{training_parameters['sound_corpus_path']}/{filename}/{training_parameters['target_track_name']}.wav"
        df_tgt = generateCoImprovisation(src_audio_filepath, SYNTH_NAME, VAE_model, flow_model, model.to(device), 
                                        training_parameters, src_scaler, tgt_scaler, src_time_scaler, tgt_time_scaler,
                                        training_parameters['use_timbre'], device, logdir)
        # print maps over latent space PCA

        X = np.load(f'./synth/trained_flows/{SYNTH_NAME}/visualize/{flow_model_name}/latents.npy')
        columns = [f'latent-{j}' for j in range(N_latents_tgt)]
        flow_latents = df_tgt[columns].values
        plotFlowTimbre(X, flow_latents, f'{logdir}/00_results/{filename}')
