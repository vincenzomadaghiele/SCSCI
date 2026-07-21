import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange, tqdm
from collections import defaultdict
import shutil
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import torch
import torchaudio
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import torchvision.datasets as Datasets
import torchvision.transforms as transforms
import torchvision.utils as vutils

from models.RES_VAE_Dynamic import VAE
from models.vgg19 import VGG19
import models.Helpers as hf


class CustomImageDataset(Dataset):
    def __init__(self, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform

        self.corpus_paths = []
        for file in os.listdir(self.img_dir):
            if file.endswith(".npy"):
                self.corpus_paths.append(os.path.join(self.img_dir, file))

    def __len__(self):
        return len(self.corpus_paths)

    def __getitem__(self, idx):
        image = torch.from_numpy(np.load(self.corpus_paths[idx]))
        # tr = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
        # image = tr(image)
        image = torch.unsqueeze(image, 0).to(torch.float32)
        if self.transform:
            image = self.transform(image)
        return image


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(8, 4, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 8, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

if __name__ == "__main__":
    
    # device 
    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")

    # load data
    corpus_name = 'myvoice'
    data_path = f'corpora/{corpus_name}/info/mel'
    corpus_paths = []
    for file in os.listdir(data_path):
        if file.endswith(".npy"):
            corpus_paths.append(os.path.join(data_path, file))
    mels = []
    save_full_mels_path = f'{data_path}/full'
    os.makedirs(save_full_mels_path, exist_ok=True)
    i = 0
    for file in corpus_paths:
        corpus_file = np.load(file, allow_pickle=True)
        for event in corpus_file:
            if event['mel'].mean() != 0:
                mel = librosa.power_to_db(event['mel'], ref=np.max)
                # mel = event['mel']
                mels.append(mel)
                np.save(f'./{save_full_mels_path}/event-{i}.npy', np.array(mel))
                i += 1
    
    print(len(mels))
    compute_stats = []
    for mel in mels:
        compute_stats.append(mel.reshape(-1))
    compute_stats = np.concatenate(compute_stats, axis=0) # OR AXIS 1
    mean = compute_stats.mean()
    std = compute_stats.std()

    # fig, ax = plt.subplots()
    # S_dB = librosa.power_to_db(mels[210], ref=np.max)
    # img = librosa.display.specshow(S_dB, x_axis='time', y_axis='mel', sr=22050, fmax=8000, ax=ax)
    # fig.colorbar(img, ax=ax, format='%+2.0f dB')
    # ax.set(title='Mel-frequency spectrogram')
    # plt.show()

    def npy_loader(path):
        # sample = torch.from_numpy(np.load(path))
        sample = np.load(path)
        return sample

    # Create dataloaders
    image_size = 64
    print("-Target Image Size %d" % image_size)
    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size),
                                    # transforms.RandomHorizontalFlip(0.5),
                                    transforms.Normalize(mean, std)])


    data_set = CustomImageDataset(save_full_mels_path, transform=transform)

    # Randomly split the dataset with a fixed random seed for reproducibility
    test_split = 0.9
    n_train_examples = int(len(data_set) * test_split)
    n_test_examples = len(data_set) - n_train_examples
    train_set, test_set = torch.utils.data.random_split(data_set, [n_train_examples, n_test_examples],
                                                        generator=torch.Generator().manual_seed(42))

    batch_size = 8
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    # Get a test image batch from the test_loader to visualise the reconstruction quality etc
    dataiter = iter(test_loader)
    test_images = next(dataiter)
    print(test_images.shape)

    # # Create AE network.
    # ch_multi = 64
    # block_widths = (1, 2, 4, 8)
    # latent_channels = 32
    # num_res_blocks = 1
    # norm_type = "gn"
    # deep_model = True
    # vae_net = VAE(channel_in=test_images.shape[1],
    #             ch=ch_multi,
    #             blocks=block_widths,
    #             latent_channels=latent_channels,
    #             num_res_blocks=num_res_blocks,
    #             norm_type=norm_type,
    #             deep_model=deep_model).to(device)

    # # Setup optimizer
    # lr = 1e-4
    # optimizer = optim.Adam(vae_net.parameters(), lr=lr)

    # if norm_type == "bn":
    #     print("-Using BatchNorm")
    # elif norm_type == "gn":
    #     print("-Using GroupNorm")
    # else:
    #     ValueError("norm_type must be bn or gn")

    # # Create the feature loss module if required
    # feature_scale = 0
    # if feature_scale > 0:
    #     feature_extractor = VGG19().to(device)
    #     print("-VGG19 Feature Loss ON")
    # else:
    #     feature_extractor = None
    #     print("-VGG19 Feature Loss OFF")

    # # Let's see how many Parameters our Model has!
    # num_model_params = 0
    # for param in vae_net.parameters():
    #     num_model_params += param.flatten().shape[0]

    # print("-This Model Has %d (Approximately %d Million) Parameters!" % (num_model_params, num_model_params//1e6))
    # fm_size = image_size//(2 ** len(block_widths))
    # print("-The Latent Space Size Is %dx%dx%d!" % (latent_channels, fm_size, fm_size))


    # # Start training loop
    # start_epoch = 0
    # nepoch = 1000
    # kl_scale = 1
    # data_logger = defaultdict(lambda: [])
    # save_interval = 256
    # save_dir = f'corpora/{corpus_name}/models/results'
    save_dir = 'trained/results'
    # model_name = 'RES_VAE'
    # save_file_name = 'RES_VAE_dynamic'
    os.makedirs(f'{save_dir}/images', exist_ok=True)
    # os.makedirs(f'{save_dir}/trained', exist_ok=True)
    # for epoch in trange(start_epoch, nepoch, leave=False):
    #     vae_net.train()
    #     for i, images in enumerate(tqdm(train_loader, leave=False)):
    #         current_iter = i + epoch * len(train_loader)
    #         images = images.to(device)
    #         bs, c, h, w = images.shape
            
    #         recon_img, mu, log_var = vae_net(images)

    #         kl_loss = hf.kl_loss(mu, log_var)
    #         mse_loss = F.mse_loss(recon_img, images)
    #         loss = kl_scale * kl_loss + mse_loss

    #         # Perception loss
    #         if feature_extractor is not None:
    #             feat_in = torch.cat((recon_img, images), 0)
    #             feature_loss = feature_extractor(feat_in)
    #             loss += feature_scale * feature_loss
    #             data_logger["feature_loss"].append(feature_loss.item())

    #         optimizer.zero_grad()
    #         loss.backward()
    #         torch.nn.utils.clip_grad_norm_(vae_net.parameters(), 10)

    #         data_logger["mu"].append(mu.mean().item())
    #         data_logger["mu_var"].append(mu.var().item())
    #         data_logger["log_var"].append(log_var.mean().item())
    #         data_logger["log_var_var"].append(log_var.var().item())
    #         data_logger["kl_loss"].append(kl_loss.item())
    #         data_logger["img_mse"].append(mse_loss.item())

    #         if (current_iter + 1) % save_interval == 0:
    #             vae_net.eval()
    #             with torch.no_grad():
    #                 recon_img, mu, log_var = vae_net(test_images.to(device))
    #                 data_logger['test_mse_loss'].append(F.mse_loss(recon_img, test_images.to(device)).item())
    #                 img_cat = torch.cat((recon_img.cpu(), test_images), 2).float()
    #                 vutils.save_image(img_cat, "%s/%s/%s_%d_test-%d.png" % (save_dir, "images", model_name, image_size, current_iter), normalize=True)
    #                 if os.path.isfile(save_dir + "/trained/" + save_file_name + ".pt"):
    #                     shutil.copyfile(src=save_dir + "/trained/" + save_file_name + ".pt", 
    #                                     dst=save_dir + "/trained/" + save_file_name + "_copy.pt")
    #                 torch.save({'epoch': epoch + 1, 'data_logger': dict(data_logger), 'model_state_dict': vae_net.state_dict(),
    #                             'optimizer_state_dict': optimizer.state_dict()}, save_dir + "/trained/" + save_file_name + ".pt")
    #                 vae_net.train()


    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    num_epochs = 100

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images in train_loader:
            images = images.to(device)
            outputs = model(images)
            loss = criterion(outputs, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        epoch_loss = running_loss / len(train_loader.dataset)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}")

            model.eval()
            with torch.no_grad():
                for data in test_loader:
                    data = data.to(device)
                    recon = model(data)
                    break
            fig, ax = plt.subplots(2, 7, figsize=(15, 4))
            for i in range(7):
                ax[0, i].imshow(data[i].cpu().numpy().transpose((1, 2, 0)))
                # ax[0, i].imshow(librosa.power_to_db(data[i].cpu().numpy().transpose(1, 2, 0), ref=np.max))
                ax[0, i].axis('off')
                ax[1, i].imshow(recon[i].cpu().numpy().transpose((1, 2, 0)))
                ax[1, i].axis('off')
            ax[0, 0].set_title('Original')
            ax[1, 0].set_title('Reconstructed')
            plt.savefig(f'{save_dir}/images/conv_AE_recon-{epoch+1}.png')

    torch.save(model.state_dict(), f'{save_dir}/conv_AE.pth')
    model.load_state_dict(torch.load(f'{save_dir}/conv_AE.pth', weights_only=True))
    # model.eval()
    # with torch.no_grad():
    #     for data in test_loader:
    #         data = data.to(device)
    #         recon = model(data)
    #         break

    # fig, ax = plt.subplots(2, 7, figsize=(15, 4))
    # for i in range(7):
    #     ax[0, i].imshow(data[i].cpu().numpy().transpose((1, 2, 0)))
    #     # ax[0, i].imshow(librosa.power_to_db(data[i].cpu().numpy().transpose(1, 2, 0), ref=np.max))
    #     ax[0, i].axis('off')
    #     ax[1, i].imshow(recon[i].cpu().numpy().transpose((1, 2, 0)))
    #     ax[1, i].axis('off')
    # ax[0, 0].set_title('Original')
    # ax[1, 0].set_title('Reconstructed')
    # plt.show()

    # model.eval()
    # latents = []
    # with torch.no_grad():
    #     for data in train_loader:
    #         data = data.to(device)
    #         latent = model.encoder(data).cpu().numpy()
    #         for i in range(latent.shape[0]):
    #             latents.append(latent[i])

    # full_latents = np.array([lat.reshape(-1) for lat in latents])
    # scaler = StandardScaler()
    # full_latents_scaled = scaler.fit_transform(full_latents)
    # pca = PCA(n_components=3)
    # X_PCA = pca.fit_transform(full_latents)
    # print(f'explained variances: {pca.explained_variance_ratio_}')
    # plot_scaler = StandardScaler()
    # X_PCA_scaled = plot_scaler.fit_transform(X_PCA)

    # fig = plt.figure(figsize=(6, 6))
    # ax = fig.add_subplot(projection='3d')
    # ax.scatter(X_PCA_scaled[:,0], X_PCA_scaled[:,1], X_PCA_scaled[:,2])
    # plt.show()
