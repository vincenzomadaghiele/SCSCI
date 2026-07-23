import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms


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
        image = torch.clip(torch.from_numpy((np.load(self.corpus_paths[idx])+80) / 80), min=0, max=1)
        # image = torch.from_numpy(np.load(self.corpus_paths[0]) / 80)
        image = torch.unsqueeze(image, 0).to(torch.float32)
        if self.transform:
            image = self.transform(image)
        return image

class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(Encoder, self).__init__()
        self.FC_input = nn.Linear(input_dim, hidden_dim)
        self.FC_input2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_mean  = nn.Linear(hidden_dim, latent_dim)
        self.FC_var   = nn.Linear (hidden_dim, latent_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
        self.training = True
    def forward(self, x):
        h_       = self.LeakyReLU(self.FC_input(x))
        h_       = self.LeakyReLU(self.FC_input2(h_))
        mean     = self.FC_mean(h_)
        log_var  = self.FC_var(h_)
        return mean, log_var
    
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super(Decoder, self).__init__()
        self.FC_hidden = nn.Linear(latent_dim, hidden_dim)
        self.FC_hidden2 = nn.Linear(hidden_dim, hidden_dim)
        self.FC_output = nn.Linear(hidden_dim, output_dim)
        self.LeakyReLU = nn.LeakyReLU(0.2)
    def forward(self, x):
        h     = self.LeakyReLU(self.FC_hidden(x))
        h     = self.LeakyReLU(self.FC_hidden2(h))
        x_hat = torch.sigmoid(self.FC_output(h))
        return x_hat
    
class Model(nn.Module):
    def __init__(self, Encoder, Decoder):
        super(Model, self).__init__()
        self.Encoder = Encoder
        self.Decoder = Decoder
    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(DEVICE)
        z = mean + var*epsilon
        return z        
    def forward(self, x):
        mean, log_var = self.Encoder(x)
        z = self.reparameterization(mean, torch.exp(0.5 * log_var)) # takes exponential function (log var -> var)
        x_hat            = self.Decoder(z)
        return x_hat, mean, log_var


if __name__ == "__main__":

    # device 
    DEVICE = torch.device("cpu")
    if torch.cuda.is_available():
        DEVICE = torch.device("cuda")
    elif torch.backends.mps.is_available():
        DEVICE = torch.device("mps")

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
                # mel = np.log(event['mel'] + 1e-7)
                mels.append(mel)
                np.save(f'./{save_full_mels_path}/event-{i}.npy', np.array(mel))
                i += 1
            
    print(f'-Number of events in dataset: {len(mels)}')
    # Create dataloaders
    image_size = 64
    print("-Target Image Size: %d" % image_size)
    transform = transforms.Compose([transforms.Resize(image_size), transforms.CenterCrop(image_size)])
    data_set = CustomImageDataset(save_full_mels_path, transform=transform)
    test_split = 0.9
    n_train_examples = int(len(data_set) * test_split)
    n_test_examples = len(data_set) - n_train_examples
    train_set, test_set = torch.utils.data.random_split(data_set, [n_train_examples, n_test_examples], 
                                                        generator=torch.Generator().manual_seed(42))
    batch_size = 16
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    dataiter = iter(test_loader)
    test_images = next(dataiter)
    image_size_flat = image_size * image_size

    hidden_dim = 128
    latent_dim = 8
    lr = 1e-3
    epochs = 1000

    encoder = Encoder(input_dim=image_size_flat, hidden_dim=hidden_dim, latent_dim=latent_dim)
    decoder = Decoder(latent_dim=latent_dim, hidden_dim = hidden_dim, output_dim = image_size_flat)
    model = Model(Encoder=encoder, Decoder=decoder).to(DEVICE)

    BCE_loss = nn.BCELoss()
    def loss_function(x, x_hat, mean, log_var):
        reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
        KLD      = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
        return reproduction_loss + KLD

    optimizer = Adam(model.parameters(), lr=lr)
    save_dir = f'corpora/{corpus_name}/trained/VAE'
    os.makedirs(f'{save_dir}/images', exist_ok=True)

    print("Start training VAE...")
    model.train()
    losses = []
    for epoch in range(epochs):
        overall_loss = 0
        for batch_idx, x in enumerate(train_loader):
            x = x.view(x.shape[0], image_size_flat)
            x = x.to(DEVICE)
            optimizer.zero_grad()
            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            overall_loss += loss.item()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {overall_loss / ((batch_idx+1)*batch_size):.4f}")
            model.eval()
            with torch.no_grad():
                for x in test_loader:
                    x = x.view(x.shape[0], image_size_flat)
                    x = x.to(DEVICE)
                    x_hat, mean, log_var = model(x)
                    break
            x = x.view(x.shape[0], image_size, image_size)
            x_hat = x_hat.view(x.shape[0], image_size, image_size)
            fig, ax = plt.subplots(2, 7, figsize=(15, 4))
            for i in range(7):
                ax[0, i].imshow(x[i].cpu().numpy())
                ax[0, i].axis('off')
                ax[1, i].imshow(x_hat[i].cpu().numpy())
                ax[1, i].axis('off')
            ax[0, 0].set_title('Original')
            ax[1, 0].set_title('Reconstructed')
            plt.savefig(f'{save_dir}/images/conv_VAE_recon-{epoch+1}.png')
            torch.save(model.state_dict(), f'{save_dir}/conv_VAE.pth')
        losses.append(overall_loss / ((batch_idx+1)*batch_size))

    model.eval()
    latents = []
    with torch.no_grad():
        for x in train_loader:
            x = x.view(x.shape[0], image_size_flat)
            x = x.to(DEVICE)
            latent, _ = model.Encoder(x)
            latent = latent.cpu().numpy()
            for i in range(latent.shape[0]):
                latents.append(latent[i])

    full_latents = np.array([lat.reshape(-1) for lat in latents])
    scaler = StandardScaler()
    full_latents_scaled = scaler.fit_transform(full_latents)
    pca = PCA(n_components=3)
    X_PCA = pca.fit_transform(full_latents)
    print(f'explained variances: {pca.explained_variance_ratio_}')
    plot_scaler = StandardScaler()
    X_PCA_scaled = plot_scaler.fit_transform(X_PCA)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot()
    ax.scatter(X_PCA_scaled[:,0], X_PCA_scaled[:,1], s=2)
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.title('Latent space encodings')
    plt.savefig(f'{save_dir}/images/scatter.png')
