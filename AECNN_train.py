import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision.transforms as transforms


class MelSpectrogramDataset(Dataset):
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
        image = torch.unsqueeze(image, 0).to(torch.float32)
        if self.transform:
            image = self.transform(image)
        return image


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 16, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(16, 8, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(8, 4, 3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4, 8, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 32, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
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
    corpus_name = 'guitar'
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
                mels.append(mel)
                np.save(f'./{save_full_mels_path}/event-{i}.npy', np.array(mel))
                i += 1
    
    print(f'-Number of events in dataset: {len(mels)}')
    compute_stats = []
    for mel in mels:
        compute_stats.append(mel.reshape(-1))
    compute_stats = np.concatenate(compute_stats, axis=0) # OR AXIS 1

    # Create dataloaders
    image_size = 64
    print("-Target Image Size: %d" % image_size)
    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size)])

    data_set = MelSpectrogramDataset(save_full_mels_path, transform=transform)

    test_split = 0.9
    n_train_examples = int(len(data_set) * test_split)
    n_test_examples = len(data_set) - n_train_examples
    train_set, test_set = torch.utils.data.random_split(data_set, [n_train_examples, n_test_examples],
                                                        generator=torch.Generator().manual_seed(42))
    batch_size = 8
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)
    dataiter = iter(test_loader)
    test_images = next(dataiter)

    model_name = 'AECNN'
    save_dir = f'corpora/{corpus_name}/trained/{model_name}'
    os.makedirs(f'{save_dir}/images', exist_ok=True)

    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 40

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
                ax[0, i].axis('off')
                ax[1, i].imshow(recon[i].cpu().numpy().transpose((1, 2, 0)))
                ax[1, i].axis('off')
            ax[0, 0].set_title('Original')
            ax[1, 0].set_title('Reconstructed')
            plt.savefig(f'{save_dir}/images/{model_name}_recon-{epoch+1}.png')
            torch.save(model.state_dict(), f'{save_dir}/{model_name}.pth')

    model.eval()
    latents = []
    with torch.no_grad():
        for data in train_loader:
            data = data.to(device)
            latent = model.encoder(data).cpu().numpy()
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
