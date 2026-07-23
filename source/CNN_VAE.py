import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch import optim
import numpy as np
import librosa

import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import os

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
        image = torch.from_numpy(np.load(self.corpus_paths[idx]) / 80)
        # tr = torchaudio.transforms.AmplitudeToDB(stype="power", top_db=80)
        # image = tr(image)
        image = torch.unsqueeze(image, 0).to(torch.float32)
        if self.transform:
            image = self.transform(image)
        return image


class VAE(nn.Module):
    def __init__(
        self, input_dim=784, hidden_dim=400, latent_dim=20,
        likelihood="gaussian"
    ):
        super(VAE, self).__init__()
        self.input_dim = input_dim # input dim depends on dataset
        self.hidden_dim = hidden_dim # hidden dim for first linear layer
        self.latent_dim = latent_dim # dimension in latent space

        # Encoder layers
        # first linear
        self.linear_encode = nn.Linear(input_dim, hidden_dim)
        # Mean of latent space
        self.linear_mu = nn.Linear(hidden_dim, latent_dim)
        # Log variance of latent space  
        self.linear_logvar = nn.Linear(hidden_dim, latent_dim)  

        # Decoder layers
        # first linear
        self.linear_decode = nn.Linear(latent_dim, hidden_dim)
        # map to likelihood parameters
        self.linear_likelihood = nn.Linear(
            hidden_dim, input_dim
        )
        if likelihood not in ["gaussian", "bernoulli"]:
            raise ValueError("Likelihood must be either 'gaussian'"
                              "or 'bernoulli'"
            )
        else:
            self.likelihood = likelihood

    def encode(self, x):
            h = F.relu(self.linear_encode(x))
            return self.linear_mu(h), self.linear_logvar(h)
    def decode(self, z):
            h = F.relu(self.linear_decode(z))
            if self.likelihood == "gaussian":
                return self.linear_likelihood(h)
            elif self.likelihood == "bernoulli":
                return torch.sigmoid(self.linear_likelihood(h))
    def sample(self, mu, logvar):
        """Sample from Gaussian posterior of z with reparametrization"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    def forward(self, x):
        mu, logvar = self.encode(x) # get posterior parameters
        z = self.sample(mu, logvar) # sample with reparametrization
        recon_x = self.decode(z) # get likelihood parameters
        return recon_x, mu, logvar


def loss_function(recon_x, x, mu, logvar, likelihood="gaussian", beta=1.0):
    """Computes the VAE loss."""
    
    if likelihood == "gaussian":
        recon_loss = F.mse_loss(recon_x, x, reduction="sum")
    elif likelihood == "bernoulli":
        recon_loss = F.binary_cross_entropy(recon_x, x, reduction="sum")
    else:
        raise ValueError("Likelihood must be either 'gaussian' or 'bernoulli'")
    # KL divergence between the posterior and the prior (both Gaussian)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + beta * KLD

def train(epoch, model, train_loader, optimizer, device):
    """Trains the VAE for one epoch."""
    model.train()
    train_loss = 0
    likelihood = model.likelihood
    for batch_idx, data in enumerate(train_loader):
        data = data.to(device).view(-1, model.input_dim)  # Flatten the data
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar, likelihood=likelihood)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    return train_loss




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
    
    print(f'-Number of events in dataset: {len(mels)}')
    # compute_stats = []
    # for mel in mels:
    #     compute_stats.append(mel.reshape(-1))
    # compute_stats = np.concatenate(compute_stats, axis=0) # OR AXIS 1
    # mean = compute_stats.mean()
    # std = compute_stats.std()
    # min = compute_stats.min()
    # max = compute_stats.max()



    # Create dataloaders
    image_size = 64
    print("-Target Image Size: %d" % image_size)
    transform = transforms.Compose([transforms.Resize(image_size),
                                    transforms.CenterCrop(image_size)])
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
    image_size = [test_images.shape[2], test_images.shape[3]]
    image_size_flat = image_size[0] * image_size[1]

    model = VAE(
        input_dim=image_size_flat,
        hidden_dim=400,
        latent_dim=20,
        likelihood="bernoulli"
    ).to(device)

    epochs = 2
    learning_rate = 1e-3
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train the model
    for epoch in range(1, epochs + 1):
        epoch_loss = train(epoch, model, train_loader, optimizer, device)
        print(f'{epoch} loss {epoch_loss}')

    with torch.no_grad():
        sample = test_images[0] #random choice I made
        recon, mu, logvar = model(sample.view(-1, image_size_flat).to(device))
        mu, logvar

    fig, (ax1, ax2) = plt.subplots(ncols=2)
    ax1.imshow(sample.squeeze(), cmap="gray")
    ax1.set_title("Original")
    recon_sample = torch.bernoulli(recon.view(image_size).detach().cpu().squeeze())
    ax2.imshow(recon.view(image_size).detach().cpu().squeeze(), cmap="gray")
    ax2.set_title("Reconstructed")
    plt.show()