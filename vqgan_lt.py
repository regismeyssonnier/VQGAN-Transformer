# -*- coding: latin-1 -*-

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim.lr_scheduler as lr_scheduler
from PIL import Image
from tqdm import tqdm
import time
from torch.utils.data import Subset
import lpips
 
class CIFAR10VQGANDataset(torch.utils.data.Dataset):
    def __init__(self, train=True):
        from torchvision import transforms

                
        #self.transform = transforms.Compose([
        #    transforms.Resize((32, 32)),  # utile si VQGAN ne prend que du 32x32
        #    transforms.ToTensor(),
        #])
        #self.dataset = CIFAR10(root='./data', train=train, download=True, transform=self.transform)

        self.transform = transforms.Compose([
            transforms.Resize((32, 32)),    # Redimensionner les images à 256x256 pixels
            transforms.ToTensor(),            # Convertir les images en tensors
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalisation pour les modèles générateurs
        ])
        dataset_path = './dataset'  # Remplacez par le chemin réel de votre dataset
        self.dataset = datasets.ImageFolder(root=dataset_path, transform=self.transform)
        #batch_size = 8  # Vous pouvez ajuster le batch size selon votre besoin
        #train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label


def load_cat_tokens(vqgan, batch_size=8):
    dataset = CIFAR10VQGANDataset(vqgan)
    #dataset = Subset(dataset, list(range(200))) 
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)


class TransformerDecoder(nn.Module):
    def __init__(self, z_channels=256, out_channels=3, num_heads=8, num_layers=1, d_model=64, seq_length=64, num_embeddings=512):
        super(TransformerDecoder, self).__init__()

        self.seq_length = seq_length
        self.out_channels = out_channels
        self.side_length = int(seq_length ** 0.5)

        # Positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(seq_length, d_model))

        # Embedding
        self.embedding = nn.Embedding(num_embeddings, d_model)

        # Transformer layers
        self.transformer_layers = nn.ModuleList(
            [nn.TransformerDecoderLayer(d_model, num_heads) for _ in range(num_layers)]
        )

        # Linear projection to image features
        self.fc_out = nn.Linear(d_model, out_channels)

        # Upsample: from (out_channels, 8, 8) to (out_channels, 32, 32)
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(out_channels, 64, kernel_size=4, stride=2, padding=1),  # 8 -> 16
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),            # 16 -> 32
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, kernel_size=3, padding=1)                      # final conv
        )

    def forward(self, z_q):
        # z_q: (batch_size, seq_length, z_channels)
        batch_size, seq_length, z_channels = z_q.shape

        side_length = self.side_length
        #if side_length * side_length != seq_length:
        #    raise ValueError(f"seq_length {seq_length} is not a perfect square")

        # Convert quantized features to indices
        z_q_indices = z_q.argmax(dim=-1)  # (batch_size, seq_length)

        # Embedding + positional encoding
        z_q_embed = self.embedding(z_q_indices) + self.positional_encoding[:seq_length, :]

        # Transformer decoding
        for layer in self.transformer_layers:
            z_q_embed = layer(z_q_embed, z_q_embed)

        # Project to image space: (batch_size, seq_length, out_channels)
        out = self.fc_out(z_q_embed)

        # Reshape to (batch_size, out_channels, height, width)
        out = out.permute(0, 2, 1).contiguous()
        out = out.view(batch_size, self.out_channels, side_length, side_length)

        # Upsample to final resolution
        #out = self.upsample(out)

        return out



class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)

    def forward(self, inputs):
        inputs = inputs.permute(0, 2, 3, 1).contiguous()  # -> (B, H, W, C)
        flat_input = inputs.view(-1, self.embedding_dim)

        distances = (
            torch.sum(flat_input**2, dim=1, keepdim=True)
            - 2 * torch.matmul(flat_input, self.embedding.weight.t())
            + torch.sum(self.embedding.weight**2, dim=1)
        )

        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)

        quantized = torch.matmul(encodings, self.embedding.weight).view(inputs.shape)
        quantized = quantized.permute(0, 3, 1, 2).contiguous()  # -> (B, C, H, W)

        # loss
        commitment_loss = self.commitment_cost * F.mse_loss(quantized.detach(), inputs.permute(0, 3, 1, 2))
        embedding_loss = F.mse_loss(quantized, inputs.permute(0, 3, 1, 2).detach())
        loss = commitment_loss + embedding_loss

        quantized = inputs.permute(0, 3, 1, 2) + (quantized - inputs.permute(0, 3, 1, 2)).detach()
        return quantized, loss, encoding_indices.view(inputs.shape[0], inputs.shape[1], inputs.shape[2])


class Encoder_o(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=128, z_channels=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, z_channels, 4, stride=2, padding=1)
        )

    def forward(self, x):
        return self.model(x)

class Encoder_dernier(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=128, z_channels=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 4, stride=2, padding=1),  # 256 → 128
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 4, stride=2, padding=1),  # 128 → 64
            nn.ReLU(),
            nn.Conv2d(hidden_channels, z_channels, 4, stride=2, padding=1),  # 64 → 32
            #nn.ReLU(),
            #nn.Conv2d(hidden_channels, z_channels, 4, stride=2, padding=1)  # 32 → 16
        )
        # version qui sort un code latent en 128x128
        #self.model = nn.Sequential(
        #    nn.Conv2d(in_channels, hidden_channels, 4, stride=2, padding=1),  # 256 → 128
        #    nn.ReLU(),
        #    nn.Conv2d(hidden_channels, z_channels, 1)  # juste changer de dim
        #)


    def forward(self, x):
        return self.model(x)


class Decoder_o(nn.Module):
    def __init__(self, out_channels=3, hidden_channels=128, z_channels=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_channels, hidden_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, out_channels, 4, stride=2, padding=1),
            nn.Tanh()  # ou tanh selon preprocessing
          
        )

    def forward(self, z):
        return self.model(z)

class Decoder_dernier(nn.Module):
    def __init__(self, out_channels=3, hidden_channels=128, z_channels=256):
        super().__init__()
        self.model = nn.Sequential(
            #nn.ConvTranspose2d(z_channels, out_channels, 4, stride=2, padding=1),  # 16 → 32
            #nn.ReLU(),
            nn.ConvTranspose2d(z_channels, hidden_channels, 4, stride=2, padding=1),  # 32 → 64
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, hidden_channels, 4, stride=2, padding=1),  # 64 → 128
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_channels, out_channels, 4, stride=2, padding=1),  # 128 → 256
            nn.Tanh()
        )

        #self.model = nn.Sequential(
        #    nn.ConvTranspose2d(z_channels, 128, kernel_size=4, stride=2, padding=1),  # 128x128 → 256x256
        #    nn.ReLU(),
        #    nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),  # [B, 128, 256, 256] → [B, 64, 256, 256]
        #    nn.ReLU(),
        #    nn.Conv2d(64, out_channels, kernel_size=3, stride=1, padding=1),  # [B, 64, 256, 256] → [B, 3, 256, 256]
        #    nn.Tanh()
        #)

        #size 128x128
        #self.model = nn.Sequential(
        #    nn.ConvTranspose2d(z_channels, 128, kernel_size=4, stride=2, padding=1),  # 128x128 → 256x256
        #    nn.ReLU(),
        #    nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1),  # Garde 256x256
        #    nn.ReLU(),
        #    nn.ConvTranspose2d(128, out_channels, kernel_size=3, stride=1, padding=1),  # Garde 256x256
        #    nn.Tanh()
        #)

    def forward(self, z):
        return self.model(z)

class Decodera(nn.Module):
    def __init__(self, out_channels=3, hidden_channels=128, z_channels=256):
        super().__init__()
        # Un seul ConvTranspose2d pour faire passer de 16x16 à 256x256
        self.model = nn.Sequential(
            nn.ConvTranspose2d(z_channels, out_channels, 16, stride=16, padding=0),  # 16 → 256
            nn.Tanh()  # Normalisation de l'image (si nécessaire)
        )

    def forward(self, z):
        return self.model(z)


import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(32, dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
            nn.GroupNorm(32, dim),
            nn.SiLU(),
            nn.Conv2d(dim, dim, 3, padding=1),
        )

    def forward(self, x):
        return x + self.block(x)

class DownsampleBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.down = nn.Conv2d(dim, dim, 4, stride=2, padding=1)

    def forward(self, x):
        return self.down(x)

class UpsampleBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.up = nn.ConvTranspose2d(dim, dim, 4, stride=2, padding=1)

    def forward(self, x):
        return self.up(x)

class NonLocalBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=4, batch_first=True)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).transpose(1, 2)  # B x HW x C
        x_attn, _ = self.attn(x_flat, x_flat, x_flat)
        x_attn = self.norm(x_attn + x_flat)
        return x_attn.transpose(1, 2).view(B, C, H, W)

class Encoder(nn.Module):
    def __init__(self, in_channels=3, C1=128, C2=256, z_dim=256):
        super().__init__()
        self.initial = nn.Conv2d(in_channels, C1, 3, padding=1)
        self.conv2 = nn.Conv2d(C1, C2, 3, padding=1)

        self.down_blocks = nn.Sequential(
            ResidualBlock(C2),
            ResidualBlock(C2),
            DownsampleBlock(C2)  # 32x32 -> 16x16
        )

        self.middle = nn.Sequential(
            ResidualBlock(C2),
            NonLocalBlock(C2),
            ResidualBlock(C2),
            NonLocalBlock(C2)
        )

        self.final = nn.Sequential(
            nn.GroupNorm(32, C2),
            nn.SiLU(),
            nn.Conv2d(C2, z_dim, 1)
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.conv2(x)
        x = self.down_blocks(x)
        x = self.middle(x)
        return self.final(x)

class Decoder(nn.Module):
    def __init__(self, out_channels=3, C1=128, C2=256, z_dim=256):
        super().__init__()
        self.initial = nn.Sequential(
            nn.GroupNorm(32, z_dim),
            nn.SiLU(),
            nn.Conv2d(z_dim, C2, 1)
        )

        self.middle = nn.Sequential(
            ResidualBlock(C2),
            NonLocalBlock(C2),
            ResidualBlock(C2)
        )

        self.up_blocks = nn.Sequential(
            ResidualBlock(C2),
            ResidualBlock(C2),
            UpsampleBlock(C2)  # 16x16 -> 32x32
        )

        self.final = nn.Conv2d(C2, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.initial(x)
        x = self.middle(x)
        x = self.up_blocks(x)
        return self.final(x)

class Discriminator(nn.Module):
    def __init__(self, in_channels=3, base_channels=64, num_layers=4):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, base_channels, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True)
        ]
        curr_channels = base_channels
        for _ in range(1, num_layers):
            layers += [
                nn.Conv2d(curr_channels, curr_channels * 2, 4, stride=2, padding=1),
                nn.BatchNorm2d(curr_channels * 2),
                nn.LeakyReLU(0.2, inplace=True)
            ]
            curr_channels *= 2
        layers.append(nn.Conv2d(curr_channels, 1, 1))  # Patch logits
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)




class VQGAN(nn.Module):
    def __init__(self, num_embeddings=1024, embedding_dim=256, commitment_cost=0.25):
        super().__init__()
        
        # Encoder: transforme image 256x256 en latent 16x16x256
        self.encoder = Encoder(
            in_channels=3,
            C1=128,       # Feature dim intermédiaire
            C2=256,       # Dimension du latent
            z_dim=embedding_dim
        )
        
        # Vector Quantizer
        self.quantizer = VectorQuantizer(
            num_embeddings=num_embeddings,
            embedding_dim=embedding_dim,
            commitment_cost=commitment_cost
        )
        
        # Decoder: transforme latent 16x16x256 en image 256x256
        self.decoder = Decoder(
            out_channels=3,
            C1=128,
            C2=256,
            z_dim=embedding_dim
        )

        self.discriminator = Discriminator(in_channels=3)

    def forward(self, x):
        # Encode image -> z_e
        z_e = self.encoder(x)  # (B, 256, 16, 16)

        # Quantize latent vector
        z_q, vq_loss, _ = self.quantizer(z_e)  # (B, 256, 16, 16)

        # Decode to reconstruct image
        x_recon = self.decoder(z_q)  # (B, 3, 256, 256)

        return x_recon, vq_loss

class VQGANT(nn.Module):
    def __init__(self, in_channels=3, z_channels=256, hidden_channels=128, out_channels=3, num_embeddings=16384, commitment_cost=0.25):
        super(VQGANT, self).__init__()

        # D finir l'encodeur
        self.encoder = Encoder(in_channels, hidden_channels, z_channels)
        
        # Instancier le quantizer avec les arguments num_embeddings et commitment_cost
        self.quantizer = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=z_channels, commitment_cost=commitment_cost)
        
        # Instancier le transformateur d codeur
        self.transformer_decoder = TransformerDecoder(z_channels=z_channels, out_channels=out_channels)

    def forward(self, x):
        # Passer l'image   travers l'encodeur
        z = self.encoder(x)  # (batch_size, z_channels, height, width)
        
        # Appliquer la quantification et obtenir la reconstruction
        z_q, vq_loss, _ = self.quantizer(z)  # z_q devrait  tre de la forme (batch_size, z_channels, height, width)
        print("shape=", z_q.shape)

        # Adapter la forme de z_q pour qu'elle corresponde aux attentes du transformer (flatten + permutation)
        batch_size, z_channel, height, width = z_q.shape
        z_q_flattened = z_q.view(batch_size, z_channel, -1).permute(0, 2, 1)  # (batch_size, seq_length, z_channels)

        # Passer la sortie quantifi e (r organis e) au d codeur
        x_recon = self.transformer_decoder(z_q_flattened)
        x_recon = F.interpolate(x_recon, size=(32, 32), mode='bilinear', align_corners=False)


        return x_recon, vq_loss



def loss_fno(x, x_recon, vq_loss):
    recon_loss = F.mse_loss(x_recon, x)
    return recon_loss + vq_loss


def loss_fn(x, x_recon, vq_loss, perceptual_weight=0.2):
    # MSE classique
    recon_loss = F.mse_loss(x_recon, x)

    # LPIPS attend des images entre [-1, 1] → normaliser si nécessaire
    # Supposons que x et x_recon sont déjà dans [-1, 1]
    perceptual_loss = lpips_loss_fn(x_recon, x).mean()

    # Total loss
    total_loss = recon_loss + vq_loss + perceptual_weight * perceptual_loss
    return total_loss

import torch.nn.functional as F

def loss_fn_vqgan_adv(x, x_recon, vq_loss, D, perceptual_weight=0.02, adv_weight=0.1):
    # Reconstruction loss (MSE)
    recon_loss = F.mse_loss(x_recon, x)

    # Perceptual loss (LPIPS) - on suppose que les images sont déjà entre [-1, 1]
    perceptual_loss = lpips_loss_fn(x_recon, x).mean()

    # Adversarial loss (générateur)
    logits_fake = D(x_recon)
    g_loss = -logits_fake.mean()

    # Total generator loss
    total_loss = recon_loss + vq_loss + perceptual_weight * perceptual_loss + adv_weight * g_loss
    return total_loss, {
        "recon": recon_loss.item(),
        "perceptual": perceptual_loss.item(),
        "vq": vq_loss.item(),
        "adv": g_loss.item(),
    }

def discriminator_loss(D, x_real, x_fake):
    # Discriminator: vraie = 1, fausse = 0
    logits_real = D(x_real)
    logits_fake = D(x_fake.detach())
    d_loss = (F.relu(1.0 - logits_real)).mean() + (F.relu(1.0 + logits_fake)).mean()
    return d_loss


INDEX = 0
def show_reconstructed_images(data, x_recon):
    global INDEX
    mean = torch.tensor([0.485, 0.456, 0.406],  device=data.device)  # Convertir mean en Tensor
    std = torch.tensor([0.229, 0.224, 0.225], device=data.device)   # Convertir std en Tensor

    # Cr er un graphique avec deux images c te   c te
    fig, ax = plt.subplots(1, 2, figsize=(5, 2.5))

    image_standardized = data[0].permute(1, 2, 0).cpu().numpy()
    image_denormalized = image_standardized #* std.cpu().numpy() + mean.cpu().numpy()
    image_denormalized = np.clip(image_denormalized, 0, 1)
    image_denormalized_255 = (image_denormalized * 255).astype(np.uint8)

    
    # Afficher l'image originale
    ax[0].imshow(image_denormalized_255)
    ax[0].set_title("Original Image")
    ax[0].axis('off')  # D sactive les axes pour une meilleure visualisation
    
    # Afficher l'image reconstruite
    # 1. Extraire l'image et remapper de [-1, 1] à [0, 1]
    reconstructed_image = x_recon[0].cpu().detach().numpy().transpose(1, 2, 0)
    reconstructed_image = (reconstructed_image + 1) / 2
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    #reconstructed_image = reconstructed_image * std + mean
    reconstructed_image = np.clip(reconstructed_image, 0, 1)
    reconstructed_image = (reconstructed_image * 255).astype(np.uint8)
    
    #print(reconstructed_image[128, 128]) 
    # Afficher l'image reconstruite
    ax[1].imshow(reconstructed_image)
    ax[1].set_title("Reconstructed Image")
    ax[1].axis('off')  # D sactive les axes pour une meilleure visualisation
    
    #plt.savefig('reconstructed_image' + str(INDEX) + '.png', bbox_inches='tight', pad_inches=0.1)
    # Afficher le graphique
    plt.tight_layout()
    plt.show()
    INDEX+=1


# Param tres d'entra nement
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epochs = 200
learning_rate = 1e-4

# Chargement du dataset CIFAR-10 pour tester (tu peux utiliser ton propre dataset)
#transform = transforms.Compose([
#    transforms.Resize((256, 256)),    # Redimensionner les images à 224x224 pixels
#    transforms.ToTensor(),            # Convertir les images en tensors
#    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalisation (pour un modèle préentraîné comme ResNet)
#])
#dataset_path = './dataset'  # Remplacez par le chemin réel de votre dataset
#dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
#dataset = Subset(dataset, list(range(64))) 
batch_size = 8  # Vous pouvez ajuster le batch size selon votre besoin
##DataLoader(dataset, batch_size=batch_size, shuffle=True)

#image, label = dataset[0]  # Récupérer la première image et son étiquette
#print("Image:", image.shape)  # Afficher la forme de l'image
#print("Label:", label)  # Afficher l'étiquette (classe)
#image = image.permute(1, 2, 0).numpy()  # Convertir de (C, H, W) à (H, W, C)

# Afficher l'image
#plt.imshow(image)
#plt.title(f'Label: {label}')
#plt.axis('off')  # Désactiver les axes
#plt.show()

# Instancier le mod le et l'optimiseur
# VQGAN et Discriminateur
model = VQGAN().to(device)
discriminator = Discriminator().to(device)

# Optimizers
optimizer_g = optim.Adam(model.parameters(), lr=learning_rate)
optimizer_d = optim.Adam(discriminator.parameters(), lr=learning_rate)

scheduler = lr_scheduler.ReduceLROnPlateau(optimizer_g, 'min', patience=3, factor=0.1)

train_loader = load_cat_tokens(model)
lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)  # ou 'alex', 'squeeze'
lpips_loss_fn.eval() 

def train(model, discriminator, dataloader, optimizer_g, optimizer_d, device, scheduler, epochs=20):
    model.train()
    discriminator.train()

    for epoch in range(epochs):
        total_loss = 0
        index = 0
        loop = tqdm(dataloader, desc=f'Epoch {epoch + 1}', leave=True)

        for (data, _) in loop:
            data = data.to(device)
            optimizer_g.zero_grad()
            optimizer_d.zero_grad()

            # === Forward VQGAN ===
            x_recon, vq_loss = model(data)

            # === Discriminator loss ===
            d_loss = discriminator_loss(discriminator, data, x_recon)
            d_loss.backward()
            optimizer_d.step()

            # === Generator loss ===
            optimizer_g.zero_grad()
            g_loss, loss_dict = loss_fn_vqgan_adv(data, x_recon, vq_loss, discriminator)
            g_loss.backward()
            optimizer_g.step()

            total_loss += g_loss.item()
            loop.set_postfix({
                "loss": g_loss.item(),
                "recon": loss_dict["recon"],
                "percep": loss_dict["perceptual"],
                "adv": loss_dict["adv"]
            })

            if (index) % 500 == 0:
                #print(x_recon[0, :, 128, 128])
                show_reconstructed_images(data, x_recon)
            
            index += 1

        avg_loss = total_loss / len(dataloader)
        scheduler.step(avg_loss)
        print(f"Epoch [{epoch+1}/{epochs}], Total Loss: {avg_loss:.4f}")

        # Save models
        torch.save(model.state_dict(), 'vqgan_model_car_newld32.pth')
        torch.save(discriminator.state_dict(), 'vqgan_discriminator_car_newld32.pth')

    print("Training complete!")


def eval(filename, model, dataloader, device, epochs=20):
    model.eval()  # Passer en mode évaluation (désactive les Dropout et BatchNorm si présents)
    model.load_state_dict(torch.load(filename))

    # Désactivation de l'optimiseur car l'évaluation ne modifie pas les poids
    with torch.no_grad():  # Désactive la création de gradients pour économiser de la mémoire
        
        index = 0
        for batch_idx, (data, _) in enumerate(dataloader):
            data = data.to(device)

            # Forward pass
            x_recon, vq_loss = model(data)

            # Afficher les images reconstruites périodiquement
            show_reconstructed_images(data, x_recon)
                        
            index += 1
            if index > epochs:
                break
                
           

    print("Evaluation complete!")


# Entra ner le mod le
train(model, discriminator, train_loader, optimizer_g, optimizer_d, device, scheduler, epochs=20)
#eval('vqgan_model3.pth', model, train_loader, device, 10)

#torch.save(model.state_dict(), 'vqgan_model3.pth')
#torch.save(model.state_dict(), 'vqgan_model_tr.pth')