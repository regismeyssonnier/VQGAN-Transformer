# -*- coding: latin-1 -*-
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
import random
import math
from torch.utils.data import Subset
#import lpips

class GPT_o(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, block_size):
        super().__init__()
        self.block_size = block_size
        self.num_heads = num_heads
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, block_size, embed_dim))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
            for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(embed_dim)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def get_ans4(self, x, attention_mask=None):
        B, T = x.shape
        assert T <= self.block_size

        # Embedding + positional encoding
        x = self.embed(x) + self.pos_embed[:, :T, :]  # (B, T, C)
        x = x.transpose(0, 1)  # (T, B, C)

        # Masque causal (autoregressif) : [T, T]
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        # Masque padding : [B, T] → [T, B]
        #key_padding_mask = None
        #if attention_mask is not None:
        #    key_padding_mask = (attention_mask == 0)  # padding = True

        # Pass through Transformer blocks
        for block in self.blocks:
            x = block(x, src_mask=causal_mask)#, src_key_padding_mask=key_padding_mask)

        x = self.ln(x)
        x = x.transpose(0, 1)  # (B, T, C)
        return self.fc_out(x)  # (B, T, vocab_size)


    def forward(self, x):
        B, T = x.shape
        #print(x.shape)
        assert T <= self.block_size

        # Embedding + positional encoding
        x = self.embed(x) + self.pos_embed[:, :T, :]  # (B, T, C)

        # Transpose for TransformerEncoderLayer: needs shape (T, B, C)
        x = x.transpose(0, 1)

        # Generate 2D causal mask of shape (T, T)
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        # Pass through Transformer blocks
        for block in self.blocks:
            x = block(x, src_mask=mask)

        x = self.ln(x)

        # Transpose back: (T, B, C) -> (B, T, C)
        x = x.transpose(0, 1)

        # Récupérer uniquement la sortie du dernier token
        last_token_output = x[:, -1, :]  # (B, C) : dernière position de la séquence

        # Passer cette sortie à travers la couche finale pour prédire un token
        return self.fc_out(last_token_output)  # (B, vocab_size) : Prédiction du token final

class GPT(nn.Module):
    def __init__(self, vocab_size, embed_dim, num_heads, num_layers, block_size, conditioning_dim=1024):
        super().__init__()
        self.block_size = block_size
        self.num_heads = num_heads
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.conditioning_embed = nn.Embedding(conditioning_dim, embed_dim)  # Embedding pour le conditionnement
        self.conditioning_embed2 = nn.Embedding(conditioning_dim, embed_dim)  # Embedding pour le conditionnement
        self.pos_embed = nn.Parameter(torch.zeros(1, block_size, embed_dim))
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads)
            for _ in range(num_layers)
        ])
        self.ln = nn.LayerNorm(embed_dim)
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def get_ans42(self, x, conditioning, conditioning2):
        B, T = x.shape
        assert T <= self.block_size

        # Embedding + positional encoding
        x = self.embed(x) + self.pos_embed[:, :T, :]  # (B, T, C)

    
        # Injection de l'embedding conditionnel
        conditioning_embedding = self.conditioning_embed(conditioning)  # (B, C)
        conditioning_embedding2 = self.conditioning_embed2(conditioning2)  # (B, C)
        x = x + conditioning_embedding  # Ajout du conditioning aux embeddings
        x = x + conditioning_embedding2

        # Transpose for TransformerEncoderLayer: needs shape (T, B, C)
        x = x.transpose(0, 1)

        # Masque causal (autoregressif) : [T, T]
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        # Pass through Transformer blocks
        for block in self.blocks:
            x = block(x, src_mask=causal_mask)#, src_key_padding_mask=key_padding_mask)

        x = self.ln(x)
        x = x.transpose(0, 1)  # (B, T, C)

        return self.fc_out(x)  # (B, T, vocab_size)

    def get_ans4(self, x, conditioning):
        B, T = x.shape
        assert T <= self.block_size

        # Embedding + positional encoding
        x = self.embed(x) + self.pos_embed[:, :T, :]  # (B, T, C)

    
        # Injection de l'embedding conditionnel
        conditioning_embedding = self.conditioning_embed(conditioning)  # (B, C)
        x = x + conditioning_embedding  # Ajout du conditioning aux embeddings

        # Transpose for TransformerEncoderLayer: needs shape (T, B, C)
        x = x.transpose(0, 1)

        # Masque causal (autoregressif) : [T, T]
        causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        # Pass through Transformer blocks
        for block in self.blocks:
            x = block(x, src_mask=causal_mask)#, src_key_padding_mask=key_padding_mask)

        x = self.ln(x)
        x = x.transpose(0, 1)  # (B, T, C)

        return self.fc_out(x)  # (B, T, vocab_size)

    def forward(self, x, conditioning):
        B, T = x.shape
        assert T <= self.block_size

        # Embedding + positional encoding
        x = self.embed(x) + self.pos_embed[:, :T, :]  # (B, T, C)

        # Injection de l'embedding conditionnel
        conditioning_embedding = self.conditioning_embed(conditioning)  # (B, C)
        x = x + conditioning_embedding  # Ajout du conditioning aux embeddings

        # Transpose for TransformerEncoderLayer: needs shape (T, B, C)
        x = x.transpose(0, 1)

        # Masque causal (autoregressif) : [T, T]
        mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()

        # Pass through Transformer blocks
        for block in self.blocks:
            x = block(x, src_mask=mask)

        x = self.ln(x)

        # Transpose back: (T, B, C) -> (B, T, C)
        x = x.transpose(0, 1)

        # Récupérer uniquement la sortie du dernier token
        last_token_output = x[:, -1, :]  # (B, C) : dernière position de la séquence

        # Passer cette sortie à travers la couche finale pour prédire un token
        return self.fc_out(last_token_output)  # (B, vocab_size) : Prédiction du token final



class CIFAR10VQGANDataset(torch.utils.data.Dataset):
    def __init__(self, vqgan, train=True):
        from torchvision.datasets import CIFAR10
        from torchvision import transforms

        self.vqgan = vqgan.eval()  # gèle les poids
        
        #self.transform = transforms.Compose([
        #    transforms.Resize((32, 32)),  # utile si VQGAN ne prend que du 32x32
        #    transforms.ToTensor(),
        #])
        #self.dataset = CIFAR10(root='./data', train=train, download=True, transform=self.transform)

        self.transform = transforms.Compose([
            transforms.Resize((256, 256)),    # Redimensionner les images à 256x256 pixels
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
        image, label = self.dataset[idx]  # image: [3, 256, 256]
    
        patches = []
        patch_size = 32
        side = 256
        # Découpe l'image 256x256 en 4x4 = 16 patchs de 64x64
        for i in range(0, side, patch_size):
            for j in range(0, side, patch_size):
                patch = image[:, i:i+patch_size, j:j+patch_size]  # shape: [3, 64, 64]
                patches.append(patch)
    
        all_indices = []
        label_tokens = torch.full((4,), label, dtype=torch.long)  # 4 tokens identiques
    
        # Passe chaque patch dans le VQGAN
        for patch in patches:
            patch = patch.to(next(self.vqgan.parameters()).device).unsqueeze(0)  # [1, 3, 64, 64]
            with torch.no_grad():
                z = self.vqgan.encoder(patch)                    # [1, C, H, W]
                z_q, _, indices = self.vqgan.quantizer(z)        # [1, H, W]
                indices = indices.view(-1).long()                # [H * W]
                all_indices.append(indices)
    
        # Concaténer tous les indices des 16 patchs
        all_indices = torch.cat(all_indices, dim=0)  # [16 * H*W]
    
        return label_tokens, all_indices, idx

    def __getitem__normal(self, idx):
        image, label = self.dataset[idx]

        # transforme l’image et encode avec VQGAN
        image = image.to(next(self.vqgan.parameters()).device).unsqueeze(0)  # [1, 3, 32, 32]
        with torch.no_grad():
            z = self.vqgan.encoder(image)                   # [1, C, H, W]
            z_q, _, indices = self.vqgan.quantizer(z)       # indices: [1, H, W]
            indices = indices.view(-1).long()               # [H * W], e.g. 8x8 = 64 tokens

        # encode le label comme 4 tokens identiques
        label_tokens = torch.full((4,), label, dtype=torch.long)

        return label_tokens, indices, idx

def load_cat_tokens(vqgan, batch_size=8):
    dataset = CIFAR10VQGANDataset(vqgan)
    dataset = Subset(dataset, list(range(20))) 
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


class VectorQuantizerGT(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizerGT, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)

    def forward(self, inputs):
        batch_size = inputs.shape[0]

        # Flatten the inputs (reshape to [batch_size, num_features])
        transformed_reshaped = inputs.view(batch_size, -1)  # Flatten to [batch_size, num_features]

        # Ensure the reshaped input has the same dimensionality as the embedding
        # Now we compute distances between transformed_reshaped and self.embeddings.weight
        distances = torch.cdist(transformed_reshaped.unsqueeze(1), self.embeddings.weight.unsqueeze(0))  # [batch_size, num_embeddings]

        encoding_indices = torch.argmin(distances, dim=2)  # Find the closest embedding

        # Embeddings associated with the closest indices
        quantized = self.embeddings(encoding_indices)  # Get the closest embeddings

        # Reshape quantized to match the input shape
        quantized = quantized.view(batch_size, 256, 8, 8)  # Assuming output shape should match [batch_size, 3, 8, 8]

        # Compute the quantization loss
        commitment_loss = F.mse_loss(quantized, inputs.detach())
        embedding_loss = F.mse_loss(quantized.detach(), inputs)

        loss = self.commitment_cost * commitment_loss + embedding_loss

        return quantized, loss, encoding_indices


class VectorQuantizerGT2(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost):
        super(VectorQuantizerGT, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost

        # Embeddings
        self.embeddings = nn.Embedding(self.num_embeddings, self.embedding_dim)

    def forward(self, inputs):
        batch_size = inputs.shape[0]

        # Flatten the inputs (reshape to [batch_size, num_features])
        transformed_reshaped = inputs.view(batch_size, -1)  # Flatten to [batch_size, num_features]

        # Ensure the reshaped input has the same dimensionality as the embedding
        # Now we compute distances between transformed_reshaped and self.embeddings.weight
        distances = torch.cdist(transformed_reshaped.unsqueeze(1), self.embeddings.weight.unsqueeze(0))  # [batch_size, num_embeddings]

        encoding_indices = torch.argmin(distances, dim=2)  # Find the closest embedding

        # Embeddings associated with the closest indices
        quantized = self.embeddings(encoding_indices)  # Get the closest embeddings

        # Reshape quantized to match the input shape
        quantized = quantized.view(batch_size, 256, 8, 8)  # Assuming output shape should match [batch_size, 3, 8, 8]

        # Compute the quantization loss
        commitment_loss = F.mse_loss(quantized, inputs.detach())
        embedding_loss = F.mse_loss(quantized.detach(), inputs)

        loss = self.commitment_cost * commitment_loss + embedding_loss

        return quantized, loss, encoding_indices




class Encodero(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=128, z_channels=256):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, z_channels, 4, stride=2, padding=1)
        )

    def forward(self, x):
        return self.model(x)

class Decodero(nn.Module):
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

#----------------------------------------------------NEW ENDDEC VQGAN
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


class VQGANGT(nn.Module):
    def __init__(self, in_channels=3, z_channels=256, hidden_channels=128, out_channels=3, num_embeddings=512, commitment_cost=0.25):
        super().__init__()
        # D finir l'encodeur
        self.encoder = Encoder(in_channels, hidden_channels, z_channels)
        self.decoder = Decoder(in_channels, hidden_channels, z_channels)
        
        # Instancier le quantizer avec les arguments num_embeddings et commitment_cost
        self.quantizer = VectorQuantizerGT(num_embeddings=64, embedding_dim=16384, commitment_cost=commitment_cost)
        
        # Instancier le transformateur d codeur
        self.transformer_decoder = GPT(vocab_size, 64, 4, 1, 1)
        self.fc = nn.Linear(294, 256*8*8)

    def forward(self, x, lab):
        batch_size, ch, h, w = x.shape
        lab_tensor = torch.tensor(lab, dtype=torch.long, device=x.device).view(batch_size, 1)  # [batch_size, 1]
        #print("lab=", lab_tensor.shape)
    
        # Passer dans le transformer
        output_trans = self.transformer_decoder(lab_tensor)
        #print("out=", output_trans.shape)
    
        # Aplatir la sortie avant de passer dans la couche FC
        output_trans = output_trans.view(batch_size, -1)
        #print("output_trans after flattening=", output_trans.shape)

        # Appliquer la couche fully connected (assurez-vous que la sortie est correcte)
        output_trans = self.fc(output_trans)
        #print("output_trans after fc=", output_trans.shape)

        # Reshaper la sortie pour qu'elle corresponde à [batch_size, 3, 8, 8]
        output_trans = output_trans.view(batch_size, 256, 8, 8)
        #print("output_trans reshaped=", output_trans.shape)

        # Passer dans le quantizer
        z_q, vq_loss, _ = self.quantizer(output_trans)
    
        # Décodage final
        x_recon = self.decoder(z_q)
    
        return x_recon, vq_loss

class TokenDataset(torch.utils.data.Dataset):
    def __init__(self, token_seqs, block_size):
        self.data = token_seqs
        self.block_size = block_size

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]  # [seq_len]
        x = seq[:-1]#[:self.block_size]     # input
        y = seq[1:]#[:self.block_size]      # target (next token)
        return x, y


class TransTraining(nn.Module):
    def __init__(self, in_channels=3, z_channels=256, hidden_channels=128, out_channels=3, num_embeddings=512, commitment_cost=0.25):
        super().__init__()
        # D finir l'encodeur
        self.encoder = Encoder(in_channels, hidden_channels, z_channels)
        self.decoder = Decoder(in_channels, hidden_channels, z_channels)
        # Instancier le quantizer avec les arguments num_embeddings et commitment_cost
        self.quantizer = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=z_channels, commitment_cost=commitment_cost)
        self.quantizere = VectorQuantizer(num_embeddings=num_embeddings, embedding_dim=z_channels, commitment_cost=commitment_cost)
        
        # Instancier le transformateur d codeur
        self.transformer_decoder = 0#GPT(vocab_size, 64, 4, 1, 1)
        
        self.fc = nn.Sequential(
            nn.Embedding(294, 4096),
            nn.Linear(4096, 8192),
            nn.ReLU(),
            nn.Linear(8192, 256*8*8)
        )

        self.model_enc = VQGAN()
        self.model_enc.load_state_dict(torch.load('vqgan_model_car_newld32.pth', map_location=torch.device('cpu')))
        self.model_enc.eval()

    def forward(self, x, lab):

        

        return x

    def test_tokenize(self, x):

        #batch_size, ch, h, w = x.shape
        #lab_tensor = torch.tensor(lab, dtype=torch.long, device=x.device).view(batch_size, 1)  # [batch_size, 1]
        
        tokens = self.tokenize(x)         # [B, 64] pour 8x8
        print(tokens)
        # Generate image from tokens
        x_fake = self.detokenize(tokens, h=8, w=8)
        

        return x_fake

    def tokenize_images(self, dataloader, device):
        """
        Encode les images du dataloader en indices de codebook avec VQGAN,
        en affichant une barre de progression.
        """
        all_indices = []
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Tokenizing images"):
                if isinstance(batch, (list, tuple)):
                    x = batch[0]  # On prend les images
                else:
                    x = batch
                x = x.to(device)
                enc = self.model_enc.encoder(x)
                _, _, indices = self.model_enc.quantizer(enc)
                #print("shape indice=", indices.shape)
                all_indices.append(indices.view(indices.size(0), -1))  # [B, H*W]

        return torch.cat(all_indices, dim=0)  # [N, seq_len]

    
    def save_tokenizer_image(self, dataloader, device):

        # Sauvegarde sous forme de tensor unique
        token_sequences = self.tokenize_images(dataloader, device)  # [N, seq_len]

        # Exemple de chemin
        torch.save(token_sequences, 'token_image_dataset.pt')

        print('tokenizer image saved')

    def load_tokenizer_image(self, block_size=256):
        # Chargement
        token_sequences = torch.load('token_image_dataset.pt')

        # Recréation du dataset
        dataset = TokenDataset(token_sequences, block_size)

        # Recréation du DataLoader
        return torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)

    def extract_custom_neighbors(self, gtok, shp):
        
        windows = []

        y_pad = (shp) // 8
        x_pad = (shp) % 8

        neighbors = []

        # Ligne du dessus (y - 1)
        #print(x_pad)
        
        if x_pad == 0:
            neighbors.append(gtok[:, -8])
            neighbors.append(gtok[:, -7])
            neighbors.append(gtok[:, -6])
                           

        elif x_pad == 7:
            neighbors.append(gtok[:, -10])
            neighbors.append(gtok[:, -9])
            neighbors.append(gtok[:, -8])

            neighbors.append(gtok[:, -2])
            neighbors.append(gtok[:,  -1])
                        
        else:
            neighbors.append(gtok[:,-9])
            neighbors.append(gtok[:,-8])
            neighbors.append(gtok[:,-7])

            neighbors.append(gtok[:, -1])

        # Stack pour ce pixel : (B, nb_voisins)
        out = torch.stack(neighbors, dim=1)
      
        return out 

    def extract_custom_neighbors2(self, gtok, shp):
        
        windows = []

        y_pad = (shp) // 8
        x_pad = (shp) % 8

        neighbors = []

        # Ligne du dessus (y - 1)
        #print(x_pad)
        
        if x_pad == 0:
            neighbors.append(gtok[:, shp-8])
            neighbors.append(gtok[:, shp-7])
            neighbors.append(gtok[:, shp-6])
                           

        elif x_pad == 7:
            neighbors.append(gtok[:, shp-10])
            neighbors.append(gtok[:, shp-9])
            neighbors.append(gtok[:, shp-8])

            neighbors.append(gtok[:, shp-2])
            neighbors.append(gtok[:,  shp-1])
                        
        else:
            neighbors.append(gtok[:,shp-9])
            neighbors.append(gtok[:,shp-8])
            neighbors.append(gtok[:,shp-7])

            neighbors.append(gtok[:, shp-1])

        # Stack pour ce pixel : (B, nb_voisins)
        out = torch.stack(neighbors, dim=1)
      
        return out 

    def sample_next_token(self, logits, temperature=0.7, top_k=40):
        logits = logits[:, -1, :]  # On prend le dernier pas de temps
        if top_k is not None:
            values, _ = torch.topk(logits, top_k)
            min_values = values[:, -1, None]
            logits = torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)
        
        probs = torch.softmax(logits / temperature, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token


    def generate_large_image(self, model, start_label, x_imi,x_imi2, num_img, out_size=32, temperature=1.0, top_k=40, arg_max=False):
        model.eval()
        device = next(model.parameters()).device
        B = 1
        total_tokens = out_size * out_size
        generated = 0#x_imi[:, 0:16]#torch.full((B, 1), start_label, dtype=torch.long).to(device)  # condition tokens
        lab = torch.full((B, 4), start_label, dtype=torch.long).to(device) 
        start = 0
        end = 1
        count = 0
             
        while count < total_tokens:
            
            # On extrait les voisins pertinents pour guider la génération
            context = 0

            start = max(0, end - 15)
            input_seq =x_imi2[:,start:end]
            #print(input_seq.shape)

            coord = torch.arange(start, end).unsqueeze(0).repeat(B, 1).to(device)
            condv = num_img[:,:end-start]

            input_seq = input_seq.to(device)
            
            logits = model.get_ans42(input_seq, coord, condv)  # [B, T, vocab]

            end+=1
            #logits = logits[:, lab.shape[1]:, :]
            if not arg_max:
                next_token = self.sample_next_token(logits, temperature=temperature, top_k=top_k)
            else:
                next_token_logits = logits[:, -1, :] 
                next_token = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            #next_token_probs = torch.softmax(next_token_logits, dim=-1)
            
            #next_token = next_token_probs.multinomial(1)
            #input_seq = torch.cat([input_seq, next_token], dim=1)
            # Ajout du token généré
            if count == 0:generated = next_token 
            else: generated = torch.cat([generated, next_token], dim=1)

            count +=1

        image_tokens = generated  # on enlève la condition (si 4 tokens pour le label)
        image =  self.detokenize(image_tokens.view(1, out_size*out_size), out_size, out_size)
        return image

    def extract_custom_neighborsHD(self, gtok, i):
        """ gtok: [B, 256] - la séquence des tokens d'une image
            i: int - index linéaire du token dans la grille 16x16
        """
        B = gtok.shape[0]
        neighbors = []

        x = i % 128
        y = i // 128

        coords = []

        for dy in [-2, -1, 0, 1, 2]:
            stop = False
            for dx in [-2, -1, 0, 1, 2]:
                if dx == 0 and dy == 0:
                    stop = True
                    break  # on ne prend pas le centre

                nx = x + dx
                ny = y + dy

                if 0 <= nx < 128 and 0 <= ny < 128:
                    ni = ny * 128 + nx
                    coords.append(ni)
            if stop:break
        coords = torch.tensor(coords, device=gtok.device, dtype=torch.long)  # [nb_voisins]
        neighbors = gtok[:, coords]  # [B, nb_voisins]

        return neighbors  # [B, nb_voisins]

    # Reformer l’image 32x32 à partir des 4 patchs de 16x16
    def reorder_tokens(self, generated):
        B, L = generated.shape  # B x 1024
        assert L == 1024

        patches = []
        for i in range(4):  # 4 patchs
            patch = generated[:, i*256:(i+1)*256]  # B x 256
            patch = patch.view(B, 16, 16)          # B x 16 x 16
            patches.append(patch)

        # Recomposer les 2x2 patchs → image finale 32x32
        top = torch.cat([patches[0], patches[1]], dim=2)  # B x 16 x 32
        bottom = torch.cat([patches[2], patches[3]], dim=2)  # B x 16 x 32
        full_image = torch.cat([top, bottom], dim=1)  # B x 32 x 32

        return full_image.view(B, -1)  # tokens ordonnés spatialement

    def reorder_tokens128x128(self, generated):
        B, L = generated.shape  # B x 16384
        assert L == 16384

        patches = []
        for i in range(64):  # 64 patchs
            patch = generated[:, i*256:(i+1)*256]  # B x 256
            patch = patch.view(B, 16, 16)          # B x 16 x 16
            patches.append(patch)

        # Recomposer les 8x8 patchs → image finale 128x128
        rows = []
        for i in range(8):
            row = torch.cat(patches[i*8:(i+1)*8], dim=2)  # concaténer 8 patchs horizontalement → B x 16 x 128
            rows.append(row)

        full_image = torch.cat(rows, dim=1)  # concaténer les lignes verticalement → B x 128 x 128

        return full_image.view(B, -1)  # tokens ordonnés spatialement

    def generate_imageHD(self, model, x_imi,x_imi2, id_img, out_size=128, temperature=1.0, top_k=40, arg_max=False):

        model.eval()
        device = next(model.parameters()).device
        B = 1
        total_tokens = out_size * out_size
        generated = torch.zeros(B, total_tokens, dtype=torch.long).to(device)

        for i in range(32):
            generated[:, i*256:(i+1)*256] = x_imi[:, i*256:(i+1)*256]  # Premier patch de 64x64
        
        #generated[:, 256:512] = x_imi[:, 256:512]
        #generated[:, 512:768] = x_imi[:, 512:768]
        #generated[:, 768:1024] = x_imi[:, 768:1024]
                           
        generated = self.reorder_tokens128x128(generated)

        for i in range(8192, total_tokens):
            # On extrait les voisins déjà générés pour le token i
            #if i < 1:
            #    continue  # Pas encore de voisins, skip ou mettre un token spécial

            # Récupère les voisins du token i
            input_window = self.extract_custom_neighborsHD(generated, i)  # [B, nb_voisins]

            # Position actuelle pour ce token
            posx = i % 128
            posy = i // 128
            coord = torch.full((B, input_window.shape[1]), posx, device=device)  # [B, nb_voisins]
            coord2 = torch.full((B, input_window.shape[1]), posy, device=device)
            #num_img = id_img.unsqueeze(1).repeat(1, input_window.shape[1])    # [B, nb_voisins]

            with torch.no_grad():
                logits = model.get_ans42(input_window, coord, coord2)  # [B, nb_voisins, vocab]
                

                next_token = 0
                if not arg_max:
                    next_token = self.sample_next_token(logits, temperature=temperature, top_k=top_k)
                else:
                    logits = logits[:, -1, :]  # [B, vocab]
                    next_token = torch.argmax(logits, dim=-1).unsqueeze(-1)
                
                generated[:, i] = next_token


        out_size = 128
        image_tokens = generated  # on enlève la condition (si 4 tokens pour le label)
        image =  self.detokenize(image_tokens.view(1, out_size*out_size), out_size, out_size)
        return image


    def generate_image_tokens(self, model_gpt, start_tokens, total_tokens=1*64):
        """
        Génère des tokens pour 256 images 8x8 (chaque image = 64 tokens).
        Retourne un tenseur de forme [1, 256, 8, 8].
        """
        model_gpt.eval()
        device = next(model_gpt.parameters()).device
        generated_tokens = start_tokens.to(device)  # [1, seq_len]
        context_length = 8

        #generated_tokens2 = torch.zeros(1, 1, dtype=torch.long, device=device)

       
        with torch.no_grad():
        
            while generated_tokens.shape[1] < total_tokens:

                #print(generated_tokens.shape[1] , total_tokens)
                input_tokens = 0
                #if generated_tokens.shape[1] < context_length:
                #    input_tokens = F.pad(generated_tokens, (context_length - generated_tokens.shape[1], 0), value=0)
                #else:
                input_tokens = self.extract_custom_neighbors(generated_tokens)

                print(generated_tokens.shape[1], input_tokens)
                print(generated_tokens)
                #time.sleep(2)
                
                output = model_gpt(input_tokens)
                next_token = torch.softmax(output, dim=-1).multinomial(1)
                #next_token = torch.argmax(output, dim=-1).unsqueeze(-1)
                generated_tokens = torch.cat((generated_tokens, next_token), dim=1)
                #generated_tokens2 = torch.cat((generated_tokens2, next_token), dim=1)

        

        x_recon = self.detokenize(generated_tokens.view(1, 64), 8, 8)
        return x_recon

    def generate_from_label(self, model, label, max_len=3136, temperature=1.0):
        model.eval()
        device = next(model.parameters()).device

        cond = torch.full((1, 4), label, dtype=torch.long).to(device)
        generated = cond

        for _ in range(max_len):
            if generated.shape[1] > model.block_size:
                generated = generated[:, -model.block_size:]

            #print(generated)

            logits = model.get_ans4(generated)
            next_token_logits = logits[:, -1, :] / temperature
            next_token = torch.softmax(next_token_logits, dim=-1).multinomial(1)
            #next_token = torch.argmax(next_token_probs, dim=-1) 
            #next_token = next_token.unsqueeze(1)  
            generated = torch.cat([generated, next_token], dim=1)

            #print(generated)

        # on enlève les tokens de condition pour ne garder que l’image
        image_tokens = generated[:, cond.shape[1]:]
        print(image_tokens)
        # décodage image
        image =  self.detokenize(image_tokens.view(1, 3136), 56, 56)
        #image = vqgan.decode(image_tokens.view(1, 8, 8))  # reshape si 8x8
        return image


    def generate_image_tokensz(self, model_gpt, start_tokens, total_tokens=256*64):
        """
        Génère des tokens pour 256 images 8x8 (chaque image = 64 tokens).
        Retourne un tenseur de forme [1, 256, 8, 8].
        """
        model_gpt.eval()
        device = next(model_gpt.parameters()).device
        generated_tokens = start_tokens.to(device)  # [1, seq_len]
        context_length = 4

        with torch.no_grad():
            while generated_tokens.shape[1] < total_tokens:

                #print(generated_tokens.shape[1] , total_tokens)
                input_tokens = 0
                #if generated_tokens.shape[1] < context_length:
                #    input_tokens = F.pad(generated_tokens, (context_length - generated_tokens.shape[1], 0), value=0)
                #else:
                input_tokens = generated_tokens[:, -context_length:]

                
                output = model_gpt(input_tokens)
                next_token = torch.argmax(output, dim=-1).unsqueeze(-1)
                next_token.zero_()
                generated_tokens = torch.cat((generated_tokens, next_token), dim=1)


        x_recon = self.detokenize(generated_tokens.view(256, 64), 8, 8)
        return x_recon

    def generate_image_tokens_samp(self, model_gpt, start_tokens, total_tokens=256*64, temperature=1.0, top_k=None):
        """
        Génère des tokens pour 256 images 8x8 (chaque image = 64 tokens).
        Utilise du sampling stochastique (avec temperature et top-k).
        Retourne un tenseur de forme [1, 256, 8, 8].
        """
        model_gpt.eval()
        device = next(model_gpt.parameters()).device
        generated_tokens = start_tokens.to(device)  # [1, seq_len]
        context_length = 4

        with torch.no_grad():
            while generated_tokens.shape[1] < total_tokens:

                if generated_tokens.shape[1] < context_length:
                    input_tokens = F.pad(generated_tokens, (context_length - generated_tokens.shape[1], 0), value=0)
                else:
                    input_tokens = generated_tokens[:, -context_length:]

                logits = model_gpt(input_tokens)  # [1, vocab_size]
                logits = logits / temperature

                if top_k is not None:
                    # Garder les top_k meilleurs logits, masquer le reste
                    values, _ = torch.topk(logits, top_k)
                    min_values = values[:, -1].unsqueeze(1)
                    logits = torch.where(logits < min_values, torch.full_like(logits, float('-inf')), logits)

                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # [1, 1]
                #next_token.zero_()
                #print(next_token)
                generated_tokens = torch.cat((generated_tokens, next_token), dim=1)

        x_recon = self.detokenize(generated_tokens.view(256, 64), 8, 8)
        return x_recon


    def predict(self, x, lab):
        
        lab_tensor = torch.full((1, 3, 8, 8), lab, dtype=torch.float32)

        for i in range(64):
            # Suppose que z_q_with_token est identique à chaque itération ou généré à chaque fois
            o_trans = self.transformer_decoder(lab_tensor)  # (1, 3, 8, 8) si batch_size = 1
            lab_tensor = torch.cat((lab_tensor, o_trans), dim=1)

        x_recon = self.decoder(lab_tensor)
        return x_recon

    @torch.no_grad()
    def tokenize(self, image):
        """
        Convertit une image en séquence de tokens (indices).
        image: Tensor [B, 3, H, W]
        return: Tensor [B, h*w] (indices)
        """
        z = self.model_enc.encoder(image)                  # [B, C, h, w]
        _, _, indices = self.model_enc.quantizer(z)        # [B, h, w]
        return indices.view(indices.size(0), -1) # [B, h*w]

    @torch.no_grad()
    def detokenize(self, indices, h, w):
        """
        Reconvertit une séquence d'indices en image.
        indices: [B, h*w] ou [B, h, w]
        h, w: dimensions du patch latent (par ex. 8, 8)
        return: Tensor [B, 3, H, W]
        """
        device = next(self.model_enc.parameters()).device
        indices = indices.to(device)
        z_q = self.get_codebook_entries(indices, shape=(indices.size(0), h, w))  # [B, C, h, w]
        x_rec = self.model_enc.decoder(z_q)  # [B, 3, H, W]
        return x_rec

    def get_codebook_entries(self, indices, shape):
        """
        Récupère les vecteurs du codebook (embeddings) à partir des indices.
        indices: Tensor [B, h, w] ou [B, h*w]
        shape: (B, h, w)
        """
        if indices.dim() == 2:
            indices = indices.view(shape[0], shape[1], shape[2])  # [B, h, w]

        # Convertir [B, h, w] -> [B*h*w]
        flat_indices = indices.view(-1)

        # Extraire les embeddings correspondants [B*h*w, C]
        embeddings = self.model_enc.quantizer.embedding(flat_indices)  # [B*h*w, C]

        # Reshape en [B, C, h, w]
        z_q = embeddings.view(shape[0], shape[1], shape[2], -1)  # [B, h, w, C]
        z_q = z_q.permute(0, 3, 1, 2).contiguous()                # [B, C, h, w]

        return z_q




import torch
import torch.nn as nn

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
    def __init__(self, in_channels=3, z_channels=256, hidden_channels=128, out_channels=3, num_embeddings=512, commitment_cost=0.25):
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


def loss_fn(x, x_recon, vq_loss, perceptual_weight=0.2):
    # MSE classique
    recon_loss = F.mse_loss(x_recon, x)

    # LPIPS attend des images entre [-1, 1] → normaliser si nécessaire
    # Supposons que x et x_recon sont déjà dans [-1, 1]
    perceptual_loss = lpips_loss_fn(x_recon, x).mean()

    # Total loss
    total_loss = recon_loss + vq_loss + perceptual_weight * perceptual_loss
    return total_loss
    
def loss_fno(x, x_recon, vq_loss):
    recon_loss = F.mse_loss(x_recon, x)
    return recon_loss + vq_loss

def loss_fn2(x, x_recon, x_recone, vq_loss):
    recon_loss = F.mse_loss(x_recone, x)
    #recon_loss += F.mse_loss(x_recone, x_recon)
    return recon_loss + vq_loss

def prepare_for_lpips(img):
    # Assurez-vous que l’image est bien [3, H, W]
    if img.dim() == 3 and img.shape[0] != 3:
        img = img.permute(2, 0, 1)  # [H, W, 3] -> [3, H, W]
    elif img.dim() == 2:
        img = img.unsqueeze(0).repeat(3, 1, 1)  # grayscale -> RGB

    # Ajouter la dimension batch seulement si absente
    if img.dim() == 3:
        img = img.unsqueeze(0)  # [1, 3, H, W]

    # Convertir en float32 et normaliser entre -1 et 1
    img = img.float() / 127.5 - 1.0

    return img.to(device)

import torch.nn.functional as F

def loss_fn_vqgan_adv(x, x_recon, vq_loss, D, perceptual_weight=0.2, adv_weight=1.0):
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
def show_reconstructed_images(data, x_recon, save=False):
    global INDEX
    mean = torch.tensor([0.485, 0.456, 0.406],  device=data.device)  # Convertir mean en Tensor
    std = torch.tensor([0.229, 0.224, 0.225], device=data.device)   # Convertir std en Tensor

    # Cr er un graphique avec deux images c te   c te
    fig, ax = plt.subplots(1, 2, figsize=(5, 2.5))

    image_standardized = data[0].permute(1, 2, 0).cpu().numpy()
    image_denormalized = (image_standardized+1)/2 #* std.cpu().numpy() + mean.cpu().numpy()
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
    if save:
        plt.savefig('reconstructed_image' + str(INDEX) + '.png', bbox_inches='tight', pad_inches=0.1)
    # Afficher le graphique
    plt.tight_layout()
    #plt.show()
    INDEX+=1

cifar10_classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

#Train Token

# Param tres d'entra nement
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = 8
epochs = 20
learning_rate = 1e-4

# Chargement du dataset CIFAR-10 pour tester (tu peux utiliser ton propre dataset)
transform = transforms.Compose([
    transforms.Resize((64, 64)),    # Redimensionner les images à 224x224 pixels
    transforms.ToTensor(),            # Convertir les images en tensors
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalisation (pour un modèle préentraîné comme ResNet)
])

# charge l'image
img = Image.open("headbl.jpg").convert("RGB")
image_test = transform(img).to(device)  # (1, 3, 32, 32)

img = Image.open("headbl.jpg").convert("RGB")
image_pos = transform(img).to(device)

#dataset_path = './dataset'  # Remplacez par le chemin réel de votre dataset
#dataset = datasets.ImageFolder(root=dataset_path, transform=transform)
#batch_size = 1  # Vous pouvez ajuster le batch size selon votre besoin
#train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

#image, label = dataset[0]  # Récupérer la première image et son étiquette
#print("Image:", image.shape)  # Afficher la forme de l'image
#print("Label:", label)  # Afficher l'étiquette (classe)
#image = image.permute(1, 2, 0).numpy()  # Convertir de (C, H, W) à (H, W, C)
# Si c’est une image RGB (3 canaux), permuter pour matplotlib
def affim(image):
    if image.ndim == 3 and image.shape[0] == 3:
        image = image.permute(1, 2, 0)  # (H, W, C)
    
    # Si l’image est normalisée [-1, 1], on la remet en [0, 1]
    image = (image + 1) / 2
    image = image.clamp(0, 1)  # pour éviter les valeurs hors bornes
    
    # Convertir en numpy pour plt
    plt.imshow(image.numpy())
    plt.title(f"Label: {label}")
    plt.axis('off')
    plt.show()

# transforme l’image et encode avec VQGAN
def calc_indice(image):
    vqgam = VQGAN().to(device)
    vqgam.load_state_dict(torch.load('vqgan_model_car_newld32.pth', map_location=torch.device('cpu')))
    vqgam.eval()
    image = image.to(next(vqgam.parameters()).device).unsqueeze(0)  # [1, 3, 32, 32]
    with torch.no_grad():
        z = vqgam.encoder(image)                   # [1, C, H, W]
        z_q, _, indices = vqgam.quantizer(z)       # indices: [1, H, W]
        indices = indices.view(-1).unsqueeze(0).long()             # [H * W], e.g. 8x8 = 64 tokens
    
    return indices

def calc_indice2(image):
    vqgam = VQGAN().to(device)
    vqgam.load_state_dict(torch.load('vqgan_model_car_newld32.pth', map_location=torch.device('cpu')))
    vqgam.eval()
    patches = []
    patch_size = 32
    side = 64
    # Découpe l'image 256x256 en 4x4 = 16 patchs de 64x64
    for i in range(0, side, patch_size):
        for j in range(0, side, patch_size):
            patch = image[:, i:i+patch_size, j:j+patch_size]  # shape: [3, 64, 64]
            patches.append(patch)
    
    all_indices = []
     
    # Passe chaque patch dans le VQGAN
    for patch in patches:
        patch = patch.to(next(vqgam.parameters()).device).unsqueeze(0)  # [1, 3, 64, 64]
        with torch.no_grad():
            z = vqgam.encoder(patch)                    # [1, C, H, W]
            z_q, _, indices = vqgam.quantizer(z)        # [1, H, W]
            indices = indices.view(-1).long()                # [H * W]
            all_indices.append(indices)
    
    # Concaténer tous les indices des 16 patchs
    all_indices = torch.cat(all_indices, dim=0)  # [16 * H*W]
    
    return all_indices.expand(1, -1).clone()

#image = calc_indice(image)
image_test = calc_indice2(image_test)
image_pos = calc_indice2(image_pos)

#lpips_loss_fn = lpips.LPIPS(net='vgg').to(device)  # ou 'alex', 'squeeze'
#lpips_loss_fn.eval() 

# Afficher l'image
#plt.imshow(image)
#plt.title(f'Label: {label}')
#plt.axis('off')  # Désactiver les axes
#plt.show()

embed_dim = 128  # Pour suivre la configuration de BERT
num_heads = 4 # Par défaut dans BERT
num_layers = 2  # Par défaut dans BERT
block_size = 64

# Instancier le mod le et l'optimiseur
model = GPT(1024, embed_dim, num_heads, num_layers, block_size).to(device)
model.load_state_dict(torch.load('vqgan_model_tr_carld64_16x16_2.pth', map_location=torch.device('cpu')))
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.1)

def reconstruct_image_from_patches(patches, grid_size=4):
    """
    patches: Tensor de forme [16, 1, 64, 64]
    grid_size: 4 → 4x4 patches (par défaut)
    
    Retourne: Tensor [1, 256, 256]
    """
    #assert patches.shape[0] == grid_size ** 2
    patch_size = patches.shape[-1]
    
    rows = []
    for i in range(grid_size):
        row = torch.cat([patches[i * grid_size + j] for j in range(grid_size)], dim=-1)  # concatène horizontalement
        rows.append(row)
    full_image = torch.cat(rows, dim=-2)  # concatène verticalement
    
    return full_image  # [1, 256, 256]



def image_generation_label2(model):
    
    vqgan = VQGAN()
    vqgan.load_state_dict(torch.load('/kaggle/working/vqgan_model_car_new.pth'))

    dataloader = load_cat_tokens(vqgan, batch_size=1)


    iterator = iter(dataloader)
    all_batches = [next(iterator) for _ in range(10)]
    print('start1')   
    
    trans_train = TransTraining()
    lab = [i for i in range(2)]
    for l in lab:
        print(l)
        temp = 1.0
        for i in range(1):
            batch = all_batches[random.randint(0, len(all_batches) - 1)] 
            cond, target = batch
            #x_imi = torch.cat([cond, target], dim=1)
            
            #x_imi = x_imi#[:, :3136//2+4]
            print('start=', cond[0][0].item())
            x_images = []
            patch_token_count = 256
            for i in range(16):
                # Indices des tokens du patch
                start = i * patch_token_count
                end = start + patch_token_count
                
                # Tu prends seulement la moitié pour le conditionnement
                x_imi = target[:, start : start + 128]  # ← 8 tokens
    
                x_image = trans_train.generate_large_image(model, cond[0][0].item(), x_imi, temperature=1.0)
                x_images.append(x_image)  # ← attention à la typo ici : "appens" → "append"
            
            # Convertit la liste en tensor : [16, 1, 16, 16]
            x_images_tensor = torch.stack(x_images)
            x_imf = reconstruct_image_from_patches(x_images_tensor)

            out_size = 16
            #x_image_o =  trans_train.detokenize(target.view(1, out_size*out_size), out_size, out_size)
            show_reconstructed_images(x_imf, x_imf)
            
def image_generation_label(model, x_imi, x_imit):
    
    vqgan = VQGAN()
    vqgan.load_state_dict(torch.load('vqgan_model_car_newld32.pth', map_location=torch.device('cpu')))

    dataloader = load_cat_tokens(vqgan, batch_size=1)


    iterator = iter(dataloader)
    #rd = random.randint(0, 20)
    #for _ in range(rd):
    #    next(iterator)

    all_batches = next(iterator)
    #batch = all_batches[0] 
    #cond, target = batch
    #x_imi = x_imi.expand(1, -1).clone()
    #x_imi = x_imi.to(device)
    print('start1')   
    
    trans_train = TransTraining()
    lab = [i for i in range(1)]
    for l in lab:
        print(l)
       
        batch = all_batches#[random.randint(0, len(all_batches) - 1)] 
        cond, target, id = batch
        x_imi2 = target.to(device)
        idx = id.to(device)
        num_img = idx.unsqueeze(1).repeat(1, 15).to(device)
        temp = -0.1
        topk = 100
        
        while(temp < 1.2):
            #for topk in range(100,60,20):
            #batch = all_batches[random.randint(0, len(all_batches) - 1)] 
            #cond, target = batch
            #x_imi = target
            
            #torch.cat([cond, target], dim=1)
            #x_imi = target.to(device)
            arg_max = False
            if temp < 0.0:
                arg_max = True
            #print('start')
            x_imi22 = x_imit.to(device)
            x_image = trans_train.generate_large_image(model, cond[0][0].item(), x_imi,x_imi22,num_img, temperature=temp, top_k=topk, arg_max=arg_max)

            x_imi22 = x_imit.to(device)
            x_image2 = trans_train.generate_large_image(model, cond[0][0].item(), x_imi,x_imi22,num_img, temperature=temp, top_k=topk, arg_max=arg_max)
    
            out_size = 32
            x_image_o =  trans_train.detokenize(target.view(1, out_size*out_size), out_size, out_size)
    
            print("temp=", temp, ", topk=", topk)
            show_reconstructed_images(x_image_o, x_image, save=True)
            show_reconstructed_images(x_image, x_image2, save=True)

            temp += 0.2
            
def reorder_tokens(generated):
    B, L = generated.shape  # B x 1024
    assert L == 1024

    patches = []
    for i in range(4):  # 4 patchs
        patch = generated[:, i*256:(i+1)*256]  # B x 256
        patch = patch.view(B, 16, 16)          # B x 16 x 16
        patches.append(patch)

    # Recomposer les 2x2 patchs → image finale 32x32
    top = torch.cat([patches[0], patches[1]], dim=2)  # B x 16 x 32
    bottom = torch.cat([patches[2], patches[3]], dim=2)  # B x 16 x 32
    full_image = torch.cat([top, bottom], dim=1)  # B x 32 x 32

    return full_image.view(B, -1)  # tokens ordonnés spatialement

def reorder_tokens128x128(generated):
    B, L = generated.shape  # B x 16384
    assert L == 16384

    patches = []
    for i in range(64):  # 64 patchs
        patch = generated[:, i*256:(i+1)*256]  # B x 256
        patch = patch.view(B, 16, 16)          # B x 16 x 16
        patches.append(patch)

    # Recomposer les 8x8 patchs → image finale 128x128
    rows = []
    for i in range(8):
        row = torch.cat(patches[i*8:(i+1)*8], dim=2)  # concaténer 8 patchs horizontalement → B x 16 x 128
        rows.append(row)

    full_image = torch.cat(rows, dim=1)  # concaténer les lignes verticalement → B x 128 x 128

    return full_image.view(B, -1)  # tokens ordonnés spatialement

def image_generation_labelP(model, x_imi, x_imit):
    
    vqgan = VQGAN()
    vqgan.load_state_dict(torch.load('vqgan_model_car_newld32.pth', map_location=torch.device('cpu')))

    dataloader = load_cat_tokens(vqgan, batch_size=1)


    iterator = iter(dataloader)
    #rd = random.randint(0, 20)
    #for _ in range(rd):
    #    next(iterator)

    all_batches = next(iterator)
    #batch = all_batches[0] 
    #cond, target = batch
    #x_imi = x_imi.expand(1, -1).clone()
    #x_imi = x_imi.to(device)
    print('start1')   
    
    trans_train = TransTraining()
    lab = [i for i in range(1)]
    for l in lab:
        print(l)
       
        batch = all_batches#[random.randint(0, len(all_batches) - 1)] 
        cond, target, id = batch
        x_imi2 = target.to(device)
        idx = id.to(device)
        num_img = idx
        temp = -0.1
        topk = 40
        
        while(temp < 1.2):
            #for topk in range(100,60,20):
            #batch = all_batches[random.randint(0, len(all_batches) - 1)] 
            #cond, target = batch
            #x_imi = target

            
            
            #torch.cat([cond, target], dim=1)
            #x_imi = target.to(device)
            arg_max = False
            if temp < 0.0:
                arg_max = True
            #print('start')
            x_imi22 = x_imi2.to(device)
            x_image = trans_train.generate_imageHD(model, x_imi22,x_imi22,num_img, temperature=temp, top_k=topk, arg_max=arg_max)

            x_imi22 = x_imi2.to(device)
            x_image2 = trans_train.generate_imageHD(model, x_imi22,x_imi22,num_img, temperature=temp, top_k=topk, arg_max=arg_max)
    
            x_image_o  =x_imi2.to(device)
            out_size = 128
            #x_image_o[:, 0:256] = x_imi2[:,0:256]
            #x_image_o[:, 256:512] = x_imi2[:,256:512] 
            #x_image_o[:, 512:768] = x_imi2[:,512:768] 
            #x_image_o[:, 768:1024] = x_imi2[:,768:1024] 

            x_image_o = reorder_tokens128x128(x_image_o)

            x_image_o =  trans_train.detokenize(x_image_o.view(1, out_size*out_size), out_size, out_size)
            print("temp=", temp, ", topk=", topk)
            show_reconstructed_images(x_image_o, x_image, save=True)
            show_reconstructed_images(x_image, x_image2, save=True)

            temp += 0.2

def image_generation(model, datal, device):
    trans_train = TransTraining()
    model.eval()

    dataloader = trans_train.load_tokenizer_image(16)
    

    for i in range(20):

        iterator = iter(dataloader)
        batch = random.choice([next(iterator) for _ in range(len(dataloader))])
        #all_batches = [next(iterator) for _ in range(len(dataloader))]
        #index = random.randint(0, len(all_batches) - 1)
        #batch = all_batches[index] 
        
        x, y = batch #next(iter(dataloader))  # x : [batch, 32]
        print(x.shape)
        start_tokens = x.view(1, -1)[0:1, :8].to(device)
        #start_tokens2 = x.view(1, -1)[0:1, :4].to(device)
        #print(start_tokens.shape)
        #for j in range(0, 60):
        #    start_tokens[:,j] = 0#random.randint(0, 511)
        #    start_tokens2[:,j] = 0#random.randint(0, 511)
        
        x_image = trans_train.generate_image_tokens(model, start_tokens)
        
        #x_image_s = trans_train.generate_image_tokensz(model, start_tokens2)

               
        show_reconstructed_images(x_image, x_image)



def transtraintoken(model, dataloader, optimizer, device, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        index = 0
        loop = tqdm(dataloader, desc=f'Epoch {epoch + 1}', leave=True)
        for (data, lab) in loop:
            data = data.to(device)
         
       
            x_recon = model.test_tokenize(data)

            if index % 500 == 0:
                show_reconstructed_images(data, x_recon)
                #time.sleep(10)


            index += 1

            #if index % 100 == 0:
            #    print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

        
       
        print(f"Epoch [{epoch+1}/{epochs}], Total Loss: {total_loss/len(dataloader):.4f}")
    
    print("Training complete!")

def transtrain_ltoi60(model, optimizer, device, scheduler, epochs=20):
    vqgan = VQGAN()
    vqgan.load_state_dict(torch.load('vqgan_model_cat.pth'))
    
    dataloader = load_cat_tokens(vqgan, batch_size=8)
    model.train()
    chunk_size = 56

    for epoch in range(epochs):
        total_loss = 0
        loop = tqdm(dataloader, desc=f'Epoch {epoch + 1}')
         
        for cond, target in loop:
            cond = cond.to(device)         # [B, Lc]
            target = target.to(device)     # [B, Ls]
            B, L = target.shape

            # On découpe avec stride = chunk_size, en gardant le dernier morceau (paddé si besoin)
            s_inputs = []
            s_targets = []

            for i in range(0, L - 1, chunk_size):
                s_input = target[:, i : i + chunk_size - 1]   # [B, ?]
                s_target = target[:, i + 1 : i + chunk_size]  # [B, ?]

                # Padding si on est en fin de séquence
                if s_input.shape[1] < chunk_size - 1:
                    pad_len = chunk_size - 1 - s_input.shape[1]
                    s_input = torch.nn.functional.pad(s_input, (0, pad_len), value=0)
                    s_target = torch.nn.functional.pad(s_target, (0, pad_len), value=0)

                s_inputs.append(s_input)
                s_targets.append(s_target)

            ind = 0
            # Boucle sur les sous-séquences
            for s_input, s_target in zip(s_inputs, s_targets):
                #print("s_input.shape =", s_input.shape)
                #print("s_target.shape =", s_target.shape)
                # → input_seq: entrée du modèle
                # → s_target: cible
           
                # Concatène label + début de la cible
                if ind == 0:
                    input_seq = torch.cat([cond, s_input], dim=1)
                else:
                    input_seq = s_input

                

                #attention_mask = (input_seq != 0).long()       # [B, L]
                #attention_mask[:, :4] = 1

                logits = model.get_ans4(input_seq)      # [B, Lc + Ls - 1, vocab]
                csh = 4
                if ind > 0:
                    csh = 0

                ind += 1
                logits_s = logits[:, csh:, :]  # on ne garde que la sortie image
                       

                min_len = min(logits_s.shape[1], s_target.shape[1])
                logits_s = logits_s[:, :min_len, :]
                s_target = s_target[:, :min_len]

                loss = F.cross_entropy(logits_s.reshape(-1, logits_s.size(-1)), s_target.reshape(-1), ignore_index=0 )
                total_loss += loss.item()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                loop.set_postfix(loss=loss.item())

        scheduler.step(total_loss/56.0 / len(dataloader))
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")
        torch.save(model.state_dict(), f'transformer_label_to_image_cat_60.pth')

    print("Training complete!")

def transtrain_ltoi(model, optimizer, device, scheduler, epochs=20):
    vqgan = VQGAN()
    vqgan.load_state_dict(torch.load('/kaggle/input/vqgan_model_cat/pytorch/default/1/vqgan_model_cat.pth'))
    
    dataloader = load_cat_tokens(vqgan, batch_size=8)
    model.train()

    

    for epoch in range(epochs):
        total_loss = 0
        loop = tqdm(dataloader, desc=f'Epoch {epoch + 1}')

        for cond, target in loop:
            cond = cond.to(device)         # [B, Lc]
            target = target.to(device)     # [B, Ls]

            s_input = target[:, :-1]
            s_target = target[:, 1:]

            # Concatène label + début de la cible
            input_seq = torch.cat([cond, s_input], dim=1)

            logits = model.get_ans4(input_seq)      # [B, Lc + Ls - 1, vocab]
            logits_s = logits[:, cond.shape[1]:, :]  # on ne garde que la sortie image

            loss = F.cross_entropy(logits_s.reshape(-1, logits_s.size(-1)), s_target.reshape(-1))
            total_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_postfix(loss=loss.item())

        scheduler.step(total_loss / len(dataloader))
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {total_loss / len(dataloader):.4f}")
        torch.save(model.state_dict(), f'/kaggle/working/transformer_label_to_image_cat3136.pth')

    print("Training complete!")

def transformer_loss_fn(output_logits, target_tokens, x_recon, x_original, lpips_model, alpha=1.0):
    # Cross-entropy sur les tokens
    ce_loss = F.cross_entropy(output_logits.view(-1, output_logits.size(-1)), target_tokens.view(-1))

    # LPIPS sur les images reconstruites
    x_recon = prepare_for_lpips(x_recon)      # (1, 3, H, W), [-1,1]
    x_original = prepare_for_lpips(x_original)

    lpips_score = lpips_model(x_recon, x_original)

    # Combinaison des deux
    loss = ce_loss + alpha * lpips_score
    return loss


def transtrain(model, optimizer, device, scheduler, x_imi, x_imip, epochs=20):
    trans_train = TransTraining()
    vqgan = VQGAN()
    vqgan.load_state_dict(torch.load('vqgan_model_car_newld32.pth', map_location=torch.device('cpu')))
    
    dataloader = load_cat_tokens(vqgan, batch_size=8)

    iterator = iter(dataloader)
    #all_batches = [next(iterator) for _ in range(10)]
    
                                
    model.train()
    lpips_loss = 0
    for epoch in range(epochs):
        total_loss = 0
        index = 0
        loop = tqdm(dataloader, desc=f'Epoch {epoch + 1}', leave=True)

        x_decoded_tokens = 0
        
        for cond, targets, id in loop:

            cond = cond.to(device)         # [B, Lc]
            targets = targets.to(device)   # [B, Ls]
            idx  = id.to(device)
            B, D = targets.shape
            print("shape=", targets.shape)
            
            # Le code latent est copié pour chaque séquence du batch
            x_image = x_imip.expand(B, -1).clone()
            
            context_len = 16
            stride = 1
            seq_len = targets.shape[1]
            loss_total = 0.0
            count = 0

                        
            for i in range(0, seq_len - context_len + 1, stride):
                #input_window = torch.cat([
                #    x_image[:, i:i+context_len],         # image info
                #    targets[:, i:i+context_len-1]        # tokens sauf le dernier
                #], dim=-1)  # [B, 2*context_len - 1]
                input_window = x_image[:, i:i+context_len-1]
                target_window = targets[:, i:i+context_len]  # [B, context_len]

                #mask = x_image > 256  # Utilise l'image conditionnelle pour le masque
                #random_tokens = torch.randint(0, 512, x_image.shape).to(device)  # Générer des tokens aléatoires
    
                # Modifier l'image d'entrée en fonction de l'image conditionnelle
                #modified_x_image = x_image.clone()
                #modified_x_image[mask] = random_tokens[mask]  # Appliquer les modifications de l'image conditionnelle à l'image d'entrée
                coord = torch.arange(i, i + context_len - 1).unsqueeze(0).repeat(B, 1).to(device)
          
                num_img = idx.unsqueeze(1).repeat(1, context_len - 1).to(device) #x_imip[:, i:i+context_len-1] 

                # Forward
                out = model.get_ans42(input_window, coord, num_img)  # [B, context_len, vocab_size]
                out_last = out[:, -1, :]            # [B, vocab_size]
                target_last = target_window[:, -1]  # [B]
                            
                # Loss
                loss = F.cross_entropy(out_last, target_last)
                    
                loss_total += loss.item()
                count += 1
            
                # Backward & step (si tu veux faire pas à pas)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
            
            loss_avg = loss_total / count
                                   
            total_loss += loss_avg

            # Backward pass et optimisation
            
            #for name, param in model.named_parameters():
            #    if "fc" in name and param.grad is not None:
            #        print(f"{name} grad norm = {param.grad.norm()}")
          
            loop.set_postfix(loss=loss_avg)
                

            index += 1
            #if index % 3000 == 0:
            #    time.sleep(5)
            torch.cuda.empty_cache()
        
        
        scheduler.step(total_loss/len(dataloader))
        print(f"Epoch [{epoch+1}/{epochs}], Total Loss: {total_loss/len(dataloader):.4f}")
        torch.save(model.state_dict(), 'vqgan_model_tr_carld32.pth')
    
    print("Training complete!")


def extract_custom_neighbors(gtok, i):
    """ gtok: [B, 256] - la séquence des tokens d'une image
        i: int - index linéaire du token dans la grille 16x16
    """
    B = gtok.shape[0]
    neighbors = []

    x = i % 16
    y = i // 16

    coords = []

    for dy in [-2,-1, 0, 1, 2]:
        stop = False
        for dx in [-2, -1, 0, 1, 2]:
            if dx == 0 and dy == 0:
                stop = True
                break  # on ne prend pas le centre

            nx = x + dx
            ny = y + dy

            if 0 <= nx < 16 and 0 <= ny < 16:
                ni = ny * 16 + nx
                coords.append(ni)
        if stop:break
    coords = torch.tensor(coords, device=gtok.device)  # [nb_voisins]
    neighbors = gtok[:, coords]  # [B, nb_voisins]

    return neighbors  # [B, nb_voisins]


def transtrainP(model, optimizer, device, scheduler, x_imi, x_imip, epochs=20):
    trans_train = TransTraining()
    vqgan = VQGAN()
    vqgan.load_state_dict(torch.load('vqgan_model_car_newld32.pth', map_location=torch.device('cpu')))
    
    dataloader = load_cat_tokens(vqgan, batch_size=8)

    iterator = iter(dataloader)
    #all_batches = [next(iterator) for _ in range(10)]
    
                                
    model.train()
    lpips_loss = 0
    for epoch in range(epochs):
        total_loss = 0
        loop = tqdm(dataloader, desc=f'Epoch {epoch + 1}', leave=True)
                    
        for _cond, _targets, _id in loop:
            index = 0
            loss_avg = 0.0

            while(index < 64):
                pos = index * 256
                #cond = _cond[:,pos:pos+256].to(device)         # [B, Lc]
                targets = _targets[:,pos:pos+256].to(device)   # [B, Ls]
                idx  = _id.to(device)
                #print("idx=", idx)
                B, D = targets.shape[0], 256
                #print("shape=", targets.shape)
            
                # Le code latent est copié pour chaque séquence du batch
                x_image = x_imip[:,pos:pos+256].expand(B, -1).clone()
            
                context_len = 16
                stride = 1
                seq_len = 256#targets.shape[1]
                loss_total = 0.0
                count = 0

                        
                for i in range(16, 256):
                    input_window = extract_custom_neighbors(targets,i)
                    target_last = targets[:, i]  # [B, context_len]

                    # Recalage spatial dans la grille 32x32
                    # (x_patch, y_patch) indique la position du patch dans la 2x2 grille
                    x_patch = (index % 8) * 16
                    y_patch = (index // 8) * 16

                    # Coordonnée dans le patch 16x16
                    x_local = i % 16
                    y_local = i // 16

                    # Coordonnée absolue dans l'image 32x32
                    x = x_patch + x_local
                    y = y_patch + y_local
                    pos_abs = y * 128 + x  # position linéaire dans la grille 32x32

                    coord = torch.full((B, input_window.shape[1]), x, device=device)
                    coord2 = torch.full((B, input_window.shape[1]), y, device=device)
                    #if idx.dim() == 1:
                    #    idx = idx.unsqueeze(1)
                    #num_img = idx.repeat(1, input_window.shape[1])  # [B, nb_voisins]

                    # Forward
                    out = model.get_ans42(input_window, coord, coord2)  # [B, context_len, vocab_size]
                    out_last = out[:, -1, :]            # [B, vocab_size]
                                             
                    # Loss
                    loss = F.cross_entropy(out_last, target_last)
                    
                    loss_total += loss.item()
                    count += 1
            
                    # Backward & step (si tu veux faire pas à pas)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                
            
                loss_avg += loss_total / count
                                   
                
                          
                loop.set_postfix(loss=loss_avg/(index+1))
                

                index += 1
                torch.cuda.empty_cache()
                                            
        total_loss += loss_avg / 64
        scheduler.step(total_loss/len(dataloader))
        print(f"Epoch [{epoch+1}/{epochs}], Total Loss: {total_loss/len(dataloader):.4f}")
        torch.save(model.state_dict(), 'vqgan_model_tr_carld64_16x16_2.pth')
    
    print("Training complete!")

# Fonction d'entra nement
def train(model, dataloader, optimizer, device, epochs=20):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        index = 0
        for batch_idx, (data, lab) in enumerate(dataloader):
            data = data.to(device)
            optimizer.zero_grad()

            # Forward pass
            lab_tokens = [tokenizer.tokens[cifar10_classes[lab.item()]] for lab in lab]
            x_recon, vq_loss = model(data, lab_tokens)

            if index % 500 == 0:
                show_reconstructed_images(data, x_recon)

            # Calcul de la perte
            loss = loss_fn(data, x_recon, vq_loss)
            total_loss += loss.item()

            # Backward pass et optimisation
            loss.backward()
            optimizer.step()

            index += 1

            if index % 100 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Step [{batch_idx}/{len(dataloader)}], Loss: {loss.item():.4f}")

        
        scheduler.step(total_loss/len(dataloader))
        print(f"Epoch [{epoch+1}/{epochs}], Total Loss: {total_loss/len(dataloader):.4f}")
    
    print("Training complete!")

def eval(filename, model, dataloader, device, epochs=20):
    model.eval()  # Passer en mode évaluation (désactive les Dropout et BatchNorm si présents)
    model.load_state_dict(torch.load(filename))

    # Désactivation de l'optimiseur car l'évaluation ne modifie pas les poids
    with torch.no_grad():  # Désactive la création de gradients pour économiser de la mémoire
        
        index = 0
        for batch_idx, (data, lab) in enumerate(dataloader):
            data = data.to(device)

            # Forward pass
            lab_tokens = [tokenizer.tokens[cifar10_classes[lab.item()]] for lab in lab]
            x_recon = model.predict(data, lab_tokens)

            # Afficher les images reconstruites périodiquement
            show_reconstructed_images(data, x_recon)
                        
            index += 1
            if index > epochs:
                break
                
           

    print("Evaluation complete!")

#model.load_state_dict(torch.load('vqgan_model_tr_carld64_16x16.pth', map_location=torch.device('cpu')))
image_generation_labelP(model, image_test, image_pos)
#model.load_state_dict(torch.load('vqgan_model_tr_one.pth'))
#image_generation(model, train_loader, device)

#TransTraining().save_tokenizer_image(train_loader, device)

# Entra ner le mod le
#transtraintoken(TransTraining(), train_loader, optimizer, device, epochs)
#transtrainP(model, optimizer, device, scheduler, image_test, image_pos,  200)
#train(model, train_loader, optimizer, device, epochs)
#eval('vqgan_model2.pth', model, train_loader, device, 10)

#torch.save(model.state_dict(), 'vqgan_modelg_trans.pth')
#torch.save(model.state_dict(), 'vqgan_model_tr_one.pth')