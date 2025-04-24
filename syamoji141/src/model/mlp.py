import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, cat_size, cat_emb, num_features) -> None:
        super().__init__()
        self.embeddings = nn.ModuleList([
            nn.Embedding(num_embeddings=cat_size[i], embedding_dim=cat_emb[i])
            for i in range(len(cat_size))
        ])

        total_input_dim = sum(cat_emb) + num_features

        self.fc1 = nn.Linear(total_input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x_cat, x_num):

        embs = []
        for i, emb_layer in enumerate(self.embeddings):
            emb_out = emb_layer(x_cat[:, i])
            embs.append(emb_out)

        x = torch.cat(embs + [x_num], dim=1)  # dim=1 で列方向に結合

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        x = self.fc3(x)
        return x