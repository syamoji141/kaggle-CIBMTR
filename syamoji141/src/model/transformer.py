import torch
import torch.nn as nn


class TableTransformer(nn.Module):
    def __init__(
        self,
        cat_sizes,      
        cat_emb_dims,   
        num_features,   
        d_model=32,
        nhead=4,
        num_layers=2,
        dropout=0.1,
        use_batch_first=True,
        multi_output=True,
    ):
        super().__init__()

        self.num_linears = nn.ModuleList([
            nn.Linear(in_features=1, out_features=d_model)
            for _ in range(num_features)
        ])

        self.cat_embeddings = nn.ModuleList()
        self.cat_emb_to_dmodel = nn.ModuleList()
        for size, emb_dim in zip(cat_sizes, cat_emb_dims):
            emb = nn.Embedding(num_embeddings=size, embedding_dim=emb_dim)
            self.cat_embeddings.append(emb)
            proj = nn.Linear(emb_dim, d_model)
            self.cat_emb_to_dmodel.append(proj)

        self.num_token = num_features + len(cat_sizes)
        self.column_embedding = nn.Embedding(self.num_token, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=use_batch_first
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.output_layer = nn.Linear(d_model, 1)
        self.multi_output = multi_output
        if self.multi_output:
            self.output_layer_v2 = nn.Linear(d_model, 1)


    def forward(self, x_cat, x_num):
        batch_size = x_cat.size(0)

        num_embs = []
        for i, linear in enumerate(self.num_linears):
            col_i = x_num[:, i].unsqueeze(-1)
            out = linear(col_i) 
            num_embs.append(out)

        cat_embs = []
        for i, (emb_layer, proj_layer) in enumerate(zip(self.cat_embeddings, self.cat_emb_to_dmodel)):
            emb_out = emb_layer(x_cat[:, i])
            emb_out = proj_layer(emb_out)
            cat_embs.append(emb_out)

        x = torch.stack(num_embs + cat_embs, dim=1) 

        device = x.device
        col_ids = torch.arange(self.num_token, device=device)
        col_emb = self.column_embedding(col_ids)
        col_emb = col_emb.unsqueeze(0).expand(batch_size, -1, -1)
        x = x + col_emb
        encoded = self.transformer_encoder(x)

        pooled = encoded.mean(dim=1)
        out = self.output_layer(pooled)
        if self.multi_output:
            out_efs_time = self.output_layer_v2(pooled)
            return out, out_efs_time
        else:
            return out