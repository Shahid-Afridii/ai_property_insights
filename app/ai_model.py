import torch
import torch.nn as nn

class PropertyModel(nn.Module):
    def __init__(self, num_postcodes, num_types, num_towns):
        super().__init__()

        self.pc_emb = nn.Embedding(num_postcodes, 32)
        self.type_emb = nn.Embedding(num_types, 8)
        self.town_emb = nn.Embedding(num_towns, 16)

        self.lstm = nn.LSTM(1, 64, 2, batch_first=True)

        self.fc = nn.Sequential(
            nn.Linear(64 + 32 + 8 + 16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(256, 128),
            nn.ReLU(),

            nn.Linear(128, 1)
        )

    def forward(self, price_seq, pc, typ, town):
        lstm_out, _ = self.lstm(price_seq)
        lstm_feat = lstm_out[:, -1, :]

        pc_feat = self.pc_emb(pc)
        type_feat = self.type_emb(typ)
        town_feat = self.town_emb(town)

        x = torch.cat([lstm_feat, pc_feat, type_feat, town_feat], dim=1)

        return self.fc(x)