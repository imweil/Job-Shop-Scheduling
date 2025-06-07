import torch
import torch.nn as nn
import torch.nn.functional as F

class ValueNet(torch.nn.Module):
    def __init__(self, state_dim = 88, hidden_dim=124):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class Encoder(nn.Module):
    def __init__(self, embed_size=88, num_layers=6, num_heads=8, ff_hidden_size=124, dropout=0):
        super(Encoder, self).__init__()

        self.embed_size = embed_size
        self.num_layers = num_layers

        self.encoder_layers = nn.ModuleList([
            EncoderLayer(embed_size, num_heads, ff_hidden_size, dropout) for _ in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.dropout(x)

        for layer in self.encoder_layers:
            x = layer(x, mask)

        return x


class Decoder(nn.Module):
    def __init__(self, out_size,embed_size=88, num_layers=6, num_heads=8, ff_hidden_size=124, dropout=0):
        super(Decoder, self).__init__()

        self.embed_size = embed_size
        self.num_layers = num_layers

        self.decoder_layers = nn.ModuleList([
            EncoderLayer(embed_size, num_heads, ff_hidden_size, dropout) for _ in range(num_layers)
        ])

        self.fc_out = nn.Linear(embed_size, out_size)

        self.softmax = nn.Softmax(dim=-1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = self.dropout(x)

        for layer in self.decoder_layers:
            x = layer(x, mask)
        x = x.view(x.size(0), -1)
        output = self.fc_out(x)
        output = self.softmax(output)

        return output


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_size, num_heads):
        super(MultiHeadAttention, self).__init__()

        self.embed_size = embed_size
        self.num_heads = num_heads

        assert embed_size % self.num_heads == 0

        self.head_dim = embed_size // self.num_heads

        self.query = nn.Linear(embed_size, embed_size)
        self.key = nn.Linear(embed_size, embed_size)
        self.value = nn.Linear(embed_size, embed_size)

        self.fc_out = nn.Linear(embed_size, embed_size)

    def forward(self, values, keys, query, mask=None):
        batch_size = query.shape[0]

        Q = self.query(query)
        K = self.key(keys)
        V = self.value(values)

        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,
                                                                            2)  # (batch_size, num_heads, seq_len, head_dim)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,
                                                                            2)  # (batch_size, num_heads, seq_len, head_dim)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1,
                                                                            2)  # (batch_size, num_heads, seq_len, head_dim)

        energy = torch.matmul(Q, K.transpose(-2, -1))  # (batch_size, num_heads, seq_len, seq_len)

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float('-1e9'))  # 应用掩码

        attention = torch.softmax(energy, dim=-1)  # 归一化

        out = torch.matmul(attention, V)  # (batch_size, num_heads, seq_len, head_dim)

        out = out.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.head_dim)

        out = self.fc_out(out)

        return out


class PositionwiseFeedForward(nn.Module):
    def __init__(self, embed_size, ff_hidden_size, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.fc1 = nn.Linear(embed_size, ff_hidden_size)
        self.fc2 = nn.Linear(ff_hidden_size, embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class EncoderLayer(nn.Module):
    def __init__(self, embed_size, num_heads, ff_hidden_size, dropout=0.1):
        super(EncoderLayer, self).__init__()

        self.attention = MultiHeadAttention(embed_size, num_heads)
        self.ffn = PositionwiseFeedForward(embed_size, ff_hidden_size, dropout)

        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        attention_out = self.attention(x, x, x, mask)
        attention_out = self.dropout1(attention_out)
        x = self.norm1(x + attention_out)

        ff_out = self.ffn(x)
        ff_out = self.dropout2(ff_out)
        out = self.norm2(x + ff_out)

        return out
