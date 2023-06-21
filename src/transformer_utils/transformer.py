import math
import torch
from torch import nn
from torch.nn import TransformerEncoderLayer, TransformerEncoder
from torch.utils.data import DataLoader


class TransformerEncoderModel(nn.Module):
    def __init__(self, ntoken, ninp, nhead, nhid, nlayers, dropout=0.5):
        super(TransformerEncoderModel, self).__init__()
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(ninp, dropout)
        encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.ninp = ninp
        self.decoder = nn.Linear(ninp, ntoken)
        self.ntoken = ntoken

        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, src, src_mask):
        src = self.encoder(src) * math.sqrt(self.ninp)
        src = self.pos_encoder(src)
        output = self.transformer_encoder(src, src_mask)
        output = self.decoder(output)
        return output


######################################################################
# ``PositionalEncoding`` module injects some information about the
# relative or absolute position of the tokens in the sequence. The
# positional encodings have the same dimension as the embeddings so that
# the two can be summed. Here, we use ``sine`` and ``cosine`` functions of
# different frequencies.
#


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


INIT_STD = 0.02
PROJ_INIT_STD = 0.01


def init_weight(weight):
    torch.nn.init.normal_(weight, 0.0, INIT_STD)


def init_bias(bias):
    torch.nn.init.constant_(bias, 0.0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Linear") != -1:
        if hasattr(m, "weight") and m.weight is not None:
            init_weight(m.weight)
        if hasattr(m, "bias") and m.bias is not None:
            init_bias(m.bias)
    elif classname.find("AdaptiveEmbedding") != -1:
        if hasattr(m, "emb_projs"):
            for i in range(len(m.emb_projs)):
                if m.emb_projs[i] is not None:
                    torch.nn.init.normal_(m.emb_projs[i], 0.0, PROJ_INIT_STD)
    elif classname.find("Embedding") != -1:
        if hasattr(m, "weight"):
            init_weight(m.weight)
    elif classname.find("ProjectedAdaptiveLogSoftmax") != -1:
        if hasattr(m, "cluster_weight") and m.cluster_weight is not None:
            init_weight(m.cluster_weight)
        if hasattr(m, "cluster_bias") and m.cluster_bias is not None:
            init_bias(m.cluster_bias)
        if hasattr(m, "out_projs"):
            for i in range(len(m.out_projs)):
                if m.out_projs[i] is not None:
                    nn.init.normal_(m.out_projs[i], 0.0, PROJ_INIT_STD)
    elif classname.find("LayerNorm") != -1:
        if hasattr(m, "weight"):
            nn.init.normal_(m.weight, 1.0, INIT_STD)
        if hasattr(m, "bias") and m.bias is not None:
            init_bias(m.bias)
    elif classname.find("TransformerLM") != -1:
        if hasattr(m, "r_emb"):
            init_weight(m.r_emb)
        if hasattr(m, "r_w_bias"):
            init_weight(m.r_w_bias)
        if hasattr(m, "r_r_bias"):
            init_weight(m.r_r_bias)
        if hasattr(m, "r_bias"):
            init_bias(m.r_bias)



@torch.no_grad()
def evaluate(model, dataloader, device, grad_accumulate, accumulate_steps, no_loss=False):
    dataloader = dataloader
    model.eval()  # Turn on the train mode
    model.to(device)
    total_loss = 0.0
    ppl_loss = 0.0
    total_len = 0
    counter = 0
    m = 0

    for batch, (data, target, seq_len) in enumerate(dataloader):
        src_mask = model.generate_square_subsequent_mask(seq_len).to(device)
        output = model(data, src_mask)
        output_flat = output.view(-1, model.ntoken)
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        loss = criterion(output_flat, target.view(-1).long()).item()
        ppl_loss += seq_len * loss
        total_len += seq_len
        if grad_accumulate:
            loss = loss / accumulate_steps
        total_loss += loss
        counter += 1
        if not grad_accumulate or counter % accumulate_steps == 0:
            m += 1
        # if self.full_batch:
        #     break

    if no_loss:
        epoch_loss = 0
    else:
        epoch_loss = total_loss / m

    return ppl_loss / total_len, epoch_loss

@torch.no_grad()
def evaluate_transformer_xl(model, dataloader, device, grad_accumulate, accumulate_steps, no_loss=False):
    model.eval()
    model.to(device)
    total_loss = 0.0
    ppl_loss = 0.0
    total_len = 0
    counter = 0
    m = 0
    mems = tuple()

    for batch, (data, target, seq_len) in enumerate(dataloader):
        ret = model(data, target, *mems)
        loss, mems = ret[0], ret[1:]
        loss = loss.mean()
        ppl_loss += seq_len * loss.item()
        total_len += seq_len
        if grad_accumulate:
            loss = loss / accumulate_steps
        total_loss += loss.item()
        counter += 1
        if not grad_accumulate or counter % accumulate_steps == 0:
            m += 1
        # if self.full_batch:
        #     break
    if no_loss:
        epoch_loss = 0
    else:
        epoch_loss = total_loss / m

    return ppl_loss / total_len, epoch_loss
