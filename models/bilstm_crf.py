import torch
import torch.nn as nn
from torchcrf import CRF


class BiLSTMCRF(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        hidden_dim=128,
        num_layers=2,
        num_tags=2,
        pad_idx=0
    ):

        super().__init__()

        # -----------------------------------------
        # 1. Embedding layer
        # -----------------------------------------

        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx
        )


        # -----------------------------------------
        # 2. Bidirectional LSTM
        # -----------------------------------------

        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3 if num_layers > 1 else 0
        )


        # -----------------------------------------
        # 3. Linear layer
        # -----------------------------------------

        # Because the LSTM is bidirectional,
        # its output size is hidden_dim * 2.

        self.hidden2tag = nn.Linear(
            hidden_dim * 2,
            num_tags
        )


        # -----------------------------------------
        # 4. CRF layer
        # -----------------------------------------

        self.crf = CRF(
            num_tags=num_tags,
            batch_first=True
        )


    def forward(self, input_ids, attention_mask, labels=None):

        # -----------------------------------------
        # Embedding
        # -----------------------------------------

        embeddings = self.embedding(input_ids)


        # -----------------------------------------
        # BiLSTM
        # -----------------------------------------

        lstm_output, _ = self.lstm(embeddings)


        # -----------------------------------------
        # Linear layer
        # -----------------------------------------

        emissions = self.hidden2tag(lstm_output)


        # -----------------------------------------
        # CRF
        # -----------------------------------------

        if labels is not None:

            # CRF expects a mask of boolean values.
            # Every sequence must have at least one
            # valid token.

            mask = attention_mask.bool()

            # CRF calculates log likelihood.
            # We use negative log likelihood as loss.

            loss = -self.crf(
                emissions,
                labels,
                mask=mask,
                reduction="mean"
            )

            return loss

        else:

            # During prediction, decode the best
            # sequence of B/I labels.

            mask = attention_mask.bool()

            predictions = self.crf.decode(
                emissions,
                mask=mask
            )

            return predictions