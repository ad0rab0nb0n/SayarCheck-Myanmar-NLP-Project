import json
import sys
import os

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


sys.path.append(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "preprocessing"
    )
)

from unicode_normalizer import clean_tokens_and_tags


MAX_LENGTH = 256


class MyanmarSegmentationDataset(Dataset):

    def __init__(self, dataset, vocab_path):

        self.dataset = dataset

        with open(
            vocab_path,
            "r",
            encoding="utf-8"
        ) as f:
            self.word_to_id = json.load(f)

        self.pad_id = self.word_to_id["<PAD>"]
        self.unk_id = self.word_to_id["<UNK>"]

        # Create an index of chunks
        self.chunks = []

        print("Preparing sequence chunks...")

        for example_index in range(len(self.dataset)):

            example = self.dataset[example_index]

            tokens = example["tokens"]
            tags = example["segment_tags"]

            tokens, tags = clean_tokens_and_tags(
                tokens,
                tags
            )

            # Split long sequences into chunks
            for start in range(
                0,
                len(tokens),
                MAX_LENGTH
            ):

                end = min(
                    start + MAX_LENGTH,
                    len(tokens)
                )

                self.chunks.append(
                    (
                        example_index,
                        start,
                        end
                    )
                )

        print(
            "Total chunks created:",
            len(self.chunks)
        )

    def __len__(self):

        return len(self.chunks)

    def __getitem__(self, index):

        example_index, start, end = self.chunks[index]

        example = self.dataset[example_index]

        tokens = example["tokens"]
        tags = example["segment_tags"]

        tokens, tags = clean_tokens_and_tags(
            tokens,
            tags
        )

        # Take only this chunk
        tokens = tokens[start:end]
        tags = tags[start:end]

        input_ids = []

        for token in tokens:

            token_id = self.word_to_id.get(
                token,
                self.unk_id
            )

            input_ids.append(token_id)

        input_ids = torch.tensor(
            input_ids,
            dtype=torch.long
        )

        labels = torch.tensor(
            tags,
            dtype=torch.long
        )

        return input_ids, labels


def collate_fn(batch):

    input_ids = [
        item[0]
        for item in batch
    ]

    labels = [
        item[1]
        for item in batch
    ]

    input_ids = pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=0
    )

    labels = pad_sequence(
        labels,
        batch_first=True,
        padding_value=0
    )

    attention_mask = (
        input_ids != 0
    )

    return (
        input_ids,
        labels,
        attention_mask
    )