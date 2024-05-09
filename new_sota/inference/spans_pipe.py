"""
    argument_mining_SuG is aimed at improving argument component 
    identification and classification based on Stab and Gurevychs prior work.
    Copyright (C) 2024  Hugo Meinhof (Theoreticallyhugo)

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU Affero General Public License as published
    by the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Affero General Public License for more details.

    You should have received a copy of the GNU Affero General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import argparse

import torch
import datasets

from tqdm import tqdm
from typing import Dict, List
from transformers import set_seed

from pipe_base import get_pipe


def inference(texts: List[str] = None):
    # =========================================
    # setting all the seeds for reproducability
    seed = 42
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)
    # set seed for huggingface stuff (the most important seed)
    set_seed(seed)
    # _________________________________________

    # =========================================
    # get the pipeline
    model_name = "longformer-spans"

    pipe = get_pipe(model_name)
    # -----------------------------------------

    # =========================================
    # get the dataset and ner_tags
    if texts is None:
        ds = datasets.load_dataset(
            "Theoreticallyhugo/essays_SuG", "spans", trust_remote_code=True
        )
        texts = ds["train"]["text"]
    # -----------------------------------------

    print("running inference")
    results = [
        out
        for out in tqdm(
            pipe(
                texts,
                batch_size=8,
            )
        )
    ]

    # freeing up vram
    del pipe

    print("weeding spans that are too short")
    # remove spans that are considered too short
    clean_results: List[List[Dict]] = []

    for result in tqdm(results):
        clean_results.append([])
        current_index = result[0]["index"]
        tokens_tmp = []
        for token in result:
            if token["index"] - current_index <= 1:
                # if its still continuing
                tokens_tmp.append(token)
                current_index = token["index"]
                continue
            elif len(tokens_tmp) <= 3:
                # if the span is over and too short, reset to new token
                tokens_tmp = [token]
                current_index = token["index"]
            else:
                # if the span is over and long enough
                clean_results[-1].extend(tokens_tmp)
                tokens_tmp = [token]
                current_index = token["index"]
        if len(tokens_tmp) > 3:
            clean_results[-1].extend(tokens_tmp)

    results = clean_results

    # find beginning and end of span
    # any connected span tokens are considered a span, unless interrupted by a B
    cls_tok: List[List[int]] = []
    sep_tok: List[List[int]] = []

    for i in range(len(results)):
        if len(results) == 0:
            continue
        # fill in first token as next loop cant handle the start
        token = results[i][0]
        current_end = token["end"]
        current_index = token["index"]
        cls_tok.append([token["start"]])
        sep_tok.append([])
        if len(results) <= 1:
            sep_tok[-1].append(current_end)
            continue
        for token in results[i][1:]:
            if token["entity"] == "B" or current_index + 1 != token["index"]:
                # begin new span
                cls_tok[-1].append(token["start"])
                sep_tok[-1].append(current_end)
            current_index = token["index"]
            current_end = token["end"]
        sep_tok[-1].append(current_end)

    print("inserting sep toks")
    # inserting cls and sep tok into the texts for the next pipe
    # these loops destroy the variables: texts, cls_tok, and sep_tok
    for i in tqdm(range(len(texts))):
        while len(cls_tok[i]) > 0:
            cur_cls = cls_tok[i].pop()
            cur_sep = sep_tok[i].pop()
            texts[i] = texts[i][:cur_sep] + "</s>" + texts[i][cur_sep:]
            texts[i] = texts[i][:cur_cls] + "<s>" + texts[i][cur_cls:]

    return texts


if __name__ == "__main__":
    texts = inference()
    for text in texts:
        input(text)
