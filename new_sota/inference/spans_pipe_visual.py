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

from transformers import set_seed
from pipe_base import get_pipe
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset
import torch
import datasets
from typing import Dict, List
from colorama import Fore, Back, Style
from transformers import AutoTokenizer


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
ds = datasets.load_dataset(
    "Theoreticallyhugo/essays_SuG", "spans", trust_remote_code=True
)
# -----------------------------------------

print("running inference")
results = [
    out
    for out in tqdm(
        pipe(
            KeyDataset(ds["train"], "text"),
            batch_size=8,
        )
    )
]

# freeing up vram
del pipe
texts = ds["train"]["text"]

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


# highlight the spans that were found
for i in range(len(cls_tok)):
    print(f">>> text {i}")
    text = ds["train"]["text"][i]
    gold_cls_tok = ds["train"]["span_begins"][i]
    gold_sep_tok = ds["train"]["span_ends"][i]
    inside_gold = False
    inside_inference = False
    for letter in range(len(text)):

        # detect change
        if letter in gold_cls_tok:
            inside_gold = True
        if letter in cls_tok[i]:
            inside_inference = True

        # apply colour
        if inside_gold and inside_inference:
            # correct match
            print(Fore.GREEN, end="")
        elif inside_gold:
            # not matched
            print(Fore.YELLOW, end="")
        elif inside_inference:
            # incorrect match
            print(Fore.RED, end="")
        else:
            # correctly not matched
            print(Fore.RESET, end="")

        # print letter
        print(text[letter], end="")

        # detect change
        if letter in gold_sep_tok:
            inside_gold = False
        if letter in sep_tok[i]:
            inside_inference = False

    print(Style.RESET_ALL)
    if input() != "":
        break

# highlight gold-spans in green
for i in range(len(ds["train"]["text"])):
    print(f">>> text {i}")
    text = ds["train"]["text"][i]
    tmp_cls_tok = ds["train"]["span_begins"][i]
    tmp_sep_tok = ds["train"]["span_ends"][i]
    for letter in range(len(text)):
        if letter in tmp_cls_tok:
            # if a span starts here, set fore to green
            print(Fore.GREEN, end="")
        print(text[letter], end="")
        # input(text[letter])
        if letter in tmp_sep_tok:
            # if a span ends here, reset fore
            print(Fore.RESET, end="")

    if input() != "":
        break
print(Style.RESET_ALL)

brk = ""
for i in range(len(cls_tok)):
    print(f">>> text {i}")
    for j in range(len(cls_tok[i])):
        brk = input(
            KeyDataset(ds["train"], "text")[i][cls_tok[i][j] : sep_tok[i][j]]
        )
        if brk != "":
            break
    if brk != "":
        break

brk = ""
for i in range(len(results)):
    for j in range(len(results[i])):
        brk = input(results[i][j])
        if brk != "":
            break
    if brk != "":
        break

print("inserting sep toks")
# inserting cls and sep tok into the texts for the next pipe
# these loops destroy the variables: texts, cls_tok, and sep_tok
for i in tqdm(range(len(texts))):
    while len(cls_tok[i]) > 0:
        cur_cls = cls_tok[i].pop()
        cur_sep = sep_tok[i].pop()
        texts[i] = texts[i][:cur_sep] + "</s>" + texts[i][cur_sep:]
        texts[i] = texts[i][:cur_cls] + "<s>" + texts[i][cur_cls:]


dss = datasets.load_dataset(
    "Theoreticallyhugo/essays_SuG", "sep_tok", trust_remote_code=True
)

model_name = "longformer-spans"
tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
pipe_tokens = [tokenizer(text, truncation=True) for text in texts]

gold_tokens = [
    tokenizer(text, truncation=True, is_split_into_words=True)
    for text in dss["train"]["tokens"]
]

for i in range(len(pipe_tokens)):
    print(Fore.YELLOW, end="")
    print(gold_tokens[i].tokens())
    print(Fore.GREEN, end="")
    print(pipe_tokens[i].tokens())
    if input():
        break

print(Style.RESET_ALL)
