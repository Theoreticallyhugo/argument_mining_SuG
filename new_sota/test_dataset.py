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

import datasets
from colorama import Fore, Back, Style

ds = []
print(">>> full_labels")
ds.append(
    datasets.load_dataset("essays_SuG", "full_labels", trust_remote_code=True)
)
print(">>> spans")
ds.append(datasets.load_dataset("essays_SuG", "spans", trust_remote_code=True))
print(">>> simple")
ds.append(
    datasets.load_dataset("essays_SuG", "simple", trust_remote_code=True)
)
print(">>> sep_tok")
ds.append(
    datasets.load_dataset("essays_SuG", "sep_tok", trust_remote_code=True)
)
print(">>> sep_tok_full_labels")
ds.append(
    datasets.load_dataset(
        "essays_SuG",
        "sep_tok_full_labels",
        trust_remote_code=True,
    )
)
tests_ds = datasets.load_dataset(
    "essays_SuG",
    "spans",
    split=[f"train[{k}%:{k+20}%]" for k in range(0, 100, 20)],
)
trains_ds = datasets.load_dataset(
    "essays_SuG",
    "spans",
    split=[f"train[:{k}%]+train[{k+20}%:]" for k in range(0, 100, 20)],
)
print("done")
for s in ds:
    print("")
    print(s["train"].features)
    print(s["train"][0])


# highlight spans in green
for i in range(len(ds[1]["train"]["text"])):
    print(f">>> text {i}")
    text = ds[1]["train"]["text"][i]
    cls_tok = ds[1]["train"]["span_begins"][i]
    sep_tok = ds[1]["train"]["span_ends"][i]
    for letter in range(len(text)):
        if letter in cls_tok:
            # if a span starts here, set fore to green
            print(Fore.GREEN, end="")
        print(text[letter], end="")
        # input(text[letter])
        if letter in sep_tok:
            # if a span ends here, reset fore
            print(Fore.RESET, end="")

    if input() != "":
        break
print(Style.RESET_ALL)
