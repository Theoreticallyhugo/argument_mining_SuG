#!/usr/bin/env python
# -*- coding: utf-8 -*-
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
import logging
import re
from collections import Counter
from pathlib import Path
from typing import assert_type

import datasets
import sep_tok_pipe
import spans_pipe
from colorama import Back, Fore, Style
from tqdm import tqdm


def get_args():
    """
    handles the argument parsing, when full_pipe.py is run from the commandline
    return:
        parsed commandline arguments
    """
    arg_par = argparse.ArgumentParser()
    arg_par.add_argument(
        "--input_path",
        "-i",
        # default=Path("./data/genres_original/"),
        type=Path,
        help="path to the data directory containing the "
        + "text files, or singular text file, to process .",
    )
    arg_par.add_argument(
        "--output_dir",
        "-o",
        # default=Path("./data/lyrics_original/"),
        type=Path,
        help="path to the directory to save the output.",
    )
    arg_par.add_argument(
        "--spans_model",
        "-s",
        default="Theoreticallyhugo/longformer-spans",
        type=str,
        help="model to use for finding the spans."
        + "either path to local model or path of huggingface repository"
        + 'in the format of "user/model"',
    )
    arg_par.add_argument(
        "--labels_model",
        "-l",
        default="Theoreticallyhugo/longformer-sep_tok",
        type=str,
        help="model to use for labeling the spans. "
        + "either path to local model or path of huggingface repository "
        + 'in the format of "user/model"',
    )
    arg_par.add_argument(
        "--verbose",
        "-v",
        default=False,
        const=True,
        nargs="?",
        help="set this flag to increase verbosity",
    )

    args = arg_par.parse_args()
    return args


def to_brat(text, pipe_out, verbose=False):
    """
    expects a text with separation tokens, and the ouput of the sep_tok model.

    returns the data in brat standoff format.
    return (txt, ann)

    search for <s> in the text, save the index, and look for the </s>.
    save its index too. then find all tokens from pipe_out that lie between
    these two separators, take the most frequent role and save it as a span
    """
    # find where each span begins and ends, by looking for sep_toks
    starts = [m.end(0) for m in re.finditer("<s>", text)]
    ends = [m.start(0) for m in re.finditer("</s>", text)]
    if len(starts) == 0 or len(ends) == 0:
        raise ValueError(
            "found no separator tokens in preparation of brat files"
        )
    elif len(starts) != len(ends):
        raise ValueError(
            "found unequal amount of opening and closing separator tokens"
        )

    # remove sep_toks
    text = text.replace("<s>", "").replace("</s>", "")
    if verbose:
        input(text)
    output_txt = text

    # adjust for removed sep_toks
    for i in range(len(starts)):
        starts[i] = starts[i] - 3 - (7 * i)
        ends[i] = ends[i] - 3 - (7 * i)

    # find the string for each span
    span_texts = [text[start:end] for start, end in zip(starts, ends)]
    if verbose:
        input(span_texts)

    # find the label for each span
    labels = []
    for start, end, index in zip(starts, ends, range(len(ends))):
        tmp_labels = []
        for result in pipe_out:
            if result["start"] >= start and result["end"] <= end:
                tmp_labels.append(result["entity"])
        # out of all labels found within the span, take the most frequent
        # and append to the list of labels
        if len(tmp_labels) == 0:
            # no labels found within current span

            # make sure to remove span and all associated data
            starts.pop(index)
            ends.pop(index)
            span_texts.pop(index)
        else:
            labels.append(Counter(tmp_labels).most_common(1)[0][0])

    # generate the output line by line in brat standoff format
    output_ann = []
    for id, label, start, end, span_text in zip(
        range(len(labels)), labels, starts, ends, span_texts
    ):
        line = f"T{id + 1}\t{label} {start} {end}\t{span_text}"
        print(line)
        output_ann.append(line + "\n")

    # if run in verbose mode, print the text with each span, coloured
    # with its labels colour
    if verbose:
        indices = starts + ends
        indices.sort()

        print(text[: indices[0]], end="")
        for i in range(1, len(indices)):
            if i % 2 == 1:
                label = labels[(i - 1) // 2]
                if label == "MajorClaim":
                    print(Fore.BLUE, end="")
                elif label == "Claim":
                    print(Fore.GREEN, end="")
                elif label == "Premise":
                    print(Fore.YELLOW, end="")
                else:
                    print(Fore.RED, end="")
                    print(label)
            else:
                print(Style.RESET_ALL, end="")
            print(text[indices[i - 1] : indices[i]], end="")
        print(Style.RESET_ALL + text[indices[-1] :])
        input()

    return output_txt, output_ann


if __name__ == "__main__":
    args = get_args()
    if args.input_path is None:
        raise ValueError("No input path specified!")
    if args.output_dir is None:
        raise ValueError("No output directory specified!")
    if not args.input_path.exists():
        raise ValueError("specified input path does not exist!")
    if not args.output_dir.is_dir():
        raise ValueError("specified output path is not a directory!")

    # doesnt matter which config were loading as we need the untouched texts
    ds = datasets.load_dataset(
        "Theoreticallyhugo/essays_SuG", "sep_tok", trust_remote_code=True
    )
    texts = ds["train"]["text"]
    ids = ds["train"]["id"]

    spans_results = spans_pipe.inference(texts)
    results = sep_tok_pipe.inference(spans_results)
    for text, result, id in zip(spans_results, results, ids):
        txt, ann = to_brat(text, result, verbose=args.verbose)
        with open(Path(f"essay_{str(id).rjust(3, '0')}.txt"), "w") as w:
            w.write(txt)
        with open(Path(f"essay_{str(id).rjust(3, '0')}.ann"), "w") as w:
            w.writelines(ann)
