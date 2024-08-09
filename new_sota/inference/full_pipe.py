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
    arg_par.add_argument(
        "--dry",
        "-d",
        default=False,
        const=True,
        nargs="?",
        help="set this flag to run without inference",
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
        # logging.info(line)
        output_ann.append(line + "\n")

    # TODO: test this

    # if run in verbose mode, print the text with each span, coloured
    # with its labels colour
    if logging.getLogger().getEffectiveLevel() == logging.INFO:
        indices = starts + ends
        indices.sort()

        output = ""
        output += text[: indices[0]]
        for i in range(1, len(indices)):
            if i % 2 == 1:
                label = labels[(i - 1) // 2]
                if label == "MajorClaim":
                    output += Fore.BLUE
                elif label == "Claim":
                    output += Fore.GREEN
                elif label == "Premise":
                    output += Fore.YELLOW
                else:
                    output += Fore.RED
                    output += label
            else:
                output += Style.RESET_ALL
            output += text[indices[i - 1] : indices[i]]
        output += Style.RESET_ALL + text[indices[-1] :]
        logging.info(output)

    return output_txt, output_ann


# TODO: test verbosity
# TODO: implement logging instead of print
# TODO: test models
# TODO: use custom input texts
# TODO: use custom output location
# TODO: test whether pipes jumble the order or not
# TODO: automatically determine encoding?


def main(input_path, output_dir, spans_model, labels_model, verbose, dry):
    if input_path is None:
        raise ValueError("No input path specified!")
    if output_dir is None:
        raise ValueError("No output directory specified!")
    if not input_path.exists():
        raise ValueError("specified input path does not exist!")
    if verbose:
        logging.getLogger().setLevel(logging.INFO)
    # ========================================
    # input path is file
    # ========================================
    if input_path.is_file():
        assert input_path.__str__().endswith(
            ".txt"
        ), "wrong file format supplied. required format is txt"
        # make sure output_dir exists
        if not output_dir.is_dir():
            output_dir.mkdir()
            logging.info(f"created {output_dir}")

        # TODO: what about encoding?
        # read source text
        text = input_path.read_text()
        # determine placement of spans
        spans_result = spans_pipe.inference(text, spans_model)
        # determine argument type of spans
        result = sep_tok_pipe.inference(spans_result, labels_model)
        # convert data to brat standoff format
        txt, ann = to_brat(text, result, verbose=verbose)
        # write txt to output dir
        output_dir.joinpath(input_path.name).write_text(txt)
        # write ann to output dir
        output_dir.joinpath(f"{input_path.name[:-3]}ann").write_text(ann)
    else:
        # ========================================
        # input path is directory
        # ========================================

        # find files that need to be processed
        # save path of source and output file
        # create output directory structure in the process
        source_files = []
        output_files = []
        for temp_path in input_path.rglob("*"):
            # keep the filestructure but replace the root in order to write
            # to the output dir:
            # new_path = new_root / old_path.relative_to(old_root)
            if not (
                temp_path.is_file() and temp_path.__str__().endswith(".txt")
            ):
                continue
            output_path = output_dir / temp_path.relative_to(input_path)

            # make sure output_path exists
            if not output_path.parent.is_dir():
                output_path.parent.mkdir(parents=True)
                logging.info(f"created {output_path.parent}")
            source_files.append(temp_path)
            output_files.append(output_path)
        # ========================================

        # read source files
        source_texts = [
            source_file.read_text() for source_file in source_files
        ]

        if dry:
            spans_results = source_files
            results = source_files
        else:
            spans_results = spans_pipe.inference(source_texts, spans_model)
            results = sep_tok_pipe.inference(spans_results, labels_model)
        for text, result, output_file in zip(
            spans_results, results, output_files
        ):
            if dry:
                txt = str(text)
                ann = str(text)
            else:
                txt, ann = to_brat(text, result, verbose=verbose)
            with open(output_file, "w") as w:
                w.write(txt)
            with open(str(output_file)[:-3] + "ann", "w") as w:
                w.writelines(ann)


if __name__ == "__main__":
    args = get_args()
    main(
        args.input_path,
        args.output_dir,
        args.spans_model,
        args.labels_model,
        args.verbose,
        args.dry,
    )
