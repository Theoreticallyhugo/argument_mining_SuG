from pathlib import Path
import re

from collections import Counter

import datasets

from tqdm import tqdm
from colorama import Fore, Back, Style

import sep_tok_pipe
import spans_pipe


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
        output_ann.append(line)

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
    # doesnt matter which config were loading as we need the untouched texts
    ds = datasets.load_dataset(
        "Theoreticallyhugo/essays_SuG", "sep_tok", trust_remote_code=True
    )
    texts = ds["test"]["text"]
    ids = ds["test"]["id"]

    spans_results = spans_pipe.inference(texts)
    results = sep_tok_pipe.inference(spans_results)
    for text, result, id in zip(texts, results, ids):
        txt, ann = to_brat(text, result)
        with open(Path(f"essay_{str(id).rjust(3, '0')}.txt"), "w") as w:
            w.write(txt)
        with open(Path(f"essay_{str(id).rjust(3, '0')}.ann"), "w") as w:
            w.writelines(ann)
