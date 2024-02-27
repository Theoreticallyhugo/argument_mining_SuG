import re

from collections import Counter

import datasets

from tqdm import tqdm
from colorama import Fore, Back, Style

import sep_tok_pipe
import spans_pipe


def to_brat(text, pipe_out, verbose=False):
    # search for <s> in the text, save the index, and look for the </s>.
    # save its index too. then find all tokens from pipe_out that lie between
    # these two separators, take the most frequent role and save it as a span
    # FIXME: remove sep-toks from the text before returning brat format
    starts = [m.end(0) for m in re.finditer("<s>", text)]
    # print(len(cls_toks))
    ends = [m.start(0) for m in re.finditer("</s>", text)]
    # print(len(sep_toks))

    # remove sep_toks
    text = text.replace("<s>", "").replace("</s>", "")
    if verbose:
        input(text)

    # adjust for removed sep_toks
    for i in range(len(starts)):
        starts[i] = starts[i] - 3 - (7 * i)
        ends[i] = ends[i] - 3 - (7 * i)

    span_texts = [text[start:end] for start, end in zip(starts, ends)]
    if verbose:
        input(span_texts)
    labels = []
    for start, end in zip(starts, ends):
        tmp_labels = []
        for result in pipe_out:
            if result["start"] >= start and result["end"] <= end:
                tmp_labels.append(result["entity"])

        # print(tmp_labels)
        # input(Counter(tmp_labels).most_common(1)[0][0])

        # out of all labels found within the span, take the most frequent
        # and append to the list of labels
        # FIXME: this crashes if there are no annotations in the span
        #   remove span if this is the case?
        labels.append(Counter(tmp_labels).most_common(1)[0][0])

    # TODO: return this as proper file
    for id, label, start, end, span_text in zip(
        range(len(labels)), labels, starts, ends, span_texts
    ):
        print(f"T{id + 1}\t{label} {start} {end}\t{span_text}")

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


if __name__ == "__main__":
    ds = datasets.load_dataset(
        "Theoreticallyhugo/essays_SuG", "sep_tok", trust_remote_code=True
    )
    texts = ds["test"]["text"]

    spans_results = spans_pipe.inference(texts)
    results = sep_tok_pipe.inference(spans_results)
    for text, result in zip(texts, results):
        to_brat(text, result)
