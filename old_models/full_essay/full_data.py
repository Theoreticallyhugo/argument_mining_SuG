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

import os
import json

from torch import topk
from datasets import Dataset
from datasets.dataset_dict import DatasetDict
from datasets import ClassLabel
from tqdm import tqdm


class data_preparator:
    def __init__(self) -> None:
        """
        by default, this will deliver the standard claim and premise labels,
        but with span_split true, it provides the datasets for spans, for when
        its the dual stage NER, where first spans are found and then classified
        """
        # set default, can be overridden
        self.__ner_tags = ClassLabel(
            names=[
                "O",
                "B-MajorClaim",
                "I-MajorClaim",
                "B-Claim",
                "I-Claim",
                "B-Premise",
                "I-Premise",
            ]
        )
        self.__find_span_tag = ClassLabel(
            names=[
                "O",
                "B-Span",
                "I-Span",
            ]
        )
        self.__label_span_tag = ClassLabel(
            names=["O", "MajorClaim", "Claim", "Premise"]
        )

    @staticmethod
    def get_essay_list(path):
        """
        read the essay.json and return a list of dicts, where each dict is an essay
        """
        with open(path, "r") as r:
            lines = r.readlines()

        essays = []
        for line in lines:
            essays.append(json.loads(line))

        return essays

    def convert_essay_sents(self, essay):
        """
        take an essay and convert it to the "AI readable" version
        returns a list of lists, of two lists each

        list of lists, where each list is a sentence, consisting of two lists,
        the first one containing the tags, and the second one containing
        the tokens
        (((ner_tag, ...), (token, ...)), ...)
        """
        # dict: for each gid that has a role, there is an entry, with the gid as key
        units = essay["argumentation"]["units"]
        role_dict = dict()
        for unit in units:
            role = unit["attributes"]["role"]
            tokens = unit["tokens"]
            for i, token in enumerate(tokens):
                if i == 0:
                    role_dict[token] = self.__ner_tags.str2int(f"B-{role}")
                else:
                    role_dict[token] = self.__ner_tags.str2int(f"I-{role}")

        # input(role_dict)

        original_sents = essay["sentences"]
        clean_sents = []
        for sent in original_sents:
            # ((ner_tags), (tokens))
            clean_sent = [[], []]
            tokens = sent["tokens"]
            for token in tokens:
                gid = token["gid"]
                surface = token["surface"]
                # get the tokens role from the role_dict, and default to 0,
                # when it doesnt have a role
                role = role_dict.get(gid, 0)
                clean_sent[0].append(role)
                clean_sent[1].append(surface)
            clean_sents.append(clean_sent)

        # input(clean_sents)

        return clean_sents

    def get_dataset_full(self, essays):
        """
        full essays dataset
        get the dataset as dict, where all the sentences are individual.
        then put all the sentences together to form the full essays,
        process it as hf dataset and return.
        """

        data_set = {
            "id": [],
            "ner_tags": [],
            "tokens": [],
        }

        for id, essay in enumerate(essays):
            # get the essay converted to just tokens and labels per sentence
            sentences = self.convert_essay_sents(essay)

            # flatten sentences to essay before adding to the dataaset
            ner_tags_tmp = []
            tokens_tmp = []
            for sent in sentences:
                ner_tags, tokens = sent
                ner_tags_tmp.extend(ner_tags)
                tokens_tmp.extend(tokens)

            # add the current essay to the dataset
            data_set["id"].append(id)
            data_set["ner_tags"].append(ner_tags_tmp)
            data_set["tokens"].append(tokens_tmp)

        return data_set
        # change dict to hf dataset and return
        # return self.dataset_from_dict(data_set, validation_size, test_size)

    def dataset_from_dict(
        self,
        data_set,
        validation_size=0.1,
        test_size=0.1,
    ):
        """
        converts a dictionary into a hf dataset, with train-test-split
        """
        # dataset with all sentences
        ds = Dataset.from_dict(data_set)

        # if validate and test are both zero
        if not validation_size and not test_size:
            # return dataset and not datasetdict with train test val split
            return ds

        # first create the training set, by splitting off the validation and test
        train_set, ds_split = ds.train_test_split(
            test_size=validation_size + test_size
        ).values()
        # now split validation and test
        validation_set, test_set = ds_split.train_test_split(
            test_size=(test_size / (validation_size + test_size))
        ).values()

        # input(train_set)
        # input(validation_set)
        # input(test_set)

        return (
            DatasetDict(
                {
                    "train": train_set,
                    "test": test_set,
                    "validation": validation_set,
                }
            ),
            self.__ner_tags,
        )

    def get_dataset_spans_labels(
        self,
        essays,
        validation_size=0.1,
        test_size=0.1,
        NER=True,
    ):
        """
        labeled spans for span finding or labeling
        NER = True for span labeling

        https://huggingface.co/docs/transformers/model_doc/longformer
        use cls_token "<s>" and sep_token "</s>" around the NER bits
        """
        # this will deliver number_of_essays lists where each list contains the
        # tokens/ ids/ tags for an entire essay
        data_set = self.get_dataset_full(essays)

        data_set_label = {
            "id": [],
            "ner_tags": [],
            "tokens": [],
        }

        print(data_set.keys())
        for i_essay in tqdm(range(len(data_set["tokens"]))):
            # prepare space for a new essay in the data_set_label
            data_set_label["id"].append(i_essay)
            data_set_label["tokens"].append([])
            data_set_label["ner_tags"].append([])

            # fill the last list of data_set_label with the current essay
            for i_tok in range(len(data_set["tokens"][i_essay])):
                if NER:
                    # TODO maybe add a token in here
                    try:
                        # token to the left is the same as current token, as token to the right
                        if (
                            data_set["ner_tags"][i_essay][i_tok - 1]
                            == data_set["ner_tags"][i_essay][i_tok]
                            and data_set["ner_tags"][i_essay][i_tok + 1]
                            == data_set["ner_tags"][i_essay][i_tok]
                        ):
                            # do nothing and just add the stuff as normal
                            pass
                        elif self.__ner_tags.int2str(data_set["ner_tags"][i_essay][i_tok])[0] == "B":
                            # the current token is the beginning of a span, so we need to add an <s> first
                            data_set_label["tokens"][-1].append("<s>")
                            data_set_label["ner_tags"][-1].append(self.__ner_tags.str2int("O"))

                        elif self.__ner_tags.int2str(data_set["ner_tags"][i_essay][i_tok])[0] == "I" \
                            and self.__ner_tags.int2str(data_set["ner_tags"][i_essay][i_tok + 1])[0] != "I":
                            # the current token is inside, but the next isnt, so we need to add an <s/> afterwards
                            pass # TODO FIXME how to add the token and tag afterwards

                        else:
                            input(
                                f'left {data_set["ner_tags"][i_essay][i_tok - 1]}, center {data_set["ner_tags"][i_essay][i_tok]}, right {data_set["ner_tags"][i_essay][i_tok + 1]}'
                            )

                    except IndexError:
                        # were either at the left or right bound

                        # if its the left bound
                        if i_tok == 0:
                            # if the essay begins with a span, add the left begin of span
                            if data_set["ner_tags"][i_essay][
                                i_tok
                            ] != self.__ner_tags.str2int("O"):
                                data_set_label["tokens"][-1].append("<s>")
                                data_set_label["ner_tags"][-1].append(self.__ner_tags.str2int("O"))
                        else:
                            # its the right bound
                            if data_set["ner_tags"][i_essay][
                                i_tok
                            ] != self.__ner_tags.str2int("O"):
                                # essay ends with a span, add right end of span
                                data_set_label["tokens"][-1].append("</s>")
                                data_set_label["ner_tags"][-1].append(self.__ner_tags.str2int("O"))

                        print("here we got a break")
                    except:
                        print("sth wrong")

                    # now that weve dealt with the special tokens, we add the
                    # normal stuff
                    data_set_label["tokens"][-1].append(
                        data_set["tokens"][i_essay][i_tok]
                    )

                        if data_set["ner_tags"][i_essay][i_tok] != self.__ner_tags.str2int(
                        "O"
                    ):
                        # removes the I- and B- from tags
                        data_set_label["ner_tags"][-1].append(
                            self.__label_span_tag.str2int(
                                self.__ner_tags.int2str(
                                    data_set["ner_tags"][i_essay][i_tok]
                                )[2:]
                            )
                        )
                    else:
                        data_set_label["ner_tags"][-1].append(
                            self.__label_span_tag.str2int("O")
                        )
                else:
                    # reduce tag to span

                    # first add the token, non modified
                    data_set_label["tokens"][-1].append(
                        data_set["tokens"][i_essay][i_tok]
                    )
                    # then add the label, reduced to span
                    # we still lack stuff here for reducing to O or span
                    # TODO

        if not NER:
            data_set_label["label"] = data_set_label["ner_tags"]
            del data_set_label["ner_tags"]
        ds_label, _ = self.dataset_from_dict(data_set_label, validation_size, test_size)

        return (
            ds_label,
            self.__label_span_tag,
        )
