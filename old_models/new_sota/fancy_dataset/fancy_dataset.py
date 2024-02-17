# für kompletten text tokens mit labels liefern

# Copyright 2020 The HuggingFace Datasets Authors and the current dataset script contributor.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# TODO: Address all TODOs and remove all explanatory comments
"""TODO: Add a description here."""


import json
from pathlib import Path

import datasets


# Find for instance the citation on arxiv or on the dataset repo/website
_CITATION = """\
@InProceedings{huggingface:dataset,
title = {a fancy dataset},
author={Hugo Meinhof, Elisa Luebbers},
year={2024}
}
"""

# TODO: Add description of the dataset here
# You can copy an official description
_DESCRIPTION = """\
This dataset contains 402 argumentative essays from non-native """

# TODO: Add a link to an official homepage for the dataset here
_HOMEPAGE = ""

# TODO: Add the licence for the dataset here if you can find it
_LICENSE = ""

# TODO: Add link to the official dataset URLs here
# The HuggingFace Datasets library doesn't host the datasets but only points to the original files.
# This can be an arbitrary nested dict/list of URLs (see below in `_split_generators` method)
# _URLS = {
#     "first_domain": "https://huggingface.co/great-new-dataset-first_domain.zip",
#     "second_domain": "https://huggingface.co/great-new-dataset-second_domain.zip",
# }


class Fancy(datasets.GeneratorBasedBuilder):
    """
    TODO: Short description of my dataset.
    """

    VERSION = datasets.Version("1.1.0")

    # This is an example of a dataset with multiple configurations.
    # If you don't want/need to define several sub-sets in your dataset,
    # just remove the BUILDER_CONFIG_CLASS and the BUILDER_CONFIGS attributes.

    # If you need to make complex sub-parts in the datasets with configurable options
    # You can create your own builder configuration class to store attribute, inheriting from datasets.BuilderConfig
    # BUILDER_CONFIG_CLASS = MyBuilderConfig

    # You will be able to load one or the other configurations in the following list with
    # data = datasets.load_dataset('my_dataset', 'first_domain')
    # data = datasets.load_dataset('my_dataset', 'second_domain')

    BUILDER_CONFIGS = [
        datasets.BuilderConfig(
            name="full_labels",
            version=VERSION,
            description="get all the data conveyed by the labels, O, B-Claim, I-Claim, etc.",
        ),
        datasets.BuilderConfig(
            name="spans",
            version=VERSION,
            description="get the spans, O, B-Span, I-Span.",
        ),
        datasets.BuilderConfig(
            name="simple",
            version=VERSION,
            description="get the labels without B/I, O, MajorClaim, Claim, Premise",
        ),
        datasets.BuilderConfig(
            name="sep_tok",
            version=VERSION,
            description="get the labels without B/I, meaning O, Claim, Premise"
            + ", etc.\n insert seperator tokens <s> ... </s>",
        ),
        datasets.BuilderConfig(
            name="sep_tok_full_labels",
            version=VERSION,
            description="get the labels with B/I, meaning O, I-Claim, I-Premise"
            + ", etc.\n insert seperator tokens <s> ... </s>",
        ),
    ]

    DEFAULT_CONFIG_NAME = "full_labels"

    def _info(self):
        # This method specifies the datasets.DatasetInfo object which contains informations and typings for the dataset
        if (
            self.config.name == "full_labels"
        ):  # This is the name of the configuration selected in BUILDER_CONFIGS above
            features = datasets.Features(
                {
                    "id": datasets.Value("int16"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.ClassLabel(
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
                    ),
                    "text": datasets.Value("string"),
                }
            )
        elif (
            self.config.name == "spans"
        ):  # This is an example to show how to have different features for "first_domain" and "second_domain"
            features = datasets.Features(
                {
                    "id": datasets.Value("int16"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.ClassLabel(
                            names=[
                                "O",
                                "B",
                                "I",
                            ]
                        )
                    ),
                    "text": datasets.Value("string"),
                }
            )
        elif (
            self.config.name == "simple"
        ):  # This is an example to show how to have different features for "first_domain" and "second_domain"
            features = datasets.Features(
                {
                    "id": datasets.Value("int16"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.ClassLabel(
                            names=[
                                "O",
                                "X_placeholder_X",
                                "MajorClaim",
                                "Claim",
                                "Premise",
                            ]
                        )
                    ),
                    "text": datasets.Value("string"),
                }
            )
        elif self.config.name == "sep_tok":
            features = datasets.Features(
                {
                    "id": datasets.Value("int16"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.ClassLabel(
                            names=[
                                "O",
                                "X_placeholder_X",
                                "MajorClaim",
                                "Claim",
                                "Premise",
                            ]
                        )
                    ),
                    "text": datasets.Value("string"),
                }
            )
        elif self.config.name == "sep_tok_full_labels":
            features = datasets.Features(
                {
                    "id": datasets.Value("int16"),
                    "tokens": datasets.Sequence(datasets.Value("string")),
                    "ner_tags": datasets.Sequence(
                        datasets.ClassLabel(
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
                    ),
                    "text": datasets.Value("string"),
                }
            )

        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # This defines the different columns of the dataset and their types
            features=features,  # Here we define them above because they are different between the two configurations
            # If there's a common (input, target) tuple from the features, uncomment supervised_keys line below and
            # specify them. They'll be used if as_supervised=True in builder.as_dataset.
            # supervised_keys=("sentence", "label"),
            # Homepage of the dataset for documentation
            homepage=_HOMEPAGE,
            # License for the dataset if available
            license=_LICENSE,
            # Citation for the dataset
            citation=_CITATION,
        )

    def _range_generator(self, train=0.8, test=0.2):
        """
        returns three range objects to access the list of essays
        these are the train, test, and validate range, where the size of the
        validation range is dictated by the other two ranges
        """
        return (
            range(0, int(402 * train)),  # train
            range(int(402 * train), int(402 * (train + test))),  # test
            range(int(402 * (train + test)), 402),  # validate
        )

    @staticmethod
    def _find_data():
        """
        try to find the data folder and return the path to it if found,
        otherwise return none

        returns:
            path to data folder or None
        """

        # get path to the current working directory
        cwd = Path.cwd()
        # check for whether the data folder is in cwd.
        # if it isnt, change cwd to its parent directory
        # do this three times only (dont want infinite recursion)
        for _ in range(3):
            if Path.is_dir(cwd / "fancy_dataset"):
                # print(f"found 'data' folder at {cwd}")
                # input(f"returning {cwd / 'data'}")
                return cwd / "fancy_dataset"
            cwd = cwd.parent
        raise FileNotFoundError("data directory has not been found")

    def _get_essay_list(self):
        """
        read the essay.json and return a list of dicts, where each dict is an essay
        """

        path = self._find_data() / "essay.json"
        with open(path, "r") as r:
            lines = r.readlines()

        essays = []
        for line in lines:
            essays.append(json.loads(line))

        return essays

    def _split_generators(self, dl_manager):
        # This method is tasked with downloading/extracting the data and defining the splits depending on the configuration
        # If several configurations are possible (listed in BUILDER_CONFIGS), the configuration selected by the user is in self.config.name

        # dl_manager is a datasets.download.DownloadManager that can be used to download and extract URLS
        # It can accept any type or nested list/dict and will give back the same structure with the url replaced with path to local files.
        # By default the archives will be extracted and a path to a cached folder where they are extracted is returned instead of the archive

        train, test, validate = self._range_generator()
        essays = self._get_essay_list()

        if len(validate) > 0:
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "data": essays,
                        "id_range": train,
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.VALIDATION,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "data": essays,
                        "id_range": validate,
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "data": essays,
                        "id_range": test,
                    },
                ),
            ]
        else:
            return [
                datasets.SplitGenerator(
                    name=datasets.Split.TRAIN,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "data": essays,
                        "id_range": train,
                    },
                ),
                datasets.SplitGenerator(
                    name=datasets.Split.TEST,
                    # These kwargs will be passed to _generate_examples
                    gen_kwargs={
                        "data": essays,
                        "id_range": test,
                    },
                ),
            ]

    def _get_id(self, essay):
        return int(essay["docID"].split("_")[-1])

    def _get_tokens(self, essay):
        tokens = []
        for sentence in essay["sentences"]:
            for token in sentence["tokens"]:
                tokens.append((token["surface"], token["gid"]))
        return tokens

    def _get_label_dict(self, essay):
        label_dict = {}
        for unit in essay["argumentation"]["units"]:
            if self.config.name == "spans":
                label = "Span"
            else:
                label = unit["attributes"]["role"]
            for i, gid in enumerate(unit["tokens"]):
                if i == 0:
                    location = "B-"
                else:
                    location = "I-"
                label_dict[gid] = location + label
        return label_dict

    def _match_tokens(self, tokens, label_dict):
        text = []
        labels = []
        for surface, gid in tokens:
            # for each token, unpack it into its surface and gid
            # then match the gid to the label and pack them back together

            # if the config requires separator tokens
            if (
                self.config.name == "sep_tok"
                or self.config.name == "sep_tok_full_labels"
            ):
                if label_dict.get(gid, "O")[0] == "B":
                    # if we are at the beginning of a span
                    # insert begin of sequence token (BOS) and "O" label
                    text.append("<s>")
                    labels.append("O")
                elif (
                    label_dict.get(gid, "O") == "O"
                    and len(labels) != 0
                    and labels[-1][0] != "O"
                ):
                    # if we are not in a span, and the previous label was
                    # of a span
                    # intert end of sequence token (EOS) and "O" label
                    text.append("</s>")
                    labels.append("O")

            # always append the surface form
            text.append(surface)

            # append the correct type of label, depending on the config
            if self.config.name == "full_labels":
                labels.append(label_dict.get(gid, "O"))

            elif self.config.name == "spans":
                labels.append(label_dict.get(gid, "O")[0])

            elif self.config.name == "simple":
                labels.append(label_dict.get(gid, "__O")[2:])

            elif self.config.name == "sep_tok":
                labels.append(label_dict.get(gid, "__O")[2:])

            elif self.config.name == "sep_tok_full_labels":
                labels.append(label_dict.get(gid, "O"))

            else:
                raise KeyError()
        return text, labels

    def _get_text(self, essay):
        return essay["text"]

    def _process_essay(self, essay):
        id = self._get_id(essay)
        # input(id)
        tokens = self._get_tokens(essay)
        # input(tokens)
        label_dict = self._get_label_dict(essay)
        # input(label_dict)
        tokens, labels = self._match_tokens(tokens, label_dict)
        # input(tokens)
        # input(labels)
        text = self._get_text(essay)
        return {"id": id, "tokens": tokens, "ner_tags": labels, "text": text}

    # method parameters are unpacked from `gen_kwargs` as given in `_split_generators`
    def _generate_examples(self, data, id_range):
        # This method handles input defined in _split_generators to yield (key, example) tuples from the dataset.
        # The `key` is for legacy reasons (tfds) and is not important in itself, but must be unique for each example.

        for id in id_range:
            # input(data[id])
            yield id, self._process_essay(data[id])
