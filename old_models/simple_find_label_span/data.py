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
from datasets import Dataset
from datasets.dataset_dict import DatasetDict
from datasets import ClassLabel
from sklearn.utils import validation
from tqdm import tqdm


# NOTE TODO FIXME XXX need datasets for spans only and then the labels for spans


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
        self.__label_span_tag = ClassLabel(names=["MajorClaim", "Claim", "Premise"])

    def set_ner_tags(self, ner_tags):
        self.__ner_tags = ner_tags

    def get_ner_tags(self):
        return self.__ner_tags

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
        take an essay and convert it to the AI readable version
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

    def get_dataset_spans(
        self,
        essays,
        validation_size=0.1,
        test_size=0.1,
    ):
        """
        provides dataset (dictionary) and labels for:
        find_spans: finding spans in paragraphs
        label_spans: the action of labeling the spans that have been found
        """
        data_set = self.get_dataset(
            essays,
            validation_size,
            test_size,
            True,
        )
        """
        return:
            data_set_find
            self.__find_span_tag
            data_set_label
            self.__label_span_tag

        prepare dataset like 
        id: [0,0,0,1,1, ... ] the index of the essay split over the paragraphs
        ner_tags: [[O, B-Span, I-Span, O, O], [B-Span, I-Span, ...], ...]  THIS IS SAVED IN INT THO. 
        list of lists of ints, meaning list of paragraphs with tokens (ai readable)
        tokens: [[hello, I, am, me,. ], [this, is, ...], ...] list of lists of strings, meaning list of paragraphs with words
        with labels: "O", "B-Span", "I-Span"
        and
        FIXME
        id: [0,0,0,1,1, ... ] the index of the essay split over the ?
        ner_tags: [[Claim, Claim], [Premise, Premise, ...] ...]  THIS IS SAVED IN INT THO
        tokens: [[I, am], [this, is, ...], ...]
        with labels: "MajorClaim", "Claim", "Premise"
        in the end we assign the most annotated label for each sentence (is that mean or median? i think its mean)

        remember converting from __ner_tags to __find_span_tag and __label_span_tag respectively
        """
        data_set_find = {
            "id": [],
            "ner_tags": [],
            "tokens": [],
        }
        # FIXME this needs a nested loop
        # corresponding tags
        # __find_span_tag
        for i in tqdm(range(len(data_set["id"]))):
            # keep the same id
            data_set_find["id"].append(data_set["id"][i])
            paragraph_tags = []
            paragraph_tokens = []
            for j in range(len(data_set["ner_tags"][i])):
                # keep all O and reduce all B or I labels to B-Span and I-Span
                # reduce by first converting to string, then only keeping the first
                # letter which is I or B. then add "-Span" and convert back to int
                if self.__ner_tags.int2str(data_set["ner_tags"][i][j]) == "O":
                    # if it is an O, append "O" with proper numbering
                    paragraph_tags.append(self.__find_span_tag.str2int("O"))
                else:
                    # input(data_set["ner_tags"][i][j])
                    # if its not an O, modify
                    # first decode then modify then encode
                    paragraph_tags.append(
                        self.__find_span_tag.str2int(
                            self.__ner_tags.int2str(data_set["ner_tags"][i][j])[0]
                            + "-Span"
                        )
                    )
                # keep the same token
                paragraph_tokens.append(data_set["tokens"][i][j])
            data_set_find["ner_tags"].append(paragraph_tags)
            data_set_find["tokens"].append(paragraph_tokens)
        ds_find, _ = self.dataset_from_dict(data_set_find, validation_size, test_size)
        return (
            ds_find,
            self.__find_span_tag,
        )

    def get_dataset_spans_labels(
        self,
        essays,
        validation_size=0.1,
        test_size=0.1,
        NER=True,
    ):
        """
        labeled spans for NER or sentence classification
        """
        data_set = self.get_dataset(
            essays,
            validation_size,
            test_size,
            True,
        )

        data_set_label = {
            "id": [],
            "ner_tags": [],
            "tokens": [],
        }

        # corresponding tags
        # __label_span_tag
        for i_para in tqdm(range(len(data_set["tokens"]))):
            # for each paragraph
            sentence_id = []
            sentence_tok = []
            sentence_tag = []
            for i_tok in range(len(data_set["tokens"][i_para])):
                # for each token in the paragraph
                if (
                    self.__ner_tags.int2str(data_set["ner_tags"][i_para][i_tok])[0]
                    == "O"
                ):
                    if sentence_tok == []:
                        continue
                    else:
                        # append the data
                        data_set_label["tokens"].append(sentence_tok)
                        if NER:
                            # append ids, matched to all tokens (as list)
                            data_set_label["ner_tags"].append(sentence_tag)
                        else:
                            # append a single id for the entire sentence
                            data_set_label["ner_tags"].append(sentence_tag[0])
                        data_set_label["id"].append(sentence_id)
                        # reset the lists
                        sentence_tok = []
                        sentence_tag = []
                        sentence_id = []
                else:
                    # append current token unchanged
                    sentence_tok.append(data_set["tokens"][i_para][i_tok])
                    # append current id unchanged
                    sentence_id.append(data_set["id"][i_para])
                    # append modified ner tag
                    # convert the saved int to the corresponding string,
                    # modify the tag, and convert to the new corresponding int

                    sentence_tag.append(
                        self.__label_span_tag.str2int(
                            self.__ner_tags.int2str(
                                data_set["ner_tags"][i_para][i_tok]
                            )[2:]
                        )
                    )

        if not NER:
            data_set_label["label"] = data_set_label["ner_tags"]
            del data_set_label["ner_tags"]
        ds_label, _ = self.dataset_from_dict(data_set_label, validation_size, test_size)

        return (
            ds_label,
            self.__label_span_tag,
        )

    def get_dataset_sents(
        self,
        essays,
        validation_size=0.1,
        test_size=0.1,
        match_paragraphs=False,
    ):
        """
        provides backwards compatability, but allows split of dictionary
        creation and dataset from dictionary creation
        """
        data_set = self.get_dataset(
            essays,
            validation_size,
            test_size,
            match_paragraphs,
        )
        return self.dataset_from_dict(data_set, validation_size, test_size)

    def get_dataset_text(
        self,
        essays,
        validation_size=0.1,
        test_size=0.1,
    ):
        """
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

        # change dict to hf dataset and return
        return self.dataset_from_dict(data_set, validation_size, test_size)

    def get_dataset(
        self,
        essays,
        validation_size=0.1,
        test_size=0.1,
        match_paragraphs=False,
    ):
        """
        return a DatasetDict with train, validation and test, for use with the nn
        if validation and test are both zero, return a single dataset
        args:
            essays: list of dicts, with each dict being an essay
            validation_size: size of the validation set as a percentage of the
                entire set size
            test_size: size of the test set as a percentage of the
                entire set size
            match_paragraphs: return paragraphs instead of sentences
        """
        assert isinstance(essays, list) or isinstance(essays, tuple)
        assert 0 < validation_size < 1
        assert 0 < test_size < 1

        data_set = {
            "id": [],
            "ner_tags": [],
            "tokens": [],
        }

        id = 0
        for i, essay in enumerate(essays):
            # debug = input(f"processing essay {i}")
            debug = False
            sentences = self.convert_essay_sents(essay)
            if debug:
                print(f"sentences: {sentences}\n")
            if match_paragraphs:
                # if we want to match paragraphs, we just replace the sentences
                # by paragraphs
                paragraphs = self.get_paragraphs(essay=essay)
                if debug:
                    print(f"paragraphs: {paragraphs}\n")
                sentences = self.match_sent_to_par(
                    paragraphs=paragraphs, sentences=sentences, debug=debug
                )
                if debug:
                    print(f"sentences_replaces: {sentences}\n")
            for sent in sentences:
                ner_tags, tokens = sent
                data_set["id"].append(id)
                id += 1
                data_set["ner_tags"].append(ner_tags)
                data_set["tokens"].append(tokens)
        return data_set

    def dataset_from_dict(
        self,
        data_set,
        validation_size=0.1,
        test_size=0.1,
    ):
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

    @staticmethod
    def get_paragraphs(essay: dict):
        """
        paragraphs are separated by a single newline.
        titles are separated by two newlines.
        here i remove the empty line between title and text body, in order to
        prevent training on an empty paragraph.

        args:
            essay: dicitonary in the structure of the original essays.json
        returns:
            list of strings, where each string is a paragraph
        """
        paragraphs = essay["text"].split("\n")
        i = 0
        while i < len(paragraphs):
            if paragraphs[i] == "":
                paragraphs.pop(i)
            else:
                i += 1
        return paragraphs

    def get_paragraphs_dataset(self, essays: list):
        print("loading dataset")
        # find paragraphs and flatten into one list
        total_paragraphs = []
        essay_index = []

        for i, essay in enumerate(tqdm(essays)):
            # for each essay
            # get all paragraphs
            paragraphs = self.get_paragraphs(essay)

            for para in paragraphs:
                # for each paragraph within an essay
                # append the paragraph to the list of all paragraphs
                total_paragraphs.append(para)

            for _ in range(len(paragraphs)):
                # for each of the paragraphs within this essay
                # append the index of the essay to the list, so that
                # in the end all paragraphs can be assembled into essays again
                essay_index.append(i)

        ds = {"paragraphs": total_paragraphs, "essay_index": essay_index}
        return Dataset.from_dict(ds)

    def match_sent_to_par(self, paragraphs: list, sentences: list, debug=False):
        """
        for one essay take all paragraphs and sentences for the matching.
        go through a paragraph and
        for each character in paragraph
            try to match character in current or next token (with git and role)
            if character doesnt match token, skip it, it may be a space or so

        this should find all the tokens that are in a paragraph, regardless of
        sentence boundaries

        args:
            paragraphs: list of str, where each str is one paraghraph
            sentences: list of two lists, where the first list contains the roles
                as int, and the second the tokens as str.
        returns:
            list of two lists: first list contains the roles/ tags as int, the
                seconds contains the tokens as str.
        """
        sentence_index = 0
        token_index = 0
        character_index = 0
        matched_paragraphs = []
        for paragraph in paragraphs:
            matched_paragraph = [[], []]
            for char in paragraph:
                if char == sentences[sentence_index][1][token_index][character_index]:
                    # if the character matches, advance pointer by one character
                    if debug:
                        print(f"matched {char}")
                    character_index += 1
                    if character_index >= len(
                        sentences[sentence_index][1][token_index]
                    ):
                        if debug:
                            print(
                                f"adding token {sentences[sentence_index][1][token_index]}"
                            )
                        # add the token to the paragraph
                        matched_paragraph[0].append(
                            sentences[sentence_index][0][token_index]
                        )
                        matched_paragraph[1].append(
                            sentences[sentence_index][1][token_index]
                        )
                        # if we reached the end of the token, reset character pointer
                        # and advance token pointer by one
                        character_index = 0
                        token_index += 1
                        if token_index >= len(sentences[sentence_index][0]):
                            if debug:
                                print(
                                    f"proceeding to next sentence: {sentence_index+1}"
                                )
                            # if we reached the end of the sentence, reset token pointer
                            # and advance sentence pointer by one
                            token_index = 0
                            sentence_index += 1
                            if sentence_index >= len(sentences):
                                # reached the end, if the loop continues after this, there is a problem
                                break
                                # print("we reached the end")
                                # print(paragraph_tokens)
                                # print("")
                                # print(paragraphs)
                                # print("")
                                # print(sentences)
                else:
                    if debug:
                        print(
                            f"skipping {char}, whilst matching {sentences[sentence_index][1][token_index][character_index]}"
                        )
            matched_paragraphs.append(matched_paragraph)
        return matched_paragraphs

    @classmethod
    def unfold(cls, data, depth=0):
        """
        for visualising the data step by step...
        """
        io = input()
        try:
            io = int(io)
            if io == 1:
                print("going back to 0")
                return 1
        except:
            pass
        io = 0
        if isinstance(data, dict):
            keys = data.keys()
            for key in keys:
                print("\t" * depth + ">" + key)
                io = cls.unfold(data[key], depth + 1)
                if io == 1 and depth > 0:
                    return 1
        elif isinstance(data, list):
            print("\t" * depth + "B" + "=" * 30)
            for elem in data:
                io = cls.unfold(elem, depth + 1)
                if io == 1 and depth > 0:
                    return 1
                print("\t" * depth + "-" * 30)
            print("\t" * depth + "E" + "=" * 30)
        else:
            try:
                print("\t" * depth + str(data) + "   " + str(type(data)))
            except:
                print("\t" * depth + str(type(data)))

    @staticmethod
    def find_paragraph(essays):
        for essay in essays:
            for character in essay["text"]:
                if character == "\n":
                    print("XXX")
                else:
                    print(character, end="")
            input()


if __name__ == "__main__":
    f = open(os.path.join(os.pardir, "data", "essay.json"), "r")
    line = f.readline()
    data = json.loads(line)

    data_preparator.unfold(data)

    data_preparator.find_paragraph(
        data_preparator.get_essay_list(os.path.join(os.pardir, "data", "essay.json"))
    )
