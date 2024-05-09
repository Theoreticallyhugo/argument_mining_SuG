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

"""
the goal is to do NER on full essays in a generalised way
"""

import argparse
import datetime

from pathlib import Path

import torch
import evaluate
import numpy as np
import datasets


from transformers import TrainingArguments
from transformers import Trainer
from transformers import set_seed
from transformers import AutoTokenizer
from transformers import DataCollatorForTokenClassification
from transformers import AutoModelForTokenClassification

from meta_data import save_meta, get_meta, backup_readme


def loud_print(*args, **kwargs):
    print(">>> ", end="")
    print(*args, **kwargs)


def train(
    model, seed, epochs, train_ds, test_ds, cross_validation_index, push=True
):
    """
    args:
        model str: name of model to train (determines repo)
        seed int: seed to use throughout training
        epoch int: number of epochs to train for
        push bool: whether to push to hub
    returns:
        dict: evaluation scores for the last epoch of the model
    """
    loud_print(f"running model {model}")
    loud_print(f"training on seed {seed},")
    loud_print(f"for {epochs} epochs,")
    loud_print(f"cross-validation index {cross_validation_index}")
    loud_print(f"pushing {push}")
    meta_list = []
    # our texts never get longer than that, hence this is basically constant
    max_text_length = 700
    loud_print(f"working with max text length {max_text_length}")

    # =========================================
    # finding the device, for compatability
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    loud_print(f"on device {device}")
    # _________________________________________

    # =========================================
    # setting all the seeds for reproducability
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)
    # set seed for huggingface stuff (the most important seed)
    set_seed(seed)
    # _________________________________________

    # =========================================

    # raw_datasets = datasets.load_dataset(
    #     "Theoreticallyhugo/essays_SuG", model, trust_remote_code=True
    # )

    # get the dataset and ner_tags
    loud_print("using dataset:")
    loud_print(train_ds)

    label_names = train_ds.features["ner_tags"].feature
    loud_print("with labels:")
    loud_print(label_names)

    id2label = {i: label for i, label in enumerate(label_names.names)}
    label2id = {v: k for k, v in id2label.items()}
    # _________________________________________

    # =========================================
    # get the tokenizer
    # model_checkpoint = "bigscience/bloom-1b1"
    model_checkpoint = "bigscience/bloom-560m"
    loud_print(f"using the model: {model_checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint, add_prefix_space=True
    )

    output_dir = "bloom-" + model
    loud_print(f"saving to: {output_dir}")
    # _________________________________________

    # =========================================
    # got this strainght from the tuturial (what did it do again)
    def align_labels_with_tokens(labels, word_ids):
        new_labels = []
        current_word = None
        for word_id in word_ids:
            if word_id != current_word:
                # Start of a new word!
                current_word = word_id
                label = -100 if word_id is None else labels[word_id]
                new_labels.append(label)
            elif word_id is None:
                # Special token
                new_labels.append(-100)
            else:
                # Same word as previous token
                label = labels[word_id]
                # If the label is B-XXX we change it to I-XXX
                if label % 2 == 1:
                    label += 1
                new_labels.append(label)

        return new_labels

    # _________________________________________

    # =========================================
    # also strainght from the tutorial. add explanation
    def tokenize_and_align_labels(examples):
        tokenized_inputs = tokenizer(
            examples["tokens"], truncation=True, is_split_into_words=True
        )
        all_labels = examples["ner_tags"]
        new_labels = []
        for i, labels in enumerate(all_labels):
            word_ids = tokenized_inputs.word_ids(i)
            new_labels.append(align_labels_with_tokens(labels, word_ids))

        tokenized_inputs["labels"] = new_labels
        return tokenized_inputs

    # _________________________________________

    # =========================================
    # tokenize the dataset_dict. so we still got the sub datasets for train and test
    # tokenized_datasets = raw_datasets.map(
    #     tokenize_and_align_labels,
    #     batched=True,
    #     remove_columns=raw_datasets["train"].column_names,
    # )
    tokenized_train_ds = train_ds.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=train_ds.column_names,
    )
    tokenized_test_ds = test_ds.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=test_ds.column_names,
    )

    # _________________________________________

    # =========================================
    # pad the tokenized_datasets to get rid of the pesky padding message
    # purely cosmetic
    padding_length = max_text_length

    def pad_dataset(ds):
        for i in range(len(ds["input_ids"])):
            len_to_pad = padding_length - len(ds["input_ids"][i])

            ds["input_ids"][i] += [
                tokenizer.pad_token_id for _ in range(len_to_pad)
            ]
            ds["attention_mask"][i] += [0 for _ in range(len_to_pad)]
            ds["labels"][i] += [-100 for _ in range(len_to_pad)]
        return ds

        # for key in ['input_ids', 'attention_mask', 'labels']:

    # tokenized_datasets = tokenized_datasets.map(
    #     pad_dataset,
    #     batched=True,
    # )
    tokenized_train_ds = tokenized_train_ds.map(
        pad_dataset,
        batched=True,
    )
    tokenized_test_ds = tokenized_test_ds.map(
        pad_dataset,
        batched=True,
    )
    # _________________________________________

    # =========================================
    # still dont know what this really does
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    # _________________________________________

    # =========================================
    # overthink and compare this to whats used in ner-per-para
    metric = evaluate.load("poseval")

    def compute_metrics(eval_preds):
        logits, labels = eval_preds
        predictions = np.argmax(logits, axis=-1)

        # remove ignored index (special tokens) and convert to labels
        # liste an paragraphen
        # jeder paragraph ist eine liste an labels
        true_labels = [
            [id2label[l] for l in label if l != -100] for label in labels
        ]

        true_predictions = [
            [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]

        if model == "sep_tok_full_labels":
            for i in range(len(true_labels)):
                for j in range(len(true_labels[i])):
                    if true_labels[i][j] != "O":
                        true_labels[i][j] = true_labels[i][j][2:]

            for i in range(len(true_predictions)):
                for j in range(len(true_predictions[i])):
                    if true_predictions[i][j] != "O":
                        true_predictions[i][j] = true_predictions[i][j][2:]

        all_metrics = metric.compute(
            predictions=true_predictions, references=true_labels
        )
        save_meta(
            Path(output_dir), seed, epochs, all_metrics, cross_validation_index
        )
        meta_list.append(all_metrics)
        return all_metrics

    # _________________________________________

    # =========================================
    # get the raw model from the hub
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        id2label=id2label,
        label2id=label2id,
        # hidden_size=1024,
        # ignore_mismatched_sizes=True,
        # attention_window=max_text_length,
    )
    # _________________________________________

    # =========================================
    # run simple training for better maintainability

    args = TrainingArguments(
        output_dir=output_dir,
        overwrite_output_dir=True,
        auto_find_batch_size=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=epochs,
        weight_decay=0.01,
        push_to_hub=push,
        hub_strategy="all_checkpoints",
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_train_ds,
        eval_dataset=tokenized_test_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    loud_print("starting training")
    trainer.train()
    loud_print("done with training, and some uploading")

    trainer.create_model_card()
    backup_readme(Path(output_dir), seed, epochs)
    # trainer.save_model(output_dir)
    if push:
        now = datetime.datetime.now()
        url = trainer.push_to_hub(
            commit_message=f"trainer: training complete at {now}.",
            blocking=True,
        )
        loud_print("done with all the uploadeing")
    # collect_meta(Path(output_dir), seed)
    # _________________________________________
    # return get_meta(Path(output_dir), seed, epochs)

    # return the evaluation scores for all epochs as a list
    for i, result in enumerate(meta_list):
        save_meta(
            Path(output_dir), seed, i + 1, result, cross_validation_index
        )
    return meta_list


def get_args():
    """
    handles the argument parsing, when main.py is run from the commandline
    :return: the arguments parsed from the command line input
    """
    arg_par = argparse.ArgumentParser()
    arg_par.add_argument(
        "--model",
        "-m",
        # default=5,
        type=str,
        help="model to train",
    )
    arg_par.add_argument(
        "--epochs",
        "-e",
        default=5,
        type=int,
        help="number of epochs to train",
    )
    arg_par.add_argument(
        "--seed",
        "-s",
        default=42,
        type=int,
        help="seed to run the model with",
    )
    arg_par.add_argument(
        "--push",
        "-p",
        default=True,
        type=bool,
        nargs="?",
        const=False,
        help="add tag to DISABLE PUSHING. \n"
        + "pushing to hub is true per default",
    )

    args = arg_par.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    model_names = [
        "full_labels",
        "spans",
        "simple",
        "sep_tok",
        "sep_tok_full_labels",
        "all",
    ]

    assert args.model in model_names

    if args.model == "all":
        models = model_names[:-1]
    else:
        models = [args.model]

    for model in models:
        # TODO:
        #   10-fold cross-validation (see also next section on rounding behavior):
        #   The validation datasets are each going to be 10%:
        #   [0%:10%], [10%:20%], ..., [90%:100%].
        #   And the training datasets are each going to be the complementary 90%:
        #   [10%:100%] (for a corresponding validation set of [0%:10%]),
        #   [0%:10%] + [20%:100%] (for a validation set of [10%:20%]), ...,
        #   [0%:90%] (for a validation set of [90%:100%]).
        tests_ds = datasets.load_dataset(
            "Theoreticallyhugo/essays_SuG",
            model,
            split=[f"train[{k}%:{k+20}%]" for k in range(0, 100, 20)],
            trust_remote_code=True,
        )
        trains_ds = datasets.load_dataset(
            "Theoreticallyhugo/essays_SuG",
            model,
            split=[f"train[:{k}%]+train[{k+20}%:]" for k in range(0, 100, 20)],
            trust_remote_code=True,
        )

        for train_ds, test_ds, index in zip(
            trains_ds, tests_ds, range(len(tests_ds))
        ):
            train(
                model,
                args.seed,
                args.epochs,
                train_ds,
                test_ds,
                index,
                args.push,
            )
