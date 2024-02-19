"""
the goal is to do NER in a single step on the entire essay
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


def train(seed, epochs, push=True):
    loud_print(f"training on seed {seed},")
    loud_print(f"for {epochs} epochs,")
    loud_print(f"pushing {push}")
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
    raw_datasets = datasets.load_dataset(
        "Theoreticallyhugo/essays_SuG", "full_labels", trust_remote_code=True
    )
    # get the dataset and ner_tags
    loud_print("using dataset:")
    loud_print(raw_datasets)

    label_names = raw_datasets["train"].features["ner_tags"].feature
    loud_print("with labels:")
    loud_print(label_names)

    id2label = {i: label for i, label in enumerate(label_names.names)}
    label2id = {v: k for k, v in id2label.items()}
    # _________________________________________

    # =========================================
    # get the tokenizer
    model_checkpoint = "allenai/longformer-base-4096"
    loud_print(f"using the model: {model_checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(
        model_checkpoint, add_prefix_space=True
    )

    output_dir = "longformer-full_labels"
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
    tokenized_datasets = raw_datasets.map(
        tokenize_and_align_labels,
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
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

    tokenized_datasets = tokenized_datasets.map(
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

        all_metrics = metric.compute(
            predictions=true_predictions, references=true_labels
        )
        save_meta(Path(output_dir), seed, epochs, all_metrics)
        return all_metrics

    # _________________________________________

    # =========================================
    # get the raw model from the hub
    model = AutoModelForTokenClassification.from_pretrained(
        model_checkpoint,
        id2label=id2label,
        label2id=label2id,
        attention_window=max_text_length,
    )
    # _________________________________________

    # =========================================
    # run simple training for better maintainability

    args = TrainingArguments(
        output_dir,
        overwrite_output_dir=True,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        num_train_epochs=epochs,
        weight_decay=0.01,
        push_to_hub=push,
        seed=seed,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )
    loud_print("starting training")
    trainer.train()
    loud_print("done with training, and some uploading")

    trainer.create_model_card()
    # trainer.save_model(output_dir)
    if push:
        now = datetime.datetime.now()
        url = trainer.push_to_hub(
            commit_message=f"trainer: training complete at {now}.",
            blocking=True,
        )
        loud_print("done with all the uploadeing")
    backup_readme(Path(output_dir), seed, epochs)
    # collect_meta(Path(output_dir), seed)
    # _________________________________________
    return get_meta(Path(output_dir), seed, epochs)


def get_args():
    """
    handles the argument parsing, when main.py is run from the commandline
    :return: the arguments parsed from the command line input
    """
    arg_par = argparse.ArgumentParser()
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
    train(args.seed, args.epochs, args.push)
