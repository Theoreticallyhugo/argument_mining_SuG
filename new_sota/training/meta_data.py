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

import json
from pathlib import Path
from tqdm import tqdm


def save_meta(
    folder: Path,
    seed: int,
    epochs: int,
    data: dict,
    cross_validation_index: int,
):
    folder = folder / "meta_data"
    folder.mkdir(parents=True, exist_ok=True)
    file = f"meta_s{seed}_e{epochs}_cvi{cross_validation_index}.json"
    with open(folder / file, "w") as w:
        json.dump(data, w)


def get_meta(folder: Path, seed: int, epochs):
    folder = folder / "meta_data"
    file = f"meta_s{seed}_e{epochs}.json"
    with open(folder / file, "r") as r:
        return json.load(r)


def backup_readme(output_dir: Path, seed: int, epochs: int):
    Path.joinpath(
        output_dir, "meta_data", f"README_s{seed}_e{epochs}.md"
    ).write_text(Path.joinpath(output_dir, "README.md").read_text())


def collect_cross_validation():
    print("collecting_cross_validation")
    cwd = Path.cwd()
    models_list = []
    for dir in tqdm(list(cwd.glob("longformer-*"))):

        if not dir.is_dir():
            continue
        # print(dir.name)

        model_name = dir.name.split("-")[1]
        model_dict = {"name": model_name, "epochs": {}}

        meta_dir = dir / "meta_data"

        for file in meta_dir.iterdir():

            if file.is_dir():
                continue

            filename = file.name.split(".")[0]
            if len(filename.split("_")) != 4:
                continue

            epoch = filename.split("_")[2][1:]
            cross_validation_index = filename.split("_")[3][3:]

            with open(file, "r") as r:
                meta_data = json.load(r)

            model_dict["epochs"][epoch] = model_dict["epochs"].get(epoch, {})
            model_dict["epochs"][epoch][cross_validation_index] = meta_data

            # print(
            #     f"model: {model_name}, at epoch {epoch}, with cvi {cross_validation_index}"
            # )

        models_list.append(model_dict)

    # input(json.dumps(models_list, sort_keys=True, indent=4))
    return models_list


def add_up_dicts(dict1, dict2):
    """
    WARNING: BOTH DICTS NEED TO HAVE THE EXACT SAME STRUCTURE.

    adds up the values (must be int or float) of two dictionaries
    """
    out_dict = {}
    for key, value in dict1.items():
        if isinstance(value, dict):
            out_dict[key] = add_up_dicts(dict1[key], dict2[key])
        else:
            out_dict[key] = dict1[key] + dict2[key]
    return out_dict


def divide_dict(input_dict, divisor):
    out_dict = {}
    for key, value in input_dict.items():
        if isinstance(value, dict):
            out_dict[key] = divide_dict(input_dict[key], divisor)
        else:
            out_dict[key] = input_dict[key] / divisor
    return out_dict


def calculate_cross_validation():
    """
    structure of the output:
    [
        {
            "name": "name_of_the_model",
            "epochs":
                {
                    "int_of_epoch":
                        {
                            "name_of_label":
                                {
                                    "f1-score": float,
                                    "precision": float,,
                                    "recall": float,
                                    "support": float,
                                },
                            "name_of_another_labe": same as before,
                            "accuracy": float,
                            "macro avg":
                                {
                                    "f1-score": float,
                                    "precision": float,
                                    "recall": float,
                                    "support": float,
                                },
                            "weighted avg":
                                {
                                    "f1-score": float,
                                    "precision": float,
                                    "recall": float,
                                    "support": float,
                                }
                        }
                }
        }
    ]
    """
    models_list = collect_cross_validation()

    print("calculating_cross_validation")
    output_list = []
    for i in tqdm(range(len(models_list))):
        output_list.append({"name": models_list[i]["name"], "epochs": {}})
        output_list[-1]["epochs"] = {}
        for epoch, data in models_list[i]["epochs"].items():
            epoch = epoch.rjust(2, "0")
            cross_validation = {}
            for cvi, fold_data in data.items():
                if cross_validation == {}:
                    cross_validation = fold_data
                    continue
                # add up all scores
                cross_validation = add_up_dicts(cross_validation, fold_data)

            # here we got the values to save!
            cross_validation = divide_dict(cross_validation, len(data))
            output_list[-1]["epochs"][epoch] = cross_validation
    return output_list


def save_cross_validation():
    cv = calculate_cross_validation()
    output_dir = Path("meta_data")
    output_dir.mkdir(exist_ok=True)
    print("saving meta_data")
    for model in tqdm(cv):
        out_file = output_dir / f"{model['name']}.json"
        with open(out_file, "w") as w:
            json.dump(model, w, sort_keys=True, indent=4)


if __name__ == "__main__":
    save_cross_validation()
