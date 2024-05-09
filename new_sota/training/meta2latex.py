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
script for reading the meta data from the meta_data directory, and writing the 
LaTeX code for a table, containing the macro-f1 scores for all models and 
their epochs
"""

import json
from pathlib import Path


def read_meta(meta_dir: Path):
    meta_dict = {}
    for file in meta_dir.iterdir():
        meta_dict[file.name.split(".")[0].replace("_", " ")] = json.loads(
            file.read_text()
        )

    return meta_dict


def json2latex(meta_dict: dict):
    headline = [
        "Makro-f1",
        "full labels",
        "spans",
        "simple",
        "sep tok full labels",
        "sep tok",
    ]
    body = [
        [] for _ in range(len(meta_dict[list(meta_dict.keys())[0]]["epochs"]))
    ]
    for i in range(len(body)):
        body[i] = [str(i + 1).rjust(2, "0")] + [
            "" for _ in range(len(headline) - 1)
        ]

    for model in headline[1:]:
        for epoch, data in meta_dict[model]["epochs"].items():
            body[int(epoch) - 1][headline.index(model)] = str(
                round(data["macro avg"]["f1-score"], 3)
            )

    out_list = (
        [
            "\\begin{table}[!h]",
            "  \\centering",
            "  \\begin{NiceTabular}{c||c|c|c|c|c} ",
            "    \\CodeBefore",
            "      \\rowcolors{2}{gray!25}{white}",
            "    \\Body",
        ]
        + [
            "    "
            + " & ".join(["\\textbf{" + s + "}" for s in headline])
            + "\\\\"
        ]
        + [
            "    \\hline",
            "    \\hline",
        ]
    )
    out_list.extend(
        [
            f"    {body[0][0]} & "
            + " & ".join([s.ljust(5, "0") for s in body[0][1:]])
            + "\\T\\\\"
        ]
    )
    for l in body[1:]:
        out_list.extend(
            [
                f"    {l[0]} & "
                + " & ".join([s.ljust(5, "0") for s in l[1:]])
                + "\\\\"
            ]
        )
    out_list += [
        "  \\end{NiceTabular}",
        "  \\vfill",
        "  \\caption{5-fold cross-validation of the macro-f1}",
        "  \\label{tab:epoch_f1}",
        "\\end{table}",
    ]

    for i in out_list:
        print(i)
    input()


if __name__ == "__main__":
    json2latex(read_meta(Path("meta_data")))
