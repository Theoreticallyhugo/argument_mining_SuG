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
from transformers import set_seed
from pipe_provider import get_pipe
from data import data_preparator
from tqdm import tqdm
from transformers.pipelines.pt_utils import KeyDataset

set_seed(42)

# =========================================
# get the pipeline
# TODO make this more general
model_name = "bert-finetuned-ner-essays-para"

pipe = get_pipe(model_name, True)
# -----------------------------------------


# =========================================
# get the dataset and ner_tags
dp = data_preparator()
essays = dp.get_essay_list(os.path.join(os.pardir, "data", "essay.json"))
para_ds = dp.get_paragraphs_dataset(essays)
# -----------------------------------------

print("running inference")
results = [
    out
    for out in tqdm(
        pipe(
            KeyDataset(para_ds, "paragraphs"),
            batch_size=8,
        )
    )
]
print(results)
