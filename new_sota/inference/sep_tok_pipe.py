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

import torch
import datasets

from typing import Dict, List
from tqdm import tqdm
from transformers import set_seed

from pipe_base import get_pipe


def inference(texts: List[str] = None):
    # =========================================
    # setting all the seeds for reproducability
    seed = 42
    # Set the seed for general torch operations
    torch.manual_seed(seed)
    # Set the seed for CUDA torch operations (ones that happen on the GPU)
    torch.cuda.manual_seed(seed)
    # set seed for huggingface stuff (the most important seed)
    set_seed(seed)
    # _________________________________________

    # =========================================
    # get the pipeline
    model_name = "longformer-sep_tok"

    pipe = get_pipe(model_name)
    # -----------------------------------------

    # =========================================
    # get the dataset and ner_tags
    if texts is None:
        ds = datasets.load_dataset(
            "Theoreticallyhugo/essays_SuG", "sep_tok", trust_remote_code=True
        )
        texts = ds["train"]["text"]
    # -----------------------------------------

    print("running inference")
    results = [
        out
        for out in tqdm(
            pipe(
                texts,
                batch_size=8,
            )
        )
    ]
    return results


if __name__ == "__main__":
    results = inference()
