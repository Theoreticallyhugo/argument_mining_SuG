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

from pathlib import Path
from typing import Dict, List, Optional

import datasets
import torch
from pipe_base import get_pipe
from tqdm import tqdm
from transformers import set_seed


def inference(
    texts: Optional[List[str]] = None,
    model: Optional[str] = "longformer-sep_tok",
):
    """

    args:
        texts List[str]: texts with injected separation tokens to label spans on
        model str: name or local path of huggingface model.
    returns:
        raw output of pipe
    """
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
    pipe = get_pipe(model)
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
