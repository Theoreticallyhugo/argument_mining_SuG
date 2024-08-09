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

import logging
from pathlib import Path

import torch
from transformers import pipeline


def get_pipe(model: str):
    """
    get pipe for huggingface model from huggingface repo or local model

    args:
        model str: name or local path of huggingface model.
    returns:
        huggingface pipe
    """
    default_user = "Theoreticallyhugo"

    if Path(model).exists():
        logging.info(f"trying to load model from local path {model}")
    elif model.count("/") == 0:
        logging.info(
            f"trying to load model {model} from huggingface with default user {default_user}"
        )
        model = f"{default_user}/{model}"
    elif model.count("/") == 1:
        logging.info(f"trying to load model {model} from huggingface.")
    else:
        raise ValueError(
            "pipe requires either local path or model name. "
            + "neither was provided."
        )

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    return pipeline("token-classification", model=model, device=device)
