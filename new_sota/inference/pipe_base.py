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

from transformers import pipeline
import torch
from typing import Optional


def get_pipe(
    local_path: Optional[Path] = None,
    model: Optional[str] = None,
    user="Theoreticallyhugo",
):
    # TODO: local_path
    """
    get pipe for huggingface model from huggingface repo or local model
    args:
        local_path: path to local model
        model: model to load from huggingface repo
        user: whose model it is
    """
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )

    print("loading pipeline")
    return pipeline(
        "token-classification", model=f"{user}/{model}", device=device
    )
