import json
from pathlib import Path


def save_meta(folder: Path, seed: int, epochs: int, data: dict):
    folder = folder / "meta_data"
    folder.mkdir(parents=True, exist_ok=True)
    file = f"meta_s{seed}_e{epochs}.json"
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


# FIXME: fix before using
def collect_meta(output_dir: Path, seed: int):
    meta_dir = output_dir / "meta_data"
    for i, c_point in enumerate(output_dir.glob("checkpoint*")):
        print(c_point)
        i_file = c_point / "trainer_state.json"
        # FIXME: the way im retrieving the epoch here is very very unsafe
        o_file = Path.joinpath(meta_dir, f"meta_full_s{seed}_e{i}.json")
        o_file.write_text(i_file.read_text())
