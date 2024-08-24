from pathlib import Path

from full_pipe import main


def test_rebuilt_filestructure():
    main(
        Path("./test_in_structure"),
        Path("./test_out_structure"),
        "aa",
        "bb",
        False,
        True,
    )
    assert Path("./test_out_structure/").is_dir()
    assert Path("./test_out_structure/S.txt").is_file()
    assert Path("./test_out_structure/a/").is_dir()
    assert Path("./test_out_structure/a/AA.txt").is_file()
    assert Path("./test_out_structure/b/").is_dir()
    assert Path("./test_out_structure/b/bb/").is_dir()
    assert Path("./test_out_structure/b/bb/BBB.txt").is_file()


def test_single_file_dry():
    main(
        Path("./test_in_file/A.txt"),
        Path("./test_out_file/"),
        "aa",
        "bb",
        False,
        True,
    )
    assert Path("./test_out_file/").is_dir()
    assert Path("./test_out_file/A.txt").is_file()
    assert Path("./test_out_file/A.ann").is_file()
