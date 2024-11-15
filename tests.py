import os
import pytest

from iox import *


@pytest.fixture
def temp_input_files():
    files = Paths("temp1.txt", "temp2.txt")
    for file in files:
        with open(file, "w") as f:
            f.write("Temporary file for testing.")
    yield files
    files.remove()

@pytest.fixture
def temp_output_files():
    files = Paths("output1.txt", "output2.txt")
    yield files
    files.remove()


def test_check_non_existing(temp_input_files):
    # Test when all files exist
    assert temp_input_files.non_existing() == []
    # Test when some files do not exist
    assert Paths("nonexistent.txt", *temp_input_files).non_existing() == Paths("nonexistent.txt")


def test_check_io_input_files(temp_input_files):
    # Test check_io when all input files exist
    check_io(
        temp_input_files,
        ["tmp"],
        ["echo", "Hello, World!"],
        force=False,
        dry_run=True,
        verbose=True,
    )
    # Test check_io when not all input files exist
    with pytest.raises(FileNotFoundError) as excinfo:
        check_io(
            temp_input_files + ["tmp"],
            ["tmp"],
            ["echo", "Hello, World!"],
            force=False,
            dry_run=True,
            verbose=True,
        )


def test_check_io_force(temp_input_files):
    # Test check_io with force execution
    mt = temp_input_files[1].lstat().st_mtime
    check_io(
        [temp_input_files[0]],
        [temp_input_files[1]],
        ["touch", "{output}"],
        force=True,
        dry_run=False,
        verbose=True,
    )
    assert temp_input_files[1].exists()
    assert temp_input_files[1].lstat().st_mtime > mt

def test_check_io_dry_run(temp_input_files):
    ieout = Paths("tmp")
    # Test check_io with dry run
    check_io(
        temp_input_files,
        ieout,
        ["cat", "{input}", "{output}"],
        force=True,
        dry_run=True,
        verbose=True,
    )
    assert ieout.non_existing() == ieout


def test_check_io_missing_output_files(temp_input_files, temp_output_files):
    # Test check_io when output files are missing after execution
    with pytest.raises(FileNotFoundError) as excinfo:
        check_io(
            temp_input_files,
            temp_output_files,
            ["echo", "Hello, World! {input}"],
            force=False,
            dry_run=False,
            verbose=True,
        )
    assert temp_output_files.non_existing() == temp_output_files


def test_check_io_update_output_files(temp_input_files, temp_output_files):
    # create output files
    check_io(temp_input_files, temp_output_files, ["touch", "{output}"])
    assert all(temp_output_files.exists())
    mt = max(temp_output_files.modification_time())

    # force touch input files
    check_io(output=temp_input_files, exec=["touch", "{output}"], force=True)
    assert max(temp_input_files.modification_time()) > mt
    mt = max(temp_output_files.modification_time())

    # update output files
    check_io(temp_input_files, temp_output_files, ["touch", "{output}"], update=True)
    assert max(temp_output_files.modification_time()) > mt


def test_check_io_parallel(temp_input_files, temp_output_files):
    wildcards = {"d": [1, 2]}
    kwargs = dict(
        input=["temp{d}.txt"],
        output=["output{d}.txt"],
        exec=["echo {input} {d} > {output}"],
        verbose=True,
        jobs=2,
    )
    check_io_parallel(wildcards, dry_run=True, **kwargs)
    assert not any(temp_output_files.exists())

    check_io_parallel(wildcards, **kwargs)
    assert all(temp_output_files.exists())

    with pytest.raises(KeyError) as excinfo:
        check_io_parallel(dict(date=[1, 2]), **kwargs)

    with pytest.raises(AssertionError) as excinfo:
        kwargs["output"] = ["{date}/{d}.txt"]
        check_io_parallel(dict(date=[1], **wildcards), **kwargs)