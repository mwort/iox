#!/usr/bin/env python3
"""iox - Check input and output to conditionally execute commands in parallel

Usage:
======
Single job:
$ iox -i input1 input2 -o output1 output2 --exec "command {input} {output}"

Single job rerun if inputs are newer than outputs:
$ iox -u -i input1 input2 -o output1 output2 --exec "command {input} {output}"

Parallel jobs:
$ iox -i input1/{d}.csv -o output/{d}.csv -d 2020 2021 2022 \\
    --exec "command {input} {output} && echo {d} > {output}"

Parallel jobs with wildcards from a file:
$ iox -i input1/{f}.csv -o output/{f}.csv --date file.txt \\
    --exec "echo {date} > {output}"

Parallel jobs with combinations of wildcards:
$ iox -i input_{a}/{d}.csv -o output/{a}_{d}.csv \\
    --combinations -a type1 type2 type2 -d 2020 2021 2022 \\
    --exec "command --year {d} -t {a} {input} {output}"
"""
import subprocess
from shutil import rmtree
import sys
import itertools
import string
from pathlib import Path
from typing import List, Dict, Any
from functools import partial


class Paths(list):
    def __init__(self, *paths):
        args = [Path(i) for i in paths]
        super().__init__(args)

    def __getattr__(self, name):
        """Return method if attr is callable else just the attribute."""
        attributes = [getattr(item, name) for item in self]
        if all(callable(i) for i in attributes):

            def method(*args, **kwargs):
                out = [item(*args, **kwargs) for item in attributes]
                # coerce to Paths if all elements are Paths
                if all(isinstance(i, Path) for i in out):
                    return Paths(*out)
                return out

            return method
        else:
            # coerce to Paths if all elements are Paths
            if all(isinstance(i, Path) for i in attributes):
                return Paths(*attributes)
            return attributes

    def __str__(self):
        return " ".join(map(str, self))

    def non_existing(self):
        return Paths(*[p for p in self if not p.exists()])

    def existing(self):
        return Paths(*[p for p in self if p.exists()])

    def remove(self):
        for path in self:
            if path.is_dir():
                rmtree(path)
            elif path.is_file():
                path.unlink()
        return

    def modification_time(self):
        return [p.stat().st_mtime for p in self]


def safe_format(s: str, **kwargs) -> str:
    class SafeFormatter(dict):
        def __missing__(self, key):
            return "{" + key + "}"
    return s.format_map(SafeFormatter(**kwargs))


def format_wildcards(
    wildcards: Dict[str, List[str]],
    paths: List[str],
    combinations: bool = False,
    allow_missing: bool = False,
) -> List[str]:
    # check that all wildcard lists are of the same length
    wclen = [len(v) for v in wildcards.values()]
    if not combinations:
        assert len(set(wclen)) == 1, "All wildcards must be of the same length."
    if not allow_missing:
        # check that the all wildcard keys appear in the check_io output argument
        for o in paths:
            ostr, keys = str(o), wildcards.keys()
            notin = " ".join([k for k in keys if f"{{{k}}}" not in ostr])
            errmsg = f"All wildcards ({notin}) must appear in {o}"
            if len(notin) > 0:
                raise KeyError(errmsg)
    # expand wildcards
    expander = itertools.product if combinations else zip
    expand = list(expander(*wildcards.values()))

    out = [
        [safe_format(str(i), **dict(zip(wildcards.keys(), wc))) for i in paths]
        for wc in expand
    ]
    return out


def check_io(
    input: Paths = Paths(),
    output: Paths = Paths(),
    exec: List[str] = [],
    force: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
    quiet: bool = False,
    update: bool = False,
    **kwargs,
) -> None:
    # make sure all Paths and convert lists to space-separated strings
    input = Paths(*input)
    output = Paths(*output)
    exec = safe_format(" ".join(exec), input=input, output=output)

    # Print for debugging
    verbose = print if verbose else lambda *a, **k: None
    verbose(f"Input: {input}")
    verbose(f"Output: {output}")
    verbose(f"Exec: {exec}")

    # job info to return
    job_info = {"input": input, "output": output, "exec": exec}

    # Check if all input paths exist
    iein = input.non_existing()
    if len(iein) > 0:
        raise FileNotFoundError(f"Input paths do not exist: {iein} - exiting.")

    # check if outputs need to be updated
    if update:
        eout = output.existing()
        if len(eout) > 0 and min(eout.modification_time()) < max(input.modification_time()):
            verbose(f"Updating: {eout}")
            force = True
    # Check if all output paths exist
    if output and all(output.exists()) and not force:
        print("All output paths exist, nothing to be done. Exiting.")
        return job_info

    if dry_run:
        eout = output.existing()
        if eout:
            print(f"Would clean and recreate: {eout}")
        print(f"Would execute: {exec}")
        print("Dry run. Exiting.")
        return job_info

    # Clean output and make sure all directories exist
    output.remove()
    output.parent.mkdir(parents=True, exist_ok=True)

    # Execute the command
    stdout = open("/dev/null", "w") if quiet else sys.stdout
    subprocess.run(exec, shell=True, check=True, stdout=stdout)

    # Exit if not all output paths exist
    ieout = output.non_existing()
    if len(ieout) > 0:
        errmsg = f"Not all outputs exist after running {exec}.\n{ieout}\nExiting."
        raise FileNotFoundError(errmsg)

    return job_info


def check_io_parallel(
    wildcards: Dict[str, str],
    input: List[Path],
    output: List[Path],
    exec: List[str],
    verbose: bool = False,
    jobs: int = 1,
    combinations: bool = False,
    **kwargs,
) -> None:
    import multiprocessing as mp

    if verbose:
        print(f"Input: {input}")
        print(f"Output: {output}")
        print(f"Wildcards: {wildcards}")

    inputs = format_wildcards(wildcards, input, combinations, allow_missing=True)
    outputs = format_wildcards(wildcards, output, combinations)
    execs = format_wildcards(wildcards, exec, combinations, allow_missing=True)

    if verbose:
        print("Parallel jobs:")
        for i, o, e in zip(inputs, outputs, execs):
            print(f"-i {i} -o {o} --exec {e}")

    def update_progress():
        nonlocal completed_jobs
        completed_jobs += 1
        percentage = (completed_jobs / total_jobs) * 100
        prec = len(str(total_jobs))
        return f"[{completed_jobs:0{prec}d}/{total_jobs} ({percentage:3.{prec-1}f}%)] "

    def report_complete(result):
        progress = update_progress()
        print(progress + result["exec"])

    def report_error(error):
        progress = update_progress()
        print(progress + str(error))

    total_jobs = len(inputs)
    completed_jobs = 0
    worker_func = partial(check_io, verbose=verbose, **kwargs)

    pool = mp.Pool(jobs)
    for i, o, e in zip(inputs, outputs, execs):
        pool.apply_async(
            worker_func,
            args=(i, o, e),
            callback=report_complete,
            error_callback=report_error,
        )
    pool.close()
    pool.join()
    return


def parse_wildcards(unknown_args: List[str]) -> Dict[str, str]:
    # Convert unknown arguments to a dictionary
    extra_args = {}
    for arg in unknown_args:
        if arg.startswith("--") or arg.startswith("-"):
            key = arg.lstrip("--").lstrip("-")
            extra_args[key] = []
        else:
            extra_args[key].append(arg)
    # if any argument only has one value, try to treat it as a file and open it
    # and replace the argument with the lines of the file
    for key, value in extra_args.items():
        if len(value) == 1:
            with open(value[0]) as f:
                extra_args[key] = f.read().splitlines()
    return extra_args


def __main__() -> None:
    """Command line interface for iox."""
    import argparse

    arguments = [
        {
            "names": ("-i", "--input"),
            "args": {"nargs": "+", "required": False, "type": Path, "default": Paths()},
            "help": "Input path/pattern",
        },
        {
            "names": ("-o", "--output"),
            "args": {"nargs": "+", "required": True, "type": Path},
            "help": "Output path/pattern",
        },
        {
            "names": ("-x", "--exec",),
            "args": {"nargs": argparse.REMAINDER, "required": True},
            "help": "Command to execute",
        },
        {
            "names": ("-f", "--force"),
            "args": {"action": "store_true"},
            "help": "Force execution even if output paths exist",
        },
        {
            "names": ("-n", "--dry-run"),
            "args": {"action": "store_true"},
            "help": "Perform a dry run without executing the command",
        },
        {
            "names": ("-j", "--jobs"),
            "args": {"type": int, "default": 1},
            "help": "Number of parallel jobs",
        },
        {
            "names": ("-c", "--combinations"),
            "args": {"action": "store_true"},
            "help": "Create all combinations of wildcards",
        },
        {
            "names": ("-u", "--update"),
            "args": {"action": "store_true"},
            "help": "Update output files if input files are newer",
        },
        {
            "names": ("-q", "--quiet"),
            "args": {"action": "store_true"},
            "help": "Suppress output from the command",
        },
        {
            "names": ("-v", "--verbose"),
            "args": {"action": "store_true"},
            "help": "Enable verbose output",
        },
        {
            "names": ("-h", "--help"),
            "args": {"action": "help"},
            "help": "Show help message and exit",
        },
    ]

    parser = argparse.ArgumentParser(
        description=__doc__,
        add_help=False,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    for arg in arguments:
        parser.add_argument(*arg["names"], help=arg["help"], **arg["args"])
    args, unknown_args = parser.parse_known_args()
    wildcards = parse_wildcards(unknown_args)

    if wildcards:
        check_io_parallel(wildcards, **vars(args))
    else:
        check_io(**vars(args))


if __name__ == "__main__":
    __main__()
