# iox
Check input and output to conditionally execute commands in parallel

## Usage:
Single job, will only run if any output does not exist, try -n for a dry run:
```
$ iox -i input1 input2 -o output1 output2 --exec "command {input} {output}"
```
Single job rerun if inputs are newer than outputs:
```
$ iox -u -i input1 input2 -o output1 output2 --exec "command {input} {output}"
```
Parallel jobs:
```
$ iox -i input1/{d}.csv -o output/{d}.csv -d 2020 2021 2022 \\
    --exec "command {input} {output} && echo {d} > {output}"
```
Parallel jobs with wildcards from a file:
```
$ iox -i input1/{f}.csv -o output/{f}.csv --date file.txt \\
    --exec "echo {date} > {output}"
```
Parallel jobs with combinations of wildcards:
```
$ iox -i input_{a}/{d}.csv -o output/{a}_{d}.csv \\
    --combinations -a type1 type2 type2 -d 2020 2021 2022 \\
    --exec "command --year {d} -t {a} {input} {output}"
```
Jobs in a pipeline, input/output paths are passed through:
```
$ echo f1 f2 f3 \
    | iox -o f4 -x "echo {input} > {output}" \
    | iox -o summary -x "cat {input} > {output}"
```


## Testing/releasing

```
vim iox.py  # update version string
git commit # make sure workdir is clean
git tag v$(./iox.py --version)
./iox.py -o venv -x virtualenv {output}
source venv/bin/activate
pip install build twine pytest
pytest tests.py
./iox.py -i *.py -o dist -u -x python -m build
python -m twine upload dist/*
```
