# _Performance Model_

_This is the hardware functional and performance model for the SQ-TC Hardware Architecture_.

## _Build_

_Install Requirements_

```
# macOS
brew install cmake ninja llvm

# Linux
sudo apt-get install cmake ninja-build llvm clang
```


_Install Packages & Build_

```
cd models/perf_model
pip install -r requirements.txt
pip install -e .
```

## _Running_

```
perf_model
```
