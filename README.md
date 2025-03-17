# Setup

## Setup script

We provide a setup script that clones the TCPD repository and installs the dependencies.

```bash
./setup.sh
```

## Setup the environment

We use uv to manage the dependencies through the `pyproject.toml` file. You can install the dependencies by running:

```bash
# Install uv if you don't have it
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install the dependencies
uv sync
```



## Downloading the data

We use the datasets from the TCPD repository, which should be downloaded into the home directory. You don't need to run it again if you have already run the setup script.

```bash
git clone git@github.com:alan-turing-institute/TCPD.git
```



# Running the ruptures benchmarks

We provide a notebook to run the benchmarks using the ruptures library in `notebooks/ruptures_benchmarks.ipynb`.

