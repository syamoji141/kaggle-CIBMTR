# CIBMTR - Equity in post-HCT Survival Predictions

<img src="img/syamoji141 - CIBMTR - Equity in post-HCT Survival Predictions.png" width="70%">

## Build Environment
### 1. install [uv](https://github.com/astral-sh/uv)

```bash
# On macOS and Linux.
$ curl -LsSf https://astral.sh/uv/install.sh | sh

# On Windows.
$ powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# With pip.
$ pip install uv
```

### 2. create virtual enviroment

if you are using Linux, Pytorch(cu121) will be installed, and if you are using Windows or MacOS, Pytorch(cpu) will be installed.  
Please refer to the following link for details.  
https://docs.astral.sh/uv/guides/integration/pytorch/#configuring-accelerators-with-environment-markers

```
uv sync
```

### 3. Activate virtual environment

```zsh
. .venv/bin/activate
```

### 4. Preprocess data

```zsh
cd syamoji141
python -m run.prepare_data
```

### 5. Train model

```zsh
python -m run.train model=tabm
```

or

```zsh
python -m run.torchsort_pairwise_loss_lgbm_model
```