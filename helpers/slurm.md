# Running Cellpose Training on Pitt CRC (SLURM)

## Environment Setup (do once)

The recommended CRC approach is a custom miniconda installation on `/ix1/` exposed as an Lmod module. This keeps everything off the 75 GB home quota and makes the SLURM script load it like any other module.

### 1. Install miniconda to /ix1/ (not home)
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
# when prompted for install path, enter:
# /ix1/<group>/<username>/custom_miniconda
# when asked to run conda init — say NO
```

### 2. Install mamba and create the cellpose environment
```bash
/ix1/<group>/<username>/custom_miniconda/bin/conda install mamba -n base -c conda-forge
/ix1/<group>/<username>/custom_miniconda/bin/conda create -n cellpose python=3.10
source /ix1/<group>/<username>/custom_miniconda/bin/activate cellpose
```

### 3. Install PyTorch with a CUDA version available on the cluster
The default `pip install cellpose` pulls PyTorch built for CUDA 13.0 which is not available on CRC (max: 12.9). Install PyTorch for CUDA 12.4 first, then cellpose:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install cellpose
pip install yaml
```

Verify the right build is installed:
```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
# expect: 2.x.x+cu124 and 12.4
```

> `torch.cuda.is_available()` will return `False` on the login node (no GPU) — this is expected. It returns `True` on GPU compute nodes.

### 4. Create a custom Lmod module for the environment
```bash
mkdir ~/modulefiles
cp /ihome/crc/modules/Core/python/ondemand-jupyter-python3.10.lua ~/modulefiles/cellpose_env.lua
# edit cellpose_env.lua: change local package_root to the env, not base miniconda:
# /ix1/<group>/<username>/custom_miniconda/envs/cellpose
```

### 5. Test loading it
```bash
module use ~/modulefiles
module load cellpose_env
module load cuda/12.4.1
which python
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

Once working, the SLURM script just needs:
```bash
module use ~/modulefiles
module load cellpose_env
module load cuda/12.4.1
```

> Do **not** run `conda init` — it modifies `~/.bashrc` in ways that break other CRC modules.

---

## Cluster: gpu | Partition: a100

### 0. Move training data to /ix1 (do once)
`$HOME` (`/ihome`) has a 75 GB quota and is for config files — not compute data.
Training data belongs in `/ix1/<group>/cellpose/` (5 TB free, designed for job staging).

```bash
# on the cluster
mkdir -p /ix1/<group>/cellpose/train /ix1/<group>/cellpose/test

# re-upload from local machine, or move on cluster:
mv ~/cellpose/train/* /ix1/<group>/cellpose/train/
mv ~/cellpose/test/*  /ix1/<group>/cellpose/test/
```

Then edit `run_trainer.slurm` and set `DATA_SRC="/ix1/<group>/cellpose"`.

### 1. Upload scripts to cluster
```bash
sftp [username]@h2p.crc.pitt.edu
put cellpose_git/helpers/trainer_slurm.py  cellpose/trainer_slurm.py
put cellpose_git/helpers/trainer.yaml      cellpose/trainer.yaml
put cellpose_git/helpers/run_trainer.slurm cellpose/run_trainer.slurm
put cellpose_git/helpers/submit.sh         cellpose/submit.sh
put cellpose_git/helpers/.env.example      cellpose/.env.example
```

### 2. Before submitting

- **Email** (`.env`, gitignored — never committed):
  ```bash
  cp ~/cellpose/.env.example ~/cellpose/.env
  # edit ~/cellpose/.env:  PITT_EMAIL=your_pitt_id@pitt.edu
  ```
  `submit.sh` sources `.env` and passes `--mail-user` to `sbatch`. (SBATCH
  directives are parsed at submit time and can't read shell vars, so the
  address has to come in on the command line.)

- **Data path**: set `DATA_SRC="/ix1/<group>/cellpose"` in `run_trainer.slurm`.

- **Hyperparameters** (`trainer.yaml`): `weight_decay`, `learning_rate`,
  `n_epochs`, `batch_size`, `model_name`. Edit the YAML rather than the
  Python — it's the single source of truth and the trainer logs the
  resolved values into the `.out`. Override its location with
  `CELLPOSE_TRAIN_CONFIG`; `run_trainer.slurm` also freezes a copy to
  `run_logs/cellpose_train_<jobid>.config.yaml`.

### 3. Submit the job
```bash
./submit.sh                       # sources .env, runs sbatch
# (any extra args are forwarded to sbatch)
```

### 4. Model output
Saved to `/ix1/<group>/cellpose/models/` after job completes.

---

## Status Commands

```bash
# Check job queue
squeue -M gpu -u $USER

# Check estimated start time
squeue -M gpu --start -u $USER

# Check partition availability (look for 'mix' or 'idle' nodes)
sinfo -M gpu -p a100,l40s -o "%P %D %t %C"

# Count pending jobs per partition (gauge competition)
squeue -M gpu -p a100,l40s -t PD --noheader | awk '{print $2}' | sort | uniq -c

# Watch live logs
cat cellpose_train_<jobid>.out

# Cancel a job
scancel <jobid>

# Check home and /ix1 storage quotas
crc-quota
```

---

## Accessing Data and Scratch

### Permanent data location
Training data lives in `/ix1/<group>/cellpose/` and persists between jobs:
```bash
ls /ix1/$(id -gn)/cellpose/
# train/   test/   models/
```

### Accessing scratch while a job is running
During a job, data is copied to node-local scratch at `/scratch/slurm-<JOBID>/`. To inspect it, SSH directly to the compute node (visible in `squeue` output or the job log):
```bash
# Find the node your job is on
squeue -M gpu -u $USER -o "%i %R %N"

# SSH to it (no password needed from login node)
ssh gpu-n35.crc.pitt.edu

# Browse scratch
ls /scratch/slurm-<JOBID>/train/
ls /scratch/slurm-<JOBID>/test/
```

Scratch is automatically cleaned up when the job exits. The EXIT trap in the SLURM script copies trained models back to `/ix1/<group>/cellpose/models/` before that happens.

---

## Notes

| Topic | Detail |
|---|---|
| GPU options | `a100` (40 GB), `a100_nvlink` (80 GB), `l40s` (48 GB) |
| Partition choice | Check `sinfo` — prefer partition with most `mix`/`idle` nodes |
| Data location | `/ix1/<group>/cellpose/` — 5 TB free, designed for compute job staging |
| Fast I/O | SLURM script copies data to `$SLURM_SCRATCH` (local NVMe) before training |
| Batch size | Was `1` on local 8 GB GPU; try `8` on A100 40 GB |
| Time limit | Default in script: 6h (`--qos=long` allows up to 6 days) |
| Python env | Custom miniconda on `/ix1/` loaded as Lmod module (`cellpose_env`) |
| PyTorch | Must install with `--index-url https://download.pytorch.org/whl/cu124` — default pip install pulls cu130 which is unavailable on CRC |
| CUDA module | `cuda/12.4.1` matches `torch+cu124`; cluster max is `cuda/12.9.0` |
| Add trained model | `python -m cellpose --add_model "/path/to/model"` |
| Hyperparameters | `trainer.yaml` (single source of truth); override path with `CELLPOSE_TRAIN_CONFIG`; resolved values logged + frozen to `run_logs/*.config.yaml` |
| Email | `PITT_EMAIL` in `.env` (gitignored), passed by `submit.sh` → `sbatch --mail-user` |
| PyYAML | `trainer_slurm.py` imports `yaml`; `pip install pyyaml` into the cellpose env if missing |
