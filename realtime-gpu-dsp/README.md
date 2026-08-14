# The radio will not wait for your FFT

This companion contains a self-contained, output-complete notebook and the
derived evidence behind the Ramwise soft-real-time DSP study.

The example constructs deterministic complex64 I/Q blocks, applies a 129-tap
FIR, non-overlapping 2,048-point FFTs, power spectra, and a peak-to-mean
detector. It runs on NumPy/SciPy everywhere and enables the CuPy path when a
CUDA GPU is available.

## Files

- `realtime_gpu_dsp.ipynb`: runnable explanation, correctness check, small
  local demonstration, and charts from the publication matrix;
- `results/capacity.csv`: service-capacity medians from three fresh-process
  replications;
- `results/paced-full.csv`: one-second exploratory queue sweep;
- `results/paced-confirm.csv`: five-second confirmation around the CPU and
  pinned-GPU boundaries;
- `results/README.md`: timing boundaries, source commits, and aggregation
  notes.

## Run

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
jupyter lab
```

The CPU notebook does not require CUDA. For a CUDA 13 GPU environment, add the
tested CuPy build with `python -m pip install cupy-cuda13x==14.1.1`. The
published outputs came from CUDA 13.1 on an NVIDIA RTX PRO 4000 Blackwell SFF
Edition.

The notebook is an explanatory companion, not the full load generator. The
large study used fresh processes, randomized condition order, a bounded queue,
and immutable raw JSON retained outside the public repository.
