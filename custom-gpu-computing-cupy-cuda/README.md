# Custom GPU Computing with CuPy and CUDA

Companion to the Ramwise article **The Kernel Was Fast. The Trip Wasn't.**

The study implements one useful numeric scoring operation at four levels:
NumPy, composed CuPy, partially fused CuPy, and handwritten CUDA. It also keeps
deliberately awkward layouts and launch configurations so readers can see why
plausible GPU optimizations sometimes help, sometimes disappear into transfer
cost, and sometimes do nothing.

Open `custom_gpu_computing.ipynb` for the guided analysis. The committed
notebook contains all outputs and reads only the four sanitized CSVs under
`results/`; inspecting or rerunning it does not require a GPU.

`full-benchmark/` contains the runnable benchmark, correctness tests, exact
resolved environment, digest-pinned CUDA base, publication matrix, and
optional Nsight Compute target. Raw JSON, profiler reports, host paths, logs,
and notebook-building utilities are intentionally excluded.

## Open or rerun the notebook

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt
jupyter execute custom_gpu_computing.ipynb --inplace
```

