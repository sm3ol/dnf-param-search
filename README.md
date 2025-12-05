Here’s a cleaned-up, more complete `README.md` you can paste over your current one and tweak as you like.

---

# DNF Parameter Search Project (CE 6320 – Data Structures & Algorithms)

This project is a small **hyperparameter search engine** for a **Dynamic Neural Field (DNF)** controller.

* The **Python side** runs the real DNF simulation on the **YCBSight-Real** dataset and logs performance metrics.
* The **C++ side** implements:

  * a **typed parameter data structure** (`DNFParams`) with constraints and random sampling
  * and a **multi-objective genetic search (NSGA-II style)** over those parameters.

The end goal for the course project is to:

1. Show a **clean data structure design** for the parameter space.
2. Implement and compare **search algorithms** over that space (NSGA-II, MOTPE, random search, etc.).
3. Connect them to a real black-box system (the DNF core).

---

## 1. Repository Layout

```text
dsa_project/
├── cpp/
│   ├── param_data_structure/
│   │   ├── include/
│   │   │   ├── dnf_params.hpp       # C++ struct for DNF hyperparameters
│   │   │   └── nlohmann/            # json.hpp header (single-header JSON library)
│   │   ├── src/
│   │   │   └── dnf_params.cpp       # isValid(), randomSample(), toJSON()
│   │   ├── tests/                   # (optional) C++ tests / playgrounds
│   │   └── paramdag_py/             # placeholder for future DAG-style DS
│   │
│   └── search/
│       └── dnf_nsga_search.cpp      # NSGA-II style multi-objective search over DNFParams
│
├── python_env/
│   ├── bb_run.py                    # main black-box runner (config + dataset → metrics + artifacts)
│   ├── bb_config.py
│   ├── bb_core.py                   # builds DNF inputs (I_ext), steps the DNF
│   ├── bb_metrics.py                # computes temporal / stability metrics from traces
│   ├── bb_report.py                 # reporting / summary utilities
│   ├── run_innerloop_sweep.py
│   ├── probe_sim.py
│   ├── pick_recommendations.py
│   ├── run_ycb_eval.py
│   ├── metrics_ycb.py
│   ├── plots/
│   │   ├── plot_pareto.py
│   │   ├── plot_probe_responses.py
│   │   ├── plot_stability_maps.py
│   │   ├── plot_efficiency_scaling.py
│   │   ├── plot_sensitivity.py
│   │   └── plot_ycb_validation.py
│   └── adapters/
│       ├── __init__.py
│       ├── base.py                  # abstract adapter interface
│       ├── registry.py              # register/get adapter by name
│       ├── ycbsight/
│       │   ├── __init__.py
│       │   ├── loader.py            # loads tactile + RGBD frames
│       │   └── mapper.py            # 2D→1D mapping, builds Gaussian I_ext
│       └── dvs_grasping/
│           ├── __init__.py
│           ├── loader.py            # bin events → frames/tensors
│           └── mapper.py            # builds I_ext from event frames
│
├── core/
│   ├── dft_core.py                  # DNF / field simulation core
│   ├── hdc_core.py                  # (optional) hyperdimensional stuff
│   └── projectors.py                # shared 2D→1D mappings, kernels, utilities
│
├── configs/
│   ├── config.yaml                  # base DNF configuration (single “tuned” candidate)
│   ├── certification.yaml           # target spec / certification thresholds
│   ├── datasets.yaml                # dataset registry and paths
│   ├── paramdag_dnf.yaml            # (future) param DAG for constrained search
│   ├── probe.yaml
│   ├── search_space.yaml            # parameter ranges for search (conceptual)
│   └── tuned/
│       ├── fast.yaml
│       ├── balanced.yaml
│       └── ultra_stable.yaml
│
├── artifacts/
│   ├── runs/                        # black-box outputs (per run)
│   │   └── ycbsight_real_test_...   # timestamped run folders
│   │       ├── samples/             # per-sample metrics/traces
│   │       ├── plots/               # generated plots (if enabled)
│   │       ├── copies/              # copy of config/cert files used
│   │       ├── summary.json         # overall summary
│   │       └── *_samples.csv        # per-sample metrics table
│   ├── pareto/                      # future: save Pareto fronts here
│   └── ycbsight/                    # dataset-specific outputs
│
├── playground/
│   ├── cpp_py_dummy/                # minimal C++ ↔ Python toy example
│   └── toy_search_pipeline/         # toy DNF search (synthetic eval, fast)
│
├── .gitignore
├── environment.yml
├── requirements.txt
├── FAQs.docx
└── README.md                        # (this file)
```

---

## 2. Core Data Structure: `DNFParams`

The central C++ data structure is:

```cpp
struct DNFParams {
    double tau;
    double dt;
    double beta;
    double theta;
    double h;
    double w_exc;
    double sigma_exc;
    double w_inh;
    double sigma_inh;
    double intensity_scale_factor;
    int    gaussian_stimulus_width;
    int    N;

    bool isValid();
    static DNFParams randomSample();
    nlohmann::json toJSON() const;
};
```

You can think of it as a **typed configuration record** with:

* **Range constraints** (e.g., `0.005 ≤ tau ≤ 0.020`, `0.0005 ≤ dt ≤ 0.0020`, etc.).
* **Derived constraints**:

  * `alpha = dt / tau` must be in `(0, 0.2)`.
  * `sigma_inh > sigma_exc + 1.0`, `3 ≤ gaussian_stimulus_width ≤ 11`.
  * `N ∈ {50, 100, 150, 200}`.
* **Operations**:

  * `isValid()` checks all constraints.
  * `randomSample()` samples random parameters until it finds a valid one.
  * `toJSON()` serializes the parameters to JSON for the Python side.

From a **data-structures** perspective:

* It’s a **record / struct** with embedded **invariants** and **sampling logic**.
* On top of this, search algorithms maintain a **population** as `std::vector<Individual>` (for NSGA-II) or `std::vector<Trial>` (for MOTPE).

---

## 3. Setting Up the Environment

### 3.1. Recommended layout (Windows)

* Project directory:
  `C:\dsa_project`  → this repo.
* Dataset directory (for YCBSight-Real):
  e.g. `C:\Datasets\YCBSight-Real\004_sugar_box\gelsight`
  (Configure the actual path in `configs/datasets.yaml`.)

### 3.2. Conda env + Python dependencies

If you have `environment.yml`:

```bash
conda env create -f environment.yml
conda activate <your_env_name>
```

Or, with plain `requirements.txt`:

```bash
pip install -r requirements.txt
```

You’ll also need:

```bash
conda install -c conda-forge imageio
```

(Required by the dataset / plotting helpers.)

### 3.3. C++ toolchain

You need a **C++17 compiler** (on Windows, e.g. `g++` via MSYS2 / Mingw-w64).

From `C:\dsa_project`, check:

```powershell
g++ --version
```

If that works, you’re good.

---

## 4. Running the Baseline DNF (No Search)

This runs the **original DNF pipeline** (Python-only), for a fixed config:

```bash
python -m python_env.bb_run ^
    --config configs/config.yaml ^
    --cert configs/certification.yaml ^
    --datasets configs/datasets.yaml ^
    --dataset ycbsight_real ^
    --split test ^
    --out artifacts/runs ^
    --max-frames 80 ^
    --per-sample --sample-steps 200 --print-every 1 --save-traces
```

This will produce a new run folder under:

```text
artifacts/runs/ycbsight_real_test_YYYY-MM-DDTHH-MM-SSZ/
```

Inside, you’ll see:

* `*_samples.csv` → per-sample metrics (`k90_ms`, `k99_ms`, `ripple_pct`, `peak`, `steps`, `pass`, ...).
* `summary.json` → a tiny summary (dataset, split, `overall_pass`, etc.).
* Optionally plots + traces if enabled.

This is your **baseline “no search” run**.

---

## 5. NSGA-II Style Search (C++ + Python)

### 5.1. Conceptual flow

The NSGA-II search integrates with the DNF pipeline conceptually like this:

1. **C++** samples a candidate `DNFParams` (via `randomSample()` or from the population).
2. It **writes a config** (YAML/JSON) describing these parameters.
3. It calls a **Python wrapper** that:

   * runs `python_env.bb_run` with that config,
   * reads the latest `*_samples.csv` under `artifacts/runs/...`,
   * computes aggregated metrics (mean k90, k99, ripple, peak, steps, pass),
   * returns them as JSON.
4. C++ reads those metrics, turns them into **multi-objective scores**, and

   * performs **non-dominated sorting (NSGA-II)**,
   * keeps a **Pareto front** of good trade-offs,
   * and evolves a new population.

> **Important note:**
> There are two “flavors” of the search code:
>
> * A **toy pipeline** in `playground/toy_search_pipeline/` that uses a cheap synthetic evaluator (fast).
> * A **DNF-integrated pipeline** in `cpp/search/dnf_nsga_search.cpp` that is designed to talk to the real Python core (slower but realistic). Integration details can be adapted based on time and compute budget.

### 5.2. Compiling the NSGA-II search (from repo root)

From `C:\dsa_project`:

```powershell
g++ cpp\search\dnf_nsga_search.cpp `
    cpp\param_data_structure\src\dnf_params.cpp `
    -I cpp\param_data_structure\include `
    -I cpp\param_data_structure\include\nlohmann `
    -std=c++17 -o dnf_nsga_search.exe
```

This will produce:

```text
dnf_nsga_search.exe   # in the repo root (or wherever you run g++)
```

(If you prefer, you can also compile from inside `cpp\search` and adjust paths accordingly.)

### 5.3. Running the NSGA-II search

Once compiled:

```powershell
.\dnf_nsga_search.exe
```

What it does (high level):

* Initializes a population of `DNFParams` (each checked by `isValid()`).
* For each generation:

  * Evaluates each individual (through the Python side or the toy evaluator).
  * Performs **fast non-dominated sorting**.
  * Computes **crowding distance** within each front.
  * Uses **binary tournament selection** (rank + crowding) to choose parents.
  * Applies **crossover** + **mutation** to generate offspring.
* At the end:

  * Extracts the first Pareto front.
  * Picks a representative solution (e.g., with minimal sum of objectives).
  * Prints:

    * Best metrics (`k90_ms`, `k99_ms`, `ripple`, `peak`, `steps`, `pass`).
    * The corresponding `DNFParams` in JSON.

This is your **“fancy” multi-objective search** over the same `DNFParams` struct.

---

## 6. Toy Playground Pipelines (For Debugging & Demo)

To avoid waiting for the full DNF simulation, there is a **toy search pipeline** under:

```text
playground/toy_search_pipeline/
    - dnf_params.hpp / .cpp (copied from core)
    - toy_eval.py            # cheap synthetic evaluator
    - toy_search.cpp         # NSGA-II or GA search using toy_eval
```

There you can:

* Compile `toy_search.cpp` with `dnf_params.cpp`.
* Run `toy_search.exe` and see:

  * printed iterations / generations,
  * synthetic “metrics” improving,
  * JSON dump of the best parameters.

The toy evaluator is designed to **behave qualitatively** like the real DNF (fast/slow, ripple, peak, steps) but is extremely cheap to run, so it’s perfect for:

* Debugging the **data structure** (`DNFParams`).
* Debugging the **search logic** (NSGA-II or MOTPE).
* Showing algorithm behavior in your report / slides.

---

## 7. What This Shows (Course Point of View)

You can explicitly highlight these for CE 6320 / Data Structures & Algorithms:

1. **Data structure design**

   * `DNFParams` as a **typed configuration record** with:

     * range constraints,
     * derived invariants (`alpha`, `sigma_inh > sigma_exc + 1`, discrete `N`),
     * random valid sampling,
     * JSON serialization.
   * Search-level containers:

     * `std::vector<Individual>` / `std::vector<Trial>` as dynamic arrays for populations/history.

2. **Algorithms**

   * **Random sampling** baseline (trivial search).
   * **Genetic algorithm / NSGA-II**:

     * fast non-dominated sorting,
     * Pareto fronts,
     * crowding distance,
     * tournament selection,
     * crossover + mutation.
   * (Optional / future) **MOTPE** or other black-box optimizers built on the same `DNFParams` struct.

3. **Black-box optimization pattern**

   * C++ doesn’t know the DNF internals; it only sees **metrics**:

     * `k90_ms`, `k99_ms`, `ripple`, `peak`, `steps`, `pass`.
   * The Python side is an expensive “oracle” (DNF simulation) that returns metrics.
   * The search algorithm treats it as a **black-box function** over the parameter space.

---

## 8. Future Work / Extensions

Things that can be added later (or described in the report as “future work”):

* Implement a **true param DAG** in `cpp/param_data_structure/paramdag_py` so parameters can depend on each other structurally, not just via `isValid()`.
* Add a clean C++ tool to **emit a full Pareto front** as a CSV or JSON under `artifacts/pareto/`.
* Pull in a **MOTPE** implementation (C++ or Python) side-by-side with NSGA-II and compare:

  * number of evaluations,
  * best metric vector,
  * runtime.
* Visualize Pareto fronts using `python_env/plots/plot_pareto.py`.

---

If you want, next step I can:

* Add a **short “Quick Start” section** tailored for your exact Windows/conda setup (with the exact env name you’re using).
* Or help you create a tiny section at the end: “How to explain this to the professor in 2 minutes” with a high-level script you can literally say in class.
