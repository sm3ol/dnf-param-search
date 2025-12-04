dsa_project/
├─ cpp/param_data_structure/
│ ├─ include/
│ ├─ src/
│ ├─ tests/
│ ├─ CMakeLists.txt
│ └─ paramdag_py/
│
├─ python_env/
│ ├─ bb_run.py                # CLI: dataset+config → outputs
│ ├─ bb_config.py
│ ├─ bb_core.py               # calls adapter to build I_ext and steps the DNF
│ ├─ bb_metrics.py
│ ├─ bb_report.py
│ ├─ run_innerloop_sweep.py
│ ├─ probe_sim.py
│ ├─ pick_recommendations.py
│ ├─ run_ycb_eval.py
│ ├─ metrics_ycb.py
│ ├─ plots/
│ │  ├─ plot_pareto.py
│ │  ├─ plot_probe_responses.py
│ │  ├─ plot_stability_maps.py
│ │  ├─ plot_efficiency_scaling.py
│ │  ├─ plot_sensitivity.py
│ │  └─ plot_ycb_validation.py
│ └─ adapters/                # <<< dataset plug-ins live here
│    ├─ __init__.py
│    ├─ base.py               # abstract Adapter interface
│    ├─ registry.py           # register/get adapter by name
│    ├─ ycbsight/             # tactile images (GelSight) + RGB-D
│    │  ├─ __init__.py
│    │  ├─ loader.py          # yields H×W frames or sequences
│    │  └─ mapper.py          # 2D→1D index, builds Gaussian I_ext
│    └─ dvs_grasping/         # DAVIS events/APS (e.g., HuCaoFighting/DVS-GraspingDataSet)
│        ├─ __init__.py
│        ├─ loader.py          # bin events → frames or sparse tensors
│        └─ mapper.py          # builds I_ext from event frame(s)
│
├─ core/
│ ├─ dft_core.py
│ ├─ hdc_core.py
│ └─ projectors.py            # shared 2D→1D mappings, kernels, utils
│
├─ configs/
│ ├─ paramdag_dnf.yaml
│ ├─ search_space.yaml
│ ├─ probe.yaml
│ ├─ config.yaml              # tuned DNF params (dataset-agnostic)
│ ├─ datasets.yaml            # <<< dataset registry (paths/options)
│ └─ tuned/
│    ├─ fast.yaml
│    ├─ balanced.yaml
│    └─ ultra_stable.yaml
│
├─ artifacts/
│ ├─ runs/…                   # black-box outputs (registry.jsonl, plots/)
│ ├─ pareto/…
│ └─ ycbsight/…
│
├─ README.md
├─ environment.yml
└─ requirements.txt




----------------------------------------------------------------------------------------------



recommended project directory: "C:\dsa_project"

recommended dataset directory: "C:\Datasets\YCBSight-Real\004_sugar_box\gelsight"

The iterations results will be found in "dsa_project/artifacts/runs"

----------------------------------------------------------------------------------------------

For a successful run/iteration:

  1) install imagio

  %conda install -c conda-forge imageio


  2) ain command to run an iteration (this is for ipython console in Spyder Software):


  !python -m python_env.bb_run \
    --config configs/config.yaml \
    --cert configs/certification.yaml \
    --datasets configs/datasets.yaml \
    --dataset ycbsight_real \
    --split test \
    --out artifacts/runs \
    --max-frames 80 \
    --per-sample --sample-steps 200 --print-every 1 --save-traces




