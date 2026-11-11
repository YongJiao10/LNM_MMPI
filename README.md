# LNM_MMPI

**Lesion Network Mapping of Multidimensional Psychopathology**

Official repository for:
**Convergent neuropsychiatric profiles associated with distinct structural and functional network patterns in patients with focal brain lesions**

Yong Jiao, Nandi Zhang, Joseph Griffis, Joel Bruss, Camryn Blumberg, Aaron D. Boes, Jing Jiang, Yu Zhang

---

## Overview

This repository provides the computational framework and custom analysis scripts for investigating multivariate brain–behavior relationships in patients with focal brain lesions. By integrating functional and structural lesion network mapping (fLNM and sLNM) with multidimensional behavioral phenotypes (MMPI-2-RF) using regularized canonical correlation analysis (rCCA) with nested cross-validation, this framework identifies convergent brain–behavior dimensions and characterizes their transcriptomic, cellular, and neurotransmitter profiles.

## Project Structure

```text
.
├── cca401/                         # Core rCCA algorithms, cross-validation, and parameter grid
│   ├── preferred.py                # Adopted baseline configuration (160 participants, 23 scales)
│   ├── nested.py                   # Nested cross-validation logic
│   ├── nested_utils.py             # Splitting, scoring, and normalization helpers
│   ├── seed_screen.py              # Multiseed screening and stability evaluation
│   └── molecular_math.py           # Molecular PLS and spatial correlation statistics
├── cca401_pipeline.py              # End-to-end rCCA pipeline execution
├── cca401_union_grid.py            # Union hyperparameter grid evaluation
├── utils.py                        # Foldwise ICC, metrics, and data processing utilities
├── plots.py                        # Visualization functions and figure aesthetics
├── test_loading_icc.py             # Verification test for loading ICC metrics
├── FC_LNM.py / FC_GSP.py           # Functional lesion network mapping from normative fMRI (GSP)
├── FC_lesion_mp.py                 # Multiprocessing lesion network mapping
├── CentOS7/                        # High-performance computing scripts for voxelwise network mapping
├── scripts/                        # Analysis, permutation testing, and figure rendering scripts
│   ├── run_160_seed1_retuning.py   # Baseline model fitting and sensitivity checks
│   ├── run_baseline_followups.py   # Permutation testing (1,000 iterations) and age/sex sensitivity
│   ├── assess_baseline_followups.py# Verification and summary of permutation tests
│   ├── rebuild_fdc_gsp1000.py      # Direct functional disconnection computation
│   ├── render_combined_contribution_panel.py
│   ├── render_tian_s1_atlas.py
│   └── molecular_figures/          # Transcriptomic (AHBA), PET, and Metascape enrichment analyses
│       ├── README.md
│       ├── analysis/               # Molecular PLS regression and enrichment queries
│       └── rendering/              # Vector figure subpanel rendering scripts
├── R/                              # Cortical and subcortical surface rendering scripts (Schaefer-100 / Tian-S1)
├── tests/                          # Unit and verification tests
├── notebooks/                      # Interactive readers and visualization workflows
│   ├── read_rcca_combined_nested.ipynb
│   ├── read_rcca_fLNM_nested.ipynb
│   ├── read_rcca_sLNM_nested.ipynb
│   ├── read_grid_fLNM.ipynb
│   ├── read_grid_sLNM.ipynb
│   ├── read_grid_scfc.ipynb
│   ├── demo_sc7_ring_plot.ipynb
│   └── load_save_mmpi.ipynb
├── legacy_cca260/                  # Historical CCA-Zoo 2.6.0 analysis workflows
├── data/reference/                 # Atlas metadata and parcellation coordinate references
├── environment_cca401.yml          # Conda environment definition
└── requirements.txt                # Python dependencies
```

## Environment Setup

### Conda Environment (Recommended)
```bash
conda env create -f environment_cca401.yml
conda activate py314
```

### Pip Installation
```bash
pip install -r requirements.txt
```

### Key Dependencies
- Python 3.14+
- `cca-zoo` (v4.0.1)
- `scikit-learn` (v1.8.0)
- `abagen` (v0.1.3)
- `brainspace` (v0.1.22)
- `neuromaps` (v0.0.5)
- `pingouin` (v0.5.5)

## Data Availability
Patient-level lesion masks and MMPI-2-RF scores are available under restricted access, subject to the policies and procedures of the Iowa Neurological Patient Registry. Data access requests may be directed to Aaron D. Boes at the University of Iowa. Normative HCP-1065 tractography is available through DSI Studio (https://brain.labsolver.org/hcp_template.html), and GSP normative connectome data are available via the GSP data access procedure (https://www.neuroinfo.org/gsp). Normative PET maps are accessible through neuromaps (https://neuromaps.readthedocs.io/), and AHBA microarray data are available from the Allen Institute (https://human.brain-map.org/).

## Code Availability
All custom analysis scripts, model specifications, nested cross-validation pipelines, and plotting routines are provided in this repository.

## Citation
If you use this code in your research, please cite:
> Jiao, Y., Zhang, N., Griffis, J., Bruss, J., Blumberg, C., Boes, A. D., Jiang, J., & Zhang, Y. Convergent neuropsychiatric profiles associated with distinct structural and functional network patterns in patients with focal brain lesions. (Under Review).
