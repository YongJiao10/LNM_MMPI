# Current molecular analyses

The formal selected candidate is defined in `cca401/preferred.py`: seed1,10outer×5inner folds, volume adjustment of both views, candidate-specific inner reference, training-score normalization,225pairs. Each unique final Combined fit supplies four continuous cortical maps without coefficient filtering. Full-sample parameters come from the current full-data selection grid. The current baseline runner is `scripts/run_160_seed1_retuning.py`; the earlier screens are retained under `code_archive_20261031/exploration/`.

The adopted corrected baseline is `results/cca401_160x23_seed1_retuned_mean_tminus1_20261031/`, with independently retuned full-data fits under `fits/` and 100000-spin/seed42/1000-bootstrap decoding under `molecular/`. Shared AHBA inputs are in `data/reference/ahba/`. PET BH-FDR is calculated separately within each CD's 10 targets. Pooled 20-test results are internal only. Pre-correction results and earlier molecular entry points are archived; access to archive contents requires author permission. The adopted decoding implementation is `scripts/molecular_figures/analysis/current_molecular.py`.

The maintained drawing entry points in `rendering/` now read the adopted baseline and write vector PDF subpanels:

- `render_ahba_spatial_correspondence.py`: Figure 6a, four PLS maps, using prepared parcel data under `figure_inputs/ahba_scatter/`.
- `render_fdc_pathway_enrichment.py`: Figure 6b, CD1-FDC positive and negative enrichment only.
- `render_targeted_receptor_screening.py`: Figure 6c, CD1-FDC receptor results and the saved within-CD FDR values. Both CD results remain in `pet.csv`.
- `render_net_spatial_correspondence.py`: Figure 6d, using prepared inputs under `figure_inputs/net_correspondence/`.
- `render_cd1_fdc_stable_gene_weights.py` and `render_cd2_fdc_stable_gene_weights.py`: supplementary gene-weight panels.
- `render_metascape_enrichment_network.py`: two separate vector PDFs, the cropped original website CD1 network and its complete two-column, wide twenty-cluster legend from `_FINAL_GO.csv`. Network content is retained; the clipped website legend image is replaced by directly generated labels and keys using its original colors. Lower-right placement is deferred to slide assembly.

These scripts consume existing results and do not refit models. Figure subpanels are not assembled manuscript figures; further plot edits require the author permissions in `AGENTS.md`.

The Metascape assembly entry additionally uses `pypdf`. In the current `py314` environment, load that pure-Python dependency by appending the Python `site-packages` directory returned by Codex `load_workspace_dependencies` to `sys.path` before running the entry with `runpy.run_path(..., run_name="__main__")`. This uses the bundled dependency without changing the scientific environment.
