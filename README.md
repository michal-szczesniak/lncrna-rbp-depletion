# lncRNA responses to RNA-binding protein depletion

Code for the analyses and figures of:

> Szcześniak MW, Wanowska E. RNA-binding protein depletion reveals reproducible, protein-specific differences between lncRNA and mRNA responses. Manuscript submitted for publication (2026).

The study re-analyses 860 ENCODE RNA-seq experiments in which one of 473 RNA-binding proteins (RBPs) was depleted by shRNA knockdown or CRISPR knockout in K562 or HepG2 cells, and compares each lncRNA with mRNAs matched for expression, gene length, splicing efficiency and stability.

## Contents

- `scripts/`: all analysis and figure scripts, numbered roughly in the order in which they were run. `SCRIPTS.md` gives a one-line description of each. Comments inside the scripts are partly in Polish.
- `env/`: the conda environments (`rbp-py.yml`, `rbp-r.yml`), the installed Python packages (`python_requirements.txt`), the Python version and the R session (`R_sessionInfo.txt`).
- Processed data are archived in Zenodo ([10.5281/zenodo.23798744](https://doi.org/10.5281/zenodo.23798744)): log2 fold-change matrices and their standard errors, per-experiment DESeq2 results, class effects of all experiments, the results of all other analyses, the data behind each figure panel, derived gene annotation tables and ENCODE metadata.

## Data

All input data are public. ENCODE accession numbers of the RBP depletion experiments, their controls and the other ENCODE datasets used are listed in Supplementary Table S1 of the article and in `data/meta/` of the Zenodo archive. Gene annotation: GENCODE v29. Tissue expression: GTEx v8.

## Requirements

Python 3.11 and R 4.5 with the packages listed in `env/`. The full analysis was run on a computing cluster; fitting DESeq2 models to all experiments needs several hundred CPU hours.

## Running

Scripts read and write paths relative to the project directory (`data/`, `results/`, `logs/`) and are run from it, for example `python scripts/60_fig1.py`.

The full analysis starts from the ENCODE files (about 100 GB), which are listed and downloaded by `00_encode_inventory.py`, `00c_make_download_list.py` and `01_download.py`, and from GENCODE v29 (`02_get_gencode.sh`). DESeq2 is fitted to every experiment by `21_deseq2_all.R`, run as a Slurm array job (`21_deseq2_all.slurm`).

To reproduce the figures without these steps, extract the Zenodo archive into the project directory, run the figure scripts in the order of the table below (they write the panels to `results/figures_pub/`), and assemble the panels into figures with `python scripts/82_compose_figures.py results/figures_pub results/figures_composed`. Locus plots (Figures 4E, 5C, S2 and S3) also need the ENCODE signal files (bigWig) listed in `data/meta/case_bigwig_files.tsv`. Figure 6 and the graphical abstract are schematic drawings and are not generated from the data.

| Figure | Script | Panel folder in `results/figures_pub/` |
|---|---|---|
| 1 | `60_fig1.py`, then `94_fig1e_audit.py` (panel E, from the results of `91_audit_controls.py`) | `fig1` |
| 2 | `61_fig2.py` | `fig2` |
| 3 | `62_fig3.py` | `fig3` |
| 4 | `63_fig4.py` | `fig4` |
| 5 | `64_fig5.py` | `fig5` |
| S1 | `66_figS1_S2_S5.py` | `figS2` |
| S2, S3 | `66_figS1_S2_S5.py` | `figS5` |
| S4 | `65_figS3_S4.py` | `figS3` |
| S5 | `66_figS1_S2_S5.py` (from the results of `38_phase3_cis_nat.py`) | `figS1` |
| S6 | `65_figS3_S4.py` | `figS4` |

Panel folders of the supplementary figures keep the names of an earlier draft, as listed in the table.

## License

Code: MIT License (see `LICENSE`). Data in the Zenodo archive: CC BY 4.0.

## Citation

Please cite the article above and the Zenodo archive ([10.5281/zenodo.23798744](https://doi.org/10.5281/zenodo.23798744)).
