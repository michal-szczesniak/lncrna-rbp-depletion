# Scripts

Scripts are numbered roughly in the order in which they were run, from the project directory. The figures of the article are made by `60_*` to `66_*`, `94_fig1e_audit.py` and `82_compose_figures.py` (see `README.md`), and the numbers in the text were checked with `44_stats_checks.py` and `90_*` to `98_*`. Scripts `41_*` to `43_*`, `47_*`, `55_*` to `57_*`, `69_*` to `72_*`, `74_*`, `76_*` and `79_*` made working reports, previews or fixes to earlier drafts of the figures. Comments inside the scripts are partly in Polish; the descriptions below are translated from them.

| Script | Description |
|---|---|
| `00_encode_inventory.py` | Phase 0: inventory of the ENCODE data (RBP depletion experiments, their controls and other assays). |
| `00b_inventory_check.py` | Phase 0: checks of the inventory. |
| `00c_make_download_list.py` | Phase 0: list of files to download (quantifications based on GENCODE v29). |
| `01_download.py` | Downloads the ENCODE files listed in the inventory, with md5 checks and resuming. |
| `02_get_gencode.sh` | Downloads the GENCODE GTF in the version used by ENCODE for the depletion quantifications (v29). |
| `10_pilot_samples.py` | Phase 1 (pilot): choice of depletions for the pilot and the table of samples (depletion and control). |
| `11_pilot_deseq2.R` | Phase 1: DESeq2 and apeglm for the pilot depletions. |
| `12_pilot_classes.py` | Phase 1: gene classes and a positive control. |
| `13_pilot_classes_mle.py` | Phase 1: checks whether the result of the previous step reflects biology or apeglm shrinkage. |
| `14_pilot_bru_stability.py` | Phase 1: baseline stability of gene classes from Bru-seq and BruChase-seq. |
| `15_pilot_bigwig_list.py` | Phase 1: list of bigWig files (unique reads, both strands) for the pilot depletions and their controls. |
| `16_pilot_eisa.py` | Phase 1: exon–intron split analysis (EISA) for the pilot. |
| `17_timelapse_inspect.py` | Phase 1: inspection of processed TimeLapse-seq files (format, genes, classes). |
| `18_timelapse_validation.py` | Phase 1: validation of EISA against direct measurements of RNA turnover (TimeLapse-seq, Ietswaart et al. 2024). |
| `20_phase2_prepare.py` | Phase 2: preparation of the analysis of all depletions. |
| `21_deseq2_all.R` | Phase 2: DESeq2 for all depletions; each run fits part "task" of "ntask" (Slurm array). |
| `21_deseq2_all.slurm` | Slurm array job that runs `21_deseq2_all.R`. |
| `22_phase2_matrix.py` | Phase 2: quality control of all depletions, matrices of effects and the general response. |
| `23_phase2_pca_diag.py` | Phase 2: diagnostics of the general response (principal components). |
| `24_phase2_batch_center.py` | Phase 2: removal of the effect of shared control samples (shared-control correction). |
| `25_phase2_class_effects.py` | Phase 2: comparison of RNA classes in all depletions (class effects). |
| `26_phase2_rbp_hits.py` | Phase 2: RBPs with a reproducible effect on lncRNAs, and robustness checks. |
| `27_phase2_functional.py` | Phase 2: consistency of the effects on lncRNAs within functional groups of RBPs. |
| `30_phase3_splice_nmd.py` | Phase 3: spliceosome, NMD and snoRNA host genes at the level of single genes. |
| `31_phase3_bigwig_list.py` | Phase 3: bigWig files for EISA of spliceosome and NMD/EJC depletions and their controls. |
| `32_phase3_eisa.py` | Phase 3: EISA for spliceosome and NMD/EJC depletions. |
| `33_phase3_fraction_files.py` | Phase 3: ENCODE files of fractionation experiments (subcellular localization of RNA). |
| `34_phase3_localization.py` | Phase 3: does the effect of an RBP on lncRNAs depend on their subcellular localization? |
| `35_phase3_eclip_files.py` | Phase 3: ENCODE eCLIP peak files for K562 and HepG2. |
| `36_phase3_eclip.py` | Phase 3: direct binding (eCLIP) versus the effect of RBP depletion. |
| `37_phase3_plasticity.py` | Phase 3: do the lncRNA–mRNA differences reflect transcriptional plasticity? |
| `38_phase3_cis_nat.py` | Phase 3: cis effects of depletion on NATs and on genes near the depleted gene, CRISPR knockout versus shRNA (Fig. S5). |
| `40_phase4_robustness.py` | Phase 4: robustness of the main result (lncRNAs minus matched mRNAs, corrected matrix, main matching). |
| `41_phase4_figures_A.py` | Phase 4: working versions of Figures 1 and 2 (earlier draft). |
| `42_phase4_figures_B.py` | Phase 4: working versions of Figures 3–5 and S1 (earlier draft). |
| `43_report.py` | Phase 4: working HTML report with figures and tables. |
| `44_stats_checks.py` | Statistical checks. |
| `45_case_inventory.py` | Inventory for the case studies of single loci. |
| `46_case_bigwigs.py` | bigWig files for the locus plots. |
| `47_case_locus.py` | Locus plots for the case studies (working version; the article uses `figlocus.py`). |
| `48_rbp_profiles.py` | Summary of what is known about the 35 RBPs with a reproducible class effect. |
| `49_encode_audits.py` | ENCODE audit flags for all depletion experiments. |
| `50_exosome_divergent.py` | Does loss of nuclear surveillance (exosome, NEXT, PAXT) raise divergent and antisense lncRNAs in particular? |
| `51_rbp_networks.py` | RBP–RBP similarity networks built separately from lncRNA and mRNA responses (Fig. S6). |
| `52_profiles_and_neighbours.py` | Response profiles of RBPs compared between cell lines, and co-response of neighbouring genes. |
| `53_specific_responders.py` | Specific responders: genes that respond strongly to the loss of one RBP and much less to the others. |
| `54_termination_theme.py` | Test of whether RBPs of transcription termination, 3′-end processing and early RNA fate specifically change lncRNAs. |
| `55_report_figs.py` | Working report v2, part A: figures. |
| `56_report_html.py` | Working report v2, part B: HTML report. |
| `57_followups.py` | Follow-up analyses after comments on working report v2. |
| `60_fig1.py` | Figure 1: approach and main result (panels A–D; panel E is drawn by `94_fig1e_audit.py`). |
| `61_fig2.py` | Figure 2: properties of lncRNAs and binding of the depleted RBP do not explain the class effect. |
| `62_fig3.py` | Figure 3: spliceosome, NMD and snoRNA host genes. |
| `63_fig4.py` | Figure 4: nuclear surveillance and divergent lncRNAs. |
| `64_fig5.py` | Figure 5: specific regulators of single lncRNAs. |
| `65_figS3_S4.py` | Figures S4 and S6 (named S3 and S4 in an earlier draft). |
| `66_figS1_S2_S5.py` | Figures S1, S2, S3 and S5 (panel folders `figS2`, `figS5` and `figS1`). |
| `67_nmd_skuteczne.py` | Check of the NMD result using only the depletions that disrupted NMD. |
| `68_gc_robustness.py` | Robustness of class effects to GC content (GC dependence of changes in depletion data, Van Nostrand et al. 2020). |
| `69_make_report_v3.py` | Creates `70_report_v3.py` from `56_report_html.py` (working report v3). |
| `70_report_v3.py` | Working report v3 (HTML). |
| `71_report_v3_fixes.py` | Fixes to working report v3. |
| `72_podgladS5.py` | Preview of locus panels (earlier draft). |
| `73_gc_35rbp.py` | Stability of the list of 35 RBPs after accounting for GC content (from the results of `68_gc_robustness.py`). |
| `74_fig1_gc_patch.py` | Figure 1 after the choice of the GC-robust set of RBPs (earlier draft). |
| `75_tableS1.py` | The 35 RBPs with a clear class effect in both cell lines, with effects after GC matching (Table S3 of the article; Table S1 in an earlier draft). |
| `76_fig1d_legend.py` | Layout fix of the legend of Figure 1D (earlier draft). |
| `77_export_gene_properties.py` | Export of gene properties measured in control samples, one row per GENCODE v29 gene (GRCh38). |
| `78_export_intron_retention.py` | Intron retention in control samples, for single introns annotated in GENCODE v29 (GRCh38). |
| `79_fig23_patch.py` | Fixes to Figures 2 and 3 (earlier draft). |
| `80_figS5_KCNQ1OT1_K562.py` | Locus plot of KCNQ1OT1 in K562 after depletion of MTREX and exosome subunits (now Fig. S3A). |
| `81_tables_S3_S4.py` | Supplementary tables (numbered S3 and S4 in an earlier draft). |
| `82_compose_figures.py` | Assembles Figures 1–5 and S1–S6 from the panels in `results/figures_pub`. |
| `90_number_provenance.py` | Register of the numbers in the manuscript: links each number in the text to analysis output. |
| `91_audit_controls.py` | Control analyses A1–A4 of the audit (results in `results/audit`, used by `94_fig1e_audit.py`). |
| `92_audit_quick.py` | Quick analyses of the audit (A3b, A5–A8, A10). |
| `93_audit_fix.py` | Completion of the audit analyses A5 and A6. |
| `94_fig1e_audit.py` | Figure 1E: agreement between cell lines (rho, 95% CI) under alternative corrections, matchings and subsets, and for pseudo-lncRNAs. |
| `95_number_checks.py` | Checks of numbers in the manuscript not assigned by `90_number_provenance.py`. |
| `96_quick_checks.py` | Check of the number of RBPs given in the text. |
| `97_checks.py` | Checks for the number register and a supplementary table. |
| `98_labs.py` | ENCODE laboratories that produced the data used (for the Acknowledgements). |
| `figlib.py` | Shared module for the article figures: style, palette, sizes in mm, vector output, significance marks. |
| `figlocus.py` | Locus plots for the article (Figures 4E, 5C, S2 and S3). |
