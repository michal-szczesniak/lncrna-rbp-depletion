#!/usr/bin/env Rscript
# Faza 2, krok 2.2: DESeq2 dla wszystkich knockdownow; skrypt liczy czesc "task" z "ntask" (tablica SLURM).
# Wyjscie: results/phase2/deseq2/<KD>.tsv.gz (na gen) i results/phase2/deseq2/summary/<KD>.tsv (na knockdown).
# Knockdowny juz policzone sa pomijane, wiec zadanie mozna bezpiecznie uruchomic ponownie.
suppressPackageStartupMessages({ library(DESeq2); library(apeglm); library(data.table) })
args <- commandArgs(trailingOnly = TRUE)
task <- as.integer(args[1]); ntask <- as.integer(args[2])
outdir <- "results/phase2/deseq2"; sumdir <- file.path(outdir, "summary")
dir.create(sumdir, recursive = TRUE, showWarnings = FALSE)
ann <- fread("data/ref/genes_v29.tsv")
aliases <- list(MTREX = "SKIV2L2", ATP5A1 = "ATP5F1A", ATP5C1 = "ATP5F1C", GNB2L1 = "RACK1")
smp <- fread("data/meta/phase2_samples.tsv")
kds <- sort(unique(smp$kd_experiment))
mine <- kds[seq(task, length(kds), by = ntask)]
message(sprintf("Zadanie %d z %d: %d knockdownow", task, ntask, length(mine)))
read_counts <- function(p) {
  x <- fread(p, select = c("gene_id", "expected_count"))
  x[startsWith(gene_id, "ENSG")]
}
for (kd in mine) {
  out_file <- file.path(outdir, paste0(kd, ".tsv.gz")); sum_file <- file.path(sumdir, paste0(kd, ".tsv"))
  if (file.exists(out_file) && file.exists(sum_file)) { message(kd, ": policzony wczesniej, pomijam"); next }
  t0 <- Sys.time()
  s <- smp[kd_experiment == kd]
  s[, sample := paste(condition, sample_experiment, bio_rep, sep = "_")]
  mats <- lapply(s$path, read_counts)
  ids <- mats[[1]]$gene_id
  stopifnot(all(vapply(mats, function(m) identical(m$gene_id, ids), logical(1))))
  cnt <- round(sapply(mats, function(m) m$expected_count))
  storage.mode(cnt) <- "integer"
  dimnames(cnt) <- list(ids, s$sample)
  cnt <- cnt[rowSums(cnt) >= 10, , drop = FALSE]
  cd <- data.frame(row.names = s$sample, condition = factor(s$condition, levels = c("control", "kd")))
  dds <- DESeq(DESeqDataSetFromMatrix(cnt, cd, ~ condition), quiet = TRUE)
  res <- results(dds, name = "condition_kd_vs_control")
  shr <- lfcShrink(dds, coef = "condition_kd_vs_control", type = "apeglm", quiet = TRUE)
  out <- data.table(gene_id = rownames(res), baseMean = res$baseMean, lfc_mle = res$log2FoldChange,
                    lfcSE_mle = res$lfcSE, stat = res$stat, pvalue = res$pvalue, padj = res$padj,
                    lfc_shr = shr$log2FoldChange)
  fwrite(out, out_file, sep = "\t")
  tg <- s$target[1]
  t_row <- merge(out, ann[gene_name %in% c(aliases[[tg]], tg), .(gene_id)], by = "gene_id")[order(-baseMean)]
  ok <- nrow(t_row) > 0
  fwrite(data.table(
    kd_experiment = kd, target = tg, cell = s$cell[1], method = s$method[1], biblioteka = s$biblioteka[1],
    n_kd = sum(s$condition == "kd"), n_ctrl = sum(s$condition == "control"), n_genes = nrow(out),
    n_padj05 = sum(out$padj < 0.05, na.rm = TRUE),
    n_up = sum(out$padj < 0.05 & out$lfc_mle > 0, na.rm = TRUE),
    n_down = sum(out$padj < 0.05 & out$lfc_mle < 0, na.rm = TRUE),
    target_lfc = if (ok) round(t_row$lfc_mle[1], 2) else NA_real_,
    target_padj = if (ok) signif(t_row$padj[1], 2) else NA_real_,
    target_baseMean = if (ok) round(t_row$baseMean[1]) else NA_real_), sum_file, sep = "\t")
  message(sprintf("%s %-10s %-5s %-6s istotne: %5d, cel log2FC: %s, %.0f s", kd, tg, s$cell[1], s$method[1],
                  sum(out$padj < 0.05, na.rm = TRUE), if (ok) sprintf("%.2f", t_row$lfc_mle[1]) else "NA",
                  as.numeric(difftime(Sys.time(), t0, units = "secs"))))
}
message("Zadanie zakonczone.")
