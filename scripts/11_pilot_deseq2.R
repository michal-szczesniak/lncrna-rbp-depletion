#!/usr/bin/env Rscript
# Faza 1, krok 2B: DESeq2 + apeglm dla knockdownow pilotazu.
# Wejscie:  data/meta/pilot_samples.tsv, data/ref/gencode.v29.annotation.gtf.gz
# Wyjscie:  results/phase1/deseq2/<eksperyment>.tsv.gz  (wynik na gen)
#           results/phase1/deseq2/summary.tsv           (podsumowanie na knockdown)
#           data/ref/genes_v29.tsv                      (tabela genow, tworzona raz)
suppressPackageStartupMessages({
  library(DESeq2); library(apeglm); library(rtracklayer); library(data.table)
})
outdir <- "results/phase1/deseq2"
dir.create(outdir, recursive = TRUE, showWarnings = FALSE)

# 1. Tabela genow GENCODE v29 (przyda sie tez w kolejnych krokach)
ann_file <- "data/ref/genes_v29.tsv"
if (!file.exists(ann_file)) {
  message("Czytam GTF (ok. 1 min)...")
  g <- import("data/ref/gencode.v29.annotation.gtf.gz", feature.type = "gene")
  fwrite(data.table(gene_id = g$gene_id, gene_name = g$gene_name, gene_type = g$gene_type,
                    chr = as.character(seqnames(g)), start = start(g), end = end(g),
                    strand = as.character(strand(g)), width = width(g)), ann_file, sep = "\t")
}
ann <- fread(ann_file)
aliases <- list(MTREX = c("MTREX", "SKIV2L2"))   # nazwy zmienione po GENCODE v29

# 2. DESeq2 dla kazdego knockdownu wzgledem jego kontroli
smp <- fread("data/meta/pilot_samples.tsv")
read_counts <- function(p) {
  x <- fread(p, select = c("gene_id", "expected_count"))
  x[startsWith(gene_id, "ENSG")]          # bez tRNA (numeryczne) i ERCC
}
summ <- list()
for (kd in unique(smp$kd_experiment)) {
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
  out <- data.table(gene_id = rownames(res), baseMean = res$baseMean,
                    lfc_mle = res$log2FoldChange, lfcSE_mle = res$lfcSE,
                    pvalue = res$pvalue, padj = res$padj,
                    lfc_shr = shr$log2FoldChange, lfcSE_shr = shr$lfcSE)
  out <- merge(out, ann[, .(gene_id, gene_name, gene_type)], by = "gene_id", all.x = TRUE)
  fwrite(out, file.path(outdir, paste0(kd, ".tsv.gz")), sep = "\t")

  tg <- s$target[1]
  names_tg <- if (!is.null(aliases[[tg]])) aliases[[tg]] else tg
  t_row <- out[gene_name %in% names_tg][order(-baseMean)]
  summ[[kd]] <- data.table(
    kd_experiment = kd, target = tg, cell = s$cell[1], method = s$method[1], pilot_role = s$pilot_role[1],
    n_kd = sum(s$condition == "kd"), n_ctrl = sum(s$condition == "control"), n_genes = nrow(out),
    n_padj05 = sum(out$padj < 0.05, na.rm = TRUE),
    n_up = sum(out$padj < 0.05 & out$lfc_mle > 0, na.rm = TRUE),
    n_down = sum(out$padj < 0.05 & out$lfc_mle < 0, na.rm = TRUE),
    target_lfc = if (nrow(t_row)) round(t_row$lfc_mle[1], 2) else NA_real_,
    target_padj = if (nrow(t_row)) signif(t_row$padj[1], 2) else NA_real_,
    target_baseMean = if (nrow(t_row)) round(t_row$baseMean[1]) else NA_real_)
  message(sprintf("%s %-8s %-5s geny: %d, istotne (padj<0.05): %d", kd, tg, s$cell[1],
                  nrow(out), summ[[kd]]$n_padj05))
}
summ <- rbindlist(summ)
fwrite(summ, file.path(outdir, "summary.tsv"), sep = "\t")
options(width = 200)
print(summ[order(pilot_role, target, cell)][, !c("n_kd", "n_ctrl")], row.names = FALSE)
message("\nKazdy knockdown: ", paste(unique(paste0(summ$n_kd, " KD + ", summ$n_ctrl, " kontroli")), collapse = "; "))
