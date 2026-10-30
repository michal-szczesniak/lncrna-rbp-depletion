"""
Eksport wlasciwosci genow z PROBEK KONTROLNYCH (bez wyciszenia) do innej analizy. Jeden wiersz na gen GENCODE v29 (GRCh38).
Zrodla to wylacznie dane juz pobrane i przetworzone w projekcie; nic nie jest pobierane od nowa.

Wyjscie:
  results/export/gene_properties_controls_v29.tsv.gz   tabela genow
  results/export/gene_properties_columns.tsv          slownik kolumn (definicje, jednostki, zrodla)
Kolumny zalezne od linii maja sufiks _HepG2 / _K562.
"""
import gzip
import re
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path("results/export")
OUT.mkdir(parents=True, exist_ok=True)
GTF = Path("data/ref/gencode.v29.annotation.gtf.gz")
CELLS = ["HepG2", "K562"]
LNC = {"lincRNA", "antysens", "dywergentny", "sense-intronic", "inne lncRNA", "gospodarz snoRNA", "gospodarz miRNA"}
EC_L2FC, EC_P = 3.0, 3.0
strip = lambda s: pd.Series(s).astype(str).str.split(".").str[0].values
man = pd.read_csv("data/raw/encode/download_manifest.tsv", sep="\t", dtype=str).drop_duplicates("file_accession", keep="last")
DOC = []


def doc(col, text):
    DOC.append({"column": col, "definition": text})


# ---------------------------------------------------------------- geny z GTF
rows = []
with gzip.open(GTF, "rt") as fh:
    for line in fh:
        if line.startswith("#"):
            continue
        f = line.split("\t", 8)
        if f[2] != "gene":
            continue
        a = f[8]
        gid = re.search(r'gene_id "([^"]+)"', a).group(1)
        rows.append({"gene_id": gid.split(".")[0], "gene_id_versioned": gid,
                     "gene_name": re.search(r'gene_name "([^"]+)"', a).group(1),
                     "gene_type": re.search(r'gene_type "([^"]+)"', a).group(1),
                     "chrom": f[0], "start": int(f[3]) - 1, "end": int(f[4]), "strand": f[6]})
G = pd.DataFrame(rows)
G = G[~G["gene_id_versioned"].str.endswith("_PAR_Y")].drop_duplicates("gene_id").set_index("gene_id")
G["gencode_version"] = "v29"
G["genome"] = "GRCh38"
cls = pd.read_csv("data/ref/gene_classes_v29.tsv", sep="\t")
cls = cls[~cls["gene_id"].astype(str).str.endswith("_PAR_Y")]
cls.index = strip(cls["gene_id"])
cls = cls.groupby(level=0).first()
G["project_class"] = cls["klasa"].reindex(G.index)
G["is_lncRNA_project"] = G["project_class"].isin(LNC)
print(f"Geny GENCODE v29: {len(G):,}; lncRNA wg klas projektu: {int(G['is_lncRNA_project'].sum()):,}", flush=True)
doc("gene_id", "Ensembl gene ID bez wersji (GENCODE v29; kopie PAR_Y pominiete).")
doc("gene_id_versioned", "Ensembl gene ID z wersja, jak w GENCODE v29.")
doc("gene_name / gene_type", "Nazwa i biotyp genu z GENCODE v29 (w v29 lncRNA maja biotypy szczegolowe: lincRNA, antisense, sense_intronic itd.).")
doc("chrom, start, end, strand", "Wspolrzedne genu, GRCh38, start 0-based (polotwarte).")
doc("project_class", "Klasa genu w projekcie: mRNA, lincRNA, antysens, dywergentny (bidirectional_promoter_lncRNA), sense-intronic, inne lncRNA, gospodarz snoRNA, gospodarz miRNA, pseudogen transkr.")
doc("is_lncRNA_project", "True dla klas lncRNA projektu (wszystkie powyzsze klasy lncRNA, w tym gospodarze snoRNA i miRNA).")

# ---------------------------------------------------------------- 1. TPM w kontrolach (RSEM, faza 2)
smp = pd.read_csv("data/meta/phase2_samples.tsv", sep="\t", dtype=str)
ctrl = smp[(smp["condition"] == "control")].drop_duplicates(["sample_experiment", "bio_rep"])
ctrl = ctrl[ctrl["path"].map(lambda p: Path(str(p)).exists())]
for lib, tag in [("poliA", "tpm_polyA"), ("calkowite RNA", "tpm_total")]:
    for cell in CELLS:
        g = ctrl[(ctrl["cell"] == cell) & (ctrl["biblioteka"] == lib)]
        vals = []
        for p in g["path"]:
            q = pd.read_csv(p, sep="\t", usecols=["gene_id", "TPM"])
            q["gene"] = strip(q["gene_id"])
            vals.append(q.groupby("gene")["TPM"].sum())
        if vals:
            m = pd.concat(vals, axis=1)
            G[f"{tag}_{cell}"] = m.median(axis=1).reindex(G.index)
        else:
            G[f"{tag}_{cell}"] = np.nan
        G[f"n_ctrl_{tag.split('_')[1]}_{cell}"] = len(vals)
        print(f"   {tag} {cell}: {len(vals)} bibliotek kontrolnych ({g['sample_experiment'].nunique()} eksperymentow kontrolnych)")
doc("tpm_polyA_<linia>", "Mediana TPM (RSEM, ENCODE, GENCODE v29) z bibliotek poli(A) eksperymentow kontrolnych (kontrolne shRNA lub CRISPR, bez wyciszenia celu), kazda replika osobno.")
doc("tpm_total_<linia>", "Jak wyzej, z bibliotek calkowitego RNA (kontrole 16 eksperymentow na calkowitym RNA); NA, jesli brak.")
doc("n_ctrl_polyA_<linia> / n_ctrl_total_<linia>", "Liczba bibliotek kontrolnych (replik) uzytych do mediany TPM.")

# ---------------------------------------------------------------- 2. Bru-seq: synteza i stabilnosc
kin = pd.read_csv("results/phase1/bru/gene_kinetics.tsv", sep="\t")
kin["gene"] = strip(kin["gene_id"])
BRU = {"ENCSR729WFH": "K562", "ENCSR633UIR": "K562", "ENCSR762OPQ": "K562",
       "ENCSR974AQD": "HepG2", "ENCSR295FEH": "HepG2", "ENCSR135DZR": "HepG2"}   # Bru, BruChase 2 h, BruChase 6 h (skrypt 14)
bf = pd.read_csv("data/meta/files_to_download_v29.tsv", sep="\t", dtype=str).merge(man[["file_accession", "path"]], on="file_accession")
bf = bf[bf["experiment"].isin(BRU) & (bf["output_type"] == "gene quantifications")]
bf = bf[bf["path"].map(lambda p: Path(str(p)).exists())]
for cell in CELLS:
    k = kin[kin["cell"] == cell].drop_duplicates("gene").set_index("gene")
    G[f"synthesis_log2rpkm_{cell}"] = k["synteza"].reindex(G.index)
    G[f"stability_2h_log2_{cell}"] = k["stab_2h"].reindex(G.index)
    G[f"stability_6h_log2_{cell}"] = k["stab_6h"].reindex(G.index)
    reps = bf[bf["experiment"].map(BRU) == cell].groupby("experiment")["bio_rep"].nunique()
    G[f"n_bru_reps_{cell}"] = int(reps.min()) if len(reps) == 3 else np.nan
    print(f"   Bru {cell}: repliki na warunek {reps.to_dict()}")
doc("synthesis_log2rpkm_<linia>", "log2(RPKM + 0,01) w Bru-seq bez poscigu (RNA syntetyzowane w czasie znakowania bromourydyna; srednia replik). Odczyty z nici sensownej genu; RPKM liczone wzgledem wszystkich odczytow sensownych w bibliotece. NA, gdy < 20 odczytow Bru w ktorejkolwiek replice.")
doc("stability_2h_log2_<linia> / stability_6h_log2_<linia>", "log2((RPKM BruChase po 2 lub 6 h + 0,01) / (RPKM Bru-seq + 0,01)); wyzsza = wiecej znakowanego RNA przetrwalo. Miara WZGLEDNA (biblioteki normalizowane osobno), nie okres poltrwania.")
doc("n_bru_reps_<linia>", "Liczba replik w kazdym warunku Bru (Bru, BruChase 2 h, BruChase 6 h); komorki nietraktowane.")

# ---------------------------------------------------------------- 3. frakcjonowanie: jadro/cytoplazma (+ chromatyna, jesli pobrana)
nuc = pd.read_csv("results/phase3/localization/nuclear_index.tsv", sep="\t", index_col=0)
nuc.index = strip(nuc.index)
ff = pd.read_csv("data/meta/fraction_files.tsv", sep="\t", dtype=str).merge(man[["file_accession", "path"]], on="file_accession", how="left")
ff["on_disk"] = ff["path"].map(lambda p: isinstance(p, str) and Path(p).exists())
print("   Pliki frakcjonowania na dysku (linia, frakcja, kwas nukleinowy):")
print(ff[ff["on_disk"]].groupby(["cell", "subcellular_fraction", "nucleic_acid"]).size().to_string())
for cell in CELLS:
    G[f"nuc_cyto_log2_{cell}"] = nuc[cell].reindex(G.index) if cell in nuc.columns else np.nan
    pa = ff[ff["on_disk"] & (ff["cell"] == cell) & (ff["nucleic_acid"] == "polyadenylated mRNA")]
    G[f"n_frac_files_{cell}"] = len(pa[pa["subcellular_fraction"].isin(["nucleus", "cytosol"])])
doc("nuc_cyto_log2_<linia>", "log2((TPM jadro + 0,1) / (TPM cytoplazma + 0,1)) z frakcjonowania RNA poli(A) ENCODE (komorki nietraktowane): srednia log2 TPM z plikow danej frakcji, potem roznica; tylko geny z max(TPM) >= 0,5 w ktorejs frakcji.")
doc("n_frac_files_<linia>", "Liczba plikow kwantyfikacji (jadro + cytoplazma, poli(A)) uzytych do nuc_cyto_log2.")


def frac_tpm(cell, frac):
    g = ff[ff["on_disk"] & (ff["cell"] == cell) & (ff["subcellular_fraction"] == frac)]
    vals = []
    for p in g["path"]:
        q = pd.read_csv(p, sep="\t", usecols=["gene_id", "TPM"])
        q["gene"] = strip(q["gene_id"])
        vals.append(np.log2(q.groupby("gene")["TPM"].sum() + 0.1))
    return (pd.concat(vals, axis=1).mean(axis=1), len(vals)) if vals else (None, 0)


for cell in CELLS:
    ch, nch = frac_tpm(cell, "chromatin")
    npl, nnp = frac_tpm(cell, "nucleoplasm")
    if ch is not None and npl is not None:
        G[f"chromatin_vs_nucleoplasm_log2_{cell}"] = (ch - npl).reindex(G.index)
        print(f"   {cell}: chromatyna/nukleoplazma z {nch} + {nnp} plikow")
    else:
        G[f"chromatin_vs_nucleoplasm_log2_{cell}"] = np.nan
        print(f"   {cell}: brak pobranych plikow frakcji chromatyny i nukleoplazmy -> kolumna NA")
doc("chromatin_vs_nucleoplasm_log2_<linia>", "log2((TPM chromatyna + 0,1) / (TPM nukleoplazma + 0,1)), jesli takie pliki ENCODE sa pobrane w projekcie; w przeciwnym razie NA. TPM sa wzgledne w obrebie biblioteki, wiec to nie jest bezwzgledny udzial frakcji.")

# ---------------------------------------------------------------- 4. splicing na poziomie genu (kontrole pilotazu, bigWig)
reg = pd.read_csv("data/ref/eisa_regions_v29.tsv.gz", sep="\t")
reg["L"] = reg["end"] - reg["start"]
lens = reg.pivot_table(index="gene_id", columns="kind", values="L", aggfunc="sum")
gene_ids = np.sort(reg["gene_id"].unique())
psmp = pd.read_csv("data/meta/pilot_samples.tsv", sep="\t", dtype=str)
pctrl = psmp[psmp["condition"] == "control"].drop_duplicates(["sample_experiment", "bio_rep"])
cnt = Path("results/phase1/eisa/eisa_counts.tsv.gz")
counts = pd.read_csv(cnt, sep="\t") if cnt.exists() else None
for cell in CELLS:
    vals, ireads = [], []
    for e, r in zip(pctrl.loc[pctrl["cell"] == cell, "sample_experiment"], pctrl.loc[pctrl["cell"] == cell, "bio_rep"]):
        f = Path(f"results/phase1/eisa/sums/{e}_{r}.npz")
        if not f.exists():
            continue
        z = np.load(f)
        E, I = pd.Series(z["exon"], index=gene_ids), pd.Series(z["intron"], index=gene_ids)
        ratio = (I / lens["intron"].reindex(gene_ids)) / (E / lens["exon"].reindex(gene_ids))
        vals.append(np.log2(ratio.where((E > 0) & (I > 0))))
        if counts is not None:
            c = counts[counts["sample"] == f"{e}_{r}"].set_index("gene_id")
            ireads.append(c["intron"].reindex(gene_ids))
    if vals:
        idx = pd.concat(vals, axis=1).median(axis=1)
        idx.index = strip(idx.index)
        G[f"splicing_eff_{cell}"] = (-idx).groupby(level=0).first().reindex(G.index)
        if ireads:
            ir = pd.concat(ireads, axis=1).median(axis=1)
            ir.index = strip(ir.index)
            G[f"splicing_intron_reads_{cell}"] = ir.groupby(level=0).first().reindex(G.index)
    else:
        G[f"splicing_eff_{cell}"] = np.nan
    G[f"n_ctrl_splicing_{cell}"] = len(vals)
    print(f"   splicing {cell}: {len(vals)} probek kontrolnych pilotazu")
doc("splicing_eff_<linia>", "-log2(gestosc sygnalu w intronach / gestosc w eksonach): eksony = suma eksonow wszystkich transkryptow genu, introny = cialo genu minus eksony WSZYSTKICH genow na tej samej nici; sygnal = bigWig ENCODE 'signal of unique reads' z wlasciwej nici; mediana z probek kontrolnych pilotazu. Wyzsza = mniej RNA niesplicingowanego wzgledem dojrzalego. Tylko geny wieloeksonowe z >= 100 pz intronow; NA, gdy brak sygnalu w eksonach lub intronach.")
doc("splicing_intron_reads_<linia>", "Mediana liczby odczytow intronowych na probke (sygnal skalibrowany do odczytow RSEM); pozwala odfiltrowac niepewne wartosci splicing_eff (np. < 10).")
doc("n_ctrl_splicing_<linia>", "Liczba probek kontrolnych (replik) pilotazu uzytych do splicing_eff.")

# ---------------------------------------------------------------- 5. dodatkowe cechy genu
pf = pd.read_csv("results/phase3/plasticity/plasticity_features.tsv", sep="\t", index_col=0)
pf.index = strip(pf.index)
G["tau_gtex"] = pf["tau"].reindex(G.index)
G["cgi_promoter"] = pf["CGI"].reindex(G.index)
doc("tau_gtex", "Swoistosc tkankowa tau (Yanai i wsp. 2005) z median TPM GTEx v8 w 52 tkankach (bez hodowli komorkowych), skala log2(TPM+1); 0 = ekspresja wszedzie, 1 = w jednej tkance; NA, gdy max < 0,5.")
doc("cgi_promoter", "1, jesli TSS genu (+-500 pz) naklada sie na wyspe CpG z UCSC cpgIslandExt (hg38), inaczej 0.")
gcf = Path("data/ref/gene_gc_v29.tsv")
if gcf.exists():
    gc = pd.read_csv(gcf, sep="\t").set_index("gene")["gc"]
    G["gc_exonic"] = gc.reindex(G.index)
    doc("gc_exonic", "Zawartosc GC sekwencji eksonowej: srednia GC transkryptow GENCODE v29 wazona dlugoscia (0-1).")

ef = pd.read_csv("data/meta/eclip_files.tsv", sep="\t", dtype=str).merge(man[["file_accession", "path"]], on="file_accession")
ef = ef[ef["path"].map(lambda p: Path(str(p)).exists())]
ef["target"] = ef["target"].replace({"SKIV2L2": "MTREX"})
BIN = 100_000
g = G.reset_index()[["gene_id", "chrom", "start", "end", "strand"]].copy()
g["b0"], g["b1"] = g["start"] // BIN, g["end"] // BIN
g = g.loc[g.index.repeat(g["b1"] - g["b0"] + 1)].copy()
g["bin"] = g["b0"] + g.groupby(level=0).cumcount()
for cell in CELLS:
    rbps, peaks = {}, pd.Series(0, index=G.index, dtype=float)
    sub = ef[ef["cell"] == cell]
    for r in sub.itertuples():
        p = pd.read_csv(r.path, sep="\t", header=None, usecols=range(8), compression="infer")
        p.columns = ["chrom", "start", "end", "name", "score", "strand", "l2fc", "mlog10p"]
        p = p[(p["l2fc"] >= EC_L2FC) & (p["mlog10p"] >= EC_P)]
        p["mid"] = (p["start"] + p["end"]) // 2
        p["bin"] = p["mid"] // BIN
        x = p[["chrom", "strand", "mid", "bin"]].merge(g, on=["chrom", "strand", "bin"])
        x = x[(x["mid"] >= x["start"]) & (x["mid"] < x["end"])]
        peaks = peaks.add(x.groupby("gene_id").size(), fill_value=0)
        rbps.setdefault(r.target, set()).update(x["gene_id"])
    n_rbp = pd.Series(0, index=G.index, dtype=float)
    for s in rbps.values():
        n_rbp.loc[list(s & set(G.index))] += 1
    G[f"eclip_n_rbp_{cell}"] = n_rbp.astype(int)
    G[f"eclip_peaks_per_kb_{cell}"] = (peaks.reindex(G.index).fillna(0) / ((G["end"] - G["start"]) / 1e3)).round(4)
    G[f"eclip_rbps_profiled_{cell}"] = len(rbps)
    print(f"   eCLIP {cell}: {len(rbps)} RBP, {len(sub)} plikow pikow")
doc("eclip_n_rbp_<linia>", "Liczba RBP (z profilowanych eCLIP w tej linii) z >= 1 istotnym pikiem w genie (srodek piku w ciele genu, ta sama nic; log2 wzbogacenia >= 3 i -log10 p >= 3, plik laczacy repliki).")
doc("eclip_peaks_per_kb_<linia>", "Liczba istotnych pikow eCLIP (wszystkie RBP razem) na kb ciala genu (eksony + introny).")
doc("eclip_rbps_profiled_<linia>", "Liczba RBP profilowanych eCLIP w tej linii (mianownik dla eclip_n_rbp).")

for cell in CELLS:
    L = pd.read_csv(f"results/phase2/matrix/lfc_bc_{cell}.tsv.gz", sep="\t", index_col=0)
    resp = L.abs().median(axis=1)
    resp.index = strip(resp.index)
    G[f"rbp_responsiveness_{cell}"] = resp.groupby(level=0).first().reindex(G.index)
doc("rbp_responsiveness_<linia>", "Mediana |log2FC| genu we wszystkich wyciszeniach RBP w tej linii (macierz skorygowana na wspolna kontrole); miara wrazliwosci poziomu genu na zaburzenia sieci RBP, NIE z kontroli.")

# ---------------------------------------------------------------- zapis i podsumowanie
G = G.reset_index().rename(columns={"index": "gene_id"})
first = ["gene_id", "gene_name", "gene_type", "gencode_version", "genome", "gene_id_versioned", "chrom", "start", "end", "strand",
         "project_class", "is_lncRNA_project"]
G = G[first + [c for c in G.columns if c not in first]]
out = OUT / "gene_properties_controls_v29.tsv.gz"
G.to_csv(out, sep="\t", index=False, float_format="%.5g")
pd.DataFrame(DOC).to_csv(OUT / "gene_properties_columns.tsv", sep="\t", index=False)

print(f"\nZapisano {out}: {len(G):,} genow x {G.shape[1]} kolumn; slownik: {OUT / 'gene_properties_columns.tsv'}")
print("\nGeny z wartoscia (ogolem / lncRNA wg klas projektu):")
lnc = G["is_lncRNA_project"]
summ = []
for base in ["tpm_polyA", "tpm_total", "synthesis_log2rpkm", "stability_6h_log2", "nuc_cyto_log2", "chromatin_vs_nucleoplasm_log2",
             "splicing_eff", "eclip_n_rbp", "rbp_responsiveness"]:
    row = {"miara": base}
    for cell in CELLS:
        c = f"{base}_{cell}"
        v = G[c] > 0 if base in ("tpm_polyA", "tpm_total") else G[c].notna()
        if base == "eclip_n_rbp":
            v = G[c] > 0
        row[f"{cell} ogolem"], row[f"{cell} lncRNA"] = int(v.sum()), int((v & lnc).sum())
    summ.append(row)
for c in ["tau_gtex", "cgi_promoter", "gc_exonic"]:
    if c in G.columns:
        v = G[c].notna()
        summ.append({"miara": c, "HepG2 ogolem": int(v.sum()), "HepG2 lncRNA": int((v & lnc).sum()), "K562 ogolem": int(v.sum()), "K562 lncRNA": int((v & lnc).sum())})
s = pd.DataFrame(summ)
s.to_csv(OUT / "gene_properties_counts.tsv", sep="\t", index=False)
print(s.to_string(index=False))
print("   (tpm_polyA: geny z TPM > 0; eclip_n_rbp: geny zwiazane przez >= 1 RBP; pozostale: wartosc nie-NA)")

# ---------------------------------------------------------------- podsumowanie do skopiowania
root = Path.cwd()
lines = ["EKSPORT WLASCIWOSCI GENOW Z PROBEK KONTROLNYCH (projekt lncRNA-RBP, ENCODE)",
         f"Genom GRCh38, adnotacja GENCODE v29; gene_id bez wersji. Katalog projektu: {root}",
         "", "PLIKI",
         f"  {root / out}   ({len(G):,} genow x {G.shape[1]} kolumn)",
         f"  {root / OUT / 'gene_properties_columns.tsv'}   (slownik kolumn)",
         f"  {root / OUT / 'gene_properties_counts.tsv'}   (liczby genow z wartosciami)",
         "", "KOLUMNY (sufiks _HepG2 / _K562 oznacza linie komorkowa)"]
lines += [f"  {d['column']}: {d['definition']}" for d in DOC]
lines += ["", "LICZBA GENOW Z WARTOSCIA (ogolem / lncRNA wg klas projektu)", s.to_string(index=False),
          "  (tpm_*: TPM > 0; eclip_n_rbp: >= 1 RBP; pozostale: wartosc nie-NA)"]
(OUT / "README_export.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
print(f"\nPodsumowanie do skopiowania: {OUT / 'README_export.txt'}")
