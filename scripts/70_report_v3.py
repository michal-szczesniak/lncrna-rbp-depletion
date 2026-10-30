"""
Raport v3 (z 56_report_html.py przez 69_make_report_v3.py): tekst w prostym jezyku (po angielsku), tabele i ryciny w jednym pliku HTML.
Ryciny: fig1, fig2, figS1 z results/phase4/figures_en (skrypt 43); fig3-fig7 z results/phase4/report_v2/figs (skrypt 55);
ryciny locus z results/phase4/cases (skrypt 47).
Wyjscie: results/phase4/report_v2/rbp_lncRNA_report_v2.html
"""
import base64
import html
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

OUT = Path("results/phase4/report_v3")
FIGS = {"fig1": "results/phase4/figures_en/fig1.png", "fig2": "results/phase4/figures_en/fig2.png",
        "figS1": "results/phase4/figures_en/figS1.png", **{f"fig{i}": f"results/phase4/report_v2/figs/fig{i}.png" for i in range(3, 8)}}
LOCUS = Path("results/phase4/cases")
EN_CLASS = {"mRNA": "mRNA", "lincRNA": "lincRNA", "antysens": "NAT", "dywergentny": "divergent (GENCODE)", "inne lncRNA": "other lncRNA",
            "gospodarz snoRNA": "snoRNA host", "gospodarz miRNA": "miRNA host", "sense-intronic": "sense-intronic",
            "pseudogen transkr.": "transcribed pseudogene"}
EN_POS = {"dywergentne": "divergent", "antysensowe": "NAT", "w poblizu": "nearby", "miedzygenowe": "intergenic", "-": "-"}
GROUPS = {"lower": {"splicing machinery": ["WBP11", "SRSF1", "PUF60", "GPKOW", "AQR", "SF1", "RBM22", "SF3A2", "SAP30BP", "DDX39B", "AKAP8L", "PSIP1"],
                    "paraspeckle proteins": ["NONO", "SFPQ"], "cytoplasmic mRNA decay and repression": ["CNOT2", "DDX6", "AGO1", "PUM2"],
                    "translation": ["RPS19", "EIF3G"], "transcription (pausing/termination)": ["PPP1R10"], "non-classical RBPs": ["HSPD1", "XRCC6"]},
          "higher": {"RNA decay": ["MTREX", "CNOT10", "STAU1"], "hnRNP, FET and splicing regulators": ["HNRNPK", "TAF15", "DAZAP1", "SUGP2"],
                     "stress granules": ["G3BP2"], "nucleolus and mitochondria": ["NOL12", "DDX28"], "non-classical RBPs": ["PHF6", "FKBP4"]}}
GROUP_OF = {r: g for d in GROUPS.values() for g, rs in d.items() for r in rs}


def img_src(path):
    p = Path(path)
    return base64.b64encode(p.read_bytes()).decode() if p.exists() else None


def figure(key, path, number, title, read, mean):
    b = img_src(path)
    im = f'<img src="data:image/png;base64,{b}" alt="Figure {number}">' if b else f'<p class="note">[missing image: {html.escape(str(path))}]</p>'
    return f'''<figure id="{key}">{im}<figcaption><p class="ft">Figure {number}. {title}</p>
<p><span class="lead">How to read it.</span> {read}</p><p><span class="lead">What it means.</span> {mean}</p></figcaption></figure>'''


def table(df, fmt=None, cls_=""):
    fmt = fmt or {}
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in df.columns)
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if isinstance(v, (bool, np.bool_)):
                cells.append(f"<td>{'yes' if v else 'no'}</td>")
            elif isinstance(v, (int, np.integer)):
                cells.append(f'<td class="n">{v:,}</td>')
            elif isinstance(v, (float, np.floating)):
                cells.append(f'<td class="n">{"" if pd.isna(v) else fmt.get(c, "{:.2f}").format(v)}</td>')
            else:
                cells.append(f"<td>{html.escape(str(v))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f'<div class="tbl {cls_}"><table><thead><tr>{head}</tr></thead><tbody>{"".join(rows)}</tbody></table></div>'


def read(path, **kw):
    p = Path(path)
    return pd.read_csv(p, sep="\t", **kw) if p.exists() else None


# ================================================================ tabele
summ = read("results/phase2/deseq2/summary_all.tsv")
t_data = summ.groupby(["method", "cell", "biblioteka"]).size().unstack("cell").fillna(0).astype(int).reset_index()
t_data["biblioteka"] = t_data["biblioteka"].map({"poliA": "poly(A) RNA (analysed)", "calkowite RNA": "total RNA (not used)"}).fillna(t_data["biblioteka"])
t_data["method"] = t_data["method"].map({"CRISPR": "CRISPR knockout", "shRNA": "shRNA knockdown"}).fillna(t_data["method"])
t_data.columns = ["How the RBP was removed", "RNA sequenced"] + [{"HepG2": "HepG2 (liver cancer)", "K562": "K562 (leukaemia)"}.get(c, c) for c in t_data.columns[2:]]
n_kd, n_rbp = len(summ), summ["target"].nunique()
n_both = int((summ.groupby("target")["cell"].nunique() == 2).sum())

rt = read("results/phase4/stats/replication_tests.tsv")
sig = read("results/phase4/stats/significance_per_experiment.tsv")
t_sig = sig.rename(columns={"linia": "Cell line", "eksperymenty": "Experiments", "q<0.05": "Significant (q < 0.05)",
                            "q<0.05 i |efekt|>=0.1": "Significant and ≥ 0.1 log2", "w tym lncRNA wyzej": "of these: lncRNAs higher",
                            "w tym lncRNA nizej": "of these: lncRNAs lower"})
r41 = read("results/phase4/robustness_summary.tsv")
vmap = {"glowny": "Main analysis", "maska 50 kb CRISPR": "Also ignore genes within 50 kb of a CRISPR-targeted gene",
        "bez slabych shRNA": "Leave out shRNA experiments in which the RBP mRNA fell by less than 30%",
        "K = 5": "5 instead of 10 reference mRNAs", "K = 20": "20 instead of 10 reference mRNAs",
        "ekspresja >= 50": "Well-expressed genes only", "tylko shRNA": "Only RBPs removed by shRNA in both lines",
        "tylko CRISPR": "Only RBPs removed by CRISPR in both lines"}
t_rob = r41.assign(wariant=r41["wariant"].map(vmap).fillna(r41["wariant"]))[["wariant", "RBP_w_obu", "replikacja_rho", "mediana_|roznicy|", "RBP_z_replikacja"]]
t_rob.columns = ["Version of the analysis", "RBPs in both lines", "Agreement between lines (rho)", "Typical class effect (log2)", "RBPs with a clear effect in both lines"]

pr = read("results/phase4/network/profile_replication.tsv")
t_prof = pr.rename(columns={"wariant": "Measure", "geny": "Genes", "RBP": "RBPs", "n": "n", "mediana r_self": "Same RBP in both lines (median r)",
                            "mediana r z innymi": "Different RBPs (median r)", "% sam na 1. miejscu": "% best match is itself",
                            "% sam w top 5%": "% itself in top 5%"})
t_prof["Genes"] = t_prof["Genes"].replace({"mRNA": "mRNA, same expression"})
t_prof["RBPs"] = t_prof["RBPs"].astype(str).str.replace("wszystkie RBP", "all").str.replace("silne (>= 2000 genow w obu)", "strong (≥ 2000 genes changed)", regex=False)

sp = read("results/phase4/stats/spliceosome_by_splicing_efficiency.tsv")
t_spl = sp.rename(columns={"linia": "Cell line", "przedzial wydajnosci splicingu": "Splicing speed in controls (bin)",
                           "lncRNA (n)": "lncRNAs", "kumulacja intronow mRNA": "Unspliced RNA increase, mRNA",
                           "kumulacja intronow lncRNA": "Unspliced RNA increase, lncRNA", "q": "q"}).drop(columns=["mRNA (n)", "p (Mann-Whitney)"])
t_spl["Splicing speed in controls (bin)"] = t_spl["Splicing speed in controls (bin)"].astype(str).str.replace("najwydajniej", "fastest").str.replace("najmniej wydajnie", "slowest")

cat = read("results/phase2/classes/functional_categories.tsv")
t_cat = cat[cat["typ"] == "funkcja"].sort_values("p")[["kategoria", "n", "mediana", "mediana_pozostalych", "q"]]
t_cat.columns = ["Function (ENCODE annotation)", "RBPs", "Median class effect", "Median of all other RBPs", "q"]

cls = read("data/ref/gene_classes_v29.tsv")
cls["gene"] = cls["gene_id"].str.split(".").str[0]
k_of = cls.drop_duplicates("gene").set_index("gene")["klasa"]
nuc = read("results/phase3/localization/nuclear_index.tsv", index_col=0)
pf = read("results/phase3/plasticity/plasticity_features.tsv", index_col=0)
order = ["mRNA", "lincRNA", "antysens", "inne lncRNA", "gospodarz miRNA", "gospodarz snoRNA"]
kn = k_of.reindex(nuc.index)
t_sno = pd.DataFrame({"Gene class": [EN_CLASS[c] for c in order],
                      "Nuclear enrichment K562": [nuc.loc[kn == c, "K562"].median() for c in order],
                      "Nuclear enrichment HepG2": [nuc.loc[kn == c, "HepG2"].median() for c in order],
                      "Tissue specificity (tau)": [pf.loc[pf["klasa"] == c, "tau"].median() for c in order],
                      "% CpG-island promoter": [100 * pf.loc[pf["klasa"] == c, "CGI"].mean() for c in order]})

sv = read("results/phase4/network/surveillance_vs_others.tsv")
t_sv = sv.assign(kategoria=sv["kategoria"].map(EN_POS))[["linia", "kategoria", "nadzor (n)", "mediana nadzor", "mediana pozostale", "q"]]
t_sv.columns = ["Cell line", "lncRNA category", "Surveillance RBPs", "Median effect, surveillance", "Median effect, all other RBPs", "q"]

nb = read("results/phase4/network/neighbour_coresponse_summary.tsv")
t_nb = nb.assign(typ=nb["typ"].map(EN_POS))[["linia", "para", "typ", "pary", "mediana r", "mediana r tla", "% r > 0.3", "% r > 0.3 w tle"]]
t_nb.columns = ["Cell line", "Pair", "Position", "Pairs", "Median r", "Median r, background", "% r > 0.3", "% r > 0.3, background"]

prof = read("results/phase4/rbp_profiles/rbp_profiles.tsv", index_col=0)
t35 = prof.reset_index()[["RBP", "class_effect_K562", "class_effect_HepG2", "metoda_K562", "metoda_HepG2",
                          "lncRNA_rosna (K562/HepG2 log2FC)", "lncRNA_spadaja (K562/HepG2 log2FC)"]]
t35.insert(1, "Group", t35["RBP"].map(GROUP_OF).fillna("-"))
t35.columns = ["RBP", "Group", "Class effect K562", "Class effect HepG2", "Method K562", "Method HepG2",
               "Most specific lncRNAs going up (log2FC K562/HepG2)", "Most specific lncRNAs going down"]

spec = read("results/phase4/specific/specific_lncRNA.tsv", index_col=0)
t_spec = spec[["lncRNA", "klasa", "polozenie", "rbp1_K562", "log2FC_K562", "log2FC_HepG2", "dom_K562", "dom_HepG2"]].copy()
t_spec["klasa"] = t_spec["klasa"].map(EN_CLASS).fillna(t_spec["klasa"])
t_spec["polozenie"] = t_spec["polozenie"].map(EN_POS).fillna(t_spec["polozenie"])
t_spec.columns = ["lncRNA", "Class", "Position", "Dominant RBP", "log2FC K562", "log2FC HepG2", "Dominance K562", "Dominance HepG2"]

# podobienstwa: cechy (jak w rycinie 3)
rv = read("results/phase2/classes/robustness_variants.tsv.gz")
loc = read("results/phase3/localization/class_effect_B_vs_C.tsv")
pla = read("results/phase3/plasticity/class_effect_C_vs_D.tsv")
ecl = read("results/phase3/eclip/eclip_directness.tsv")
e2 = pd.concat([ecl[["linia", "RBP", "klasa_wszystkie"]].rename(columns={"RBP": "target", "klasa_wszystkie": "roznica"}).assign(wariant="b"),
                ecl[["linia", "RBP", "klasa_niezwiazane"]].rename(columns={"RBP": "target", "klasa_niezwiazane": "roznica"}).assign(wariant="a")])


def rs(df, v):
    s = df[df["wariant"] == v]
    w = s.groupby(["target", "linia"])["roznica"].mean().unstack("linia").dropna()
    return spearmanr(w["K562"], w["HepG2"])[0], s["roznica"].abs().median()


feat = []
for name, df, a, b in [("Splicing efficiency and stability (on top of expression and length)", rv, "A_na_zbiorze_B", "B"),
                       ("Location in the cell (nucleus vs cytoplasm)", loc, "B", "C"),
                       ("Tissue specificity and CpG-island promoter (all seven properties together)", pla, "C", "D"),
                       ("Direct binding: only genes not bound by the removed RBP (eCLIP)", e2, "b", "a")]:
    (ra, ea), (rb, eb_) = rs(df, a), rs(df, b)
    feat.append({"Property added to the comparison": name, "Agreement before": ra, "Agreement after": rb,
                 "Typical class effect before": ea, "Typical class effect after": eb_})
t_feat = pd.DataFrame(feat)

cis = read("results/phase3/cis_nat/cis_effects.tsv")
cats = ["NAT", "naklad. ta sama nic", "sasiad <=10 kb, ta sama nic", "sasiad <=10 kb, przeciwna nic", "sasiad 10-50 kb"]
cats_en = ["antisense partner (NAT)", "overlapping, same strand", "≤10 kb, same strand", "≤10 kb, opposite strand", "10–50 kb"]
g7 = cis[cis["grupa"] == "mRNA"].groupby(["kategoria", "metoda"]).agg(
    med=("delta", "median"), down=("padj", lambda x: 100 * ((x < 0.05) & (cis.loc[x.index, "lfc"] < 0)).mean())).reset_index()
t_cis = g7.pivot_table(index="kategoria", columns="metoda", values=["med", "down"]).reindex(cats)
t_cis.columns = [f"{'Median change' if a == 'med' else '% clearly lower'}, {b}" for a, b in t_cis.columns]
t_cis.index = cats_en
t_cis = t_cis.reset_index().rename(columns={"index": "Protein-coding gene near the removed RBP gene"})

TESTS = pd.DataFrame([
    ["Is the class effect of one experiment different from zero?", "Wilcoxon signed-rank on per-lncRNA differences", "~815 experiments", "Benjamini-Hochberg within each cell line", "p values are optimistic because genes are not independent; used only as supporting evidence"],
    ["Does the class effect of the same RBP agree between cell lines?", "Spearman correlation; permutation test (10,000); bootstrap 95% CI", "1 main test (+ robustness variants)", "none needed (single main test)", "main evidence"],
    ["Clear effect in both lines (35 RBPs)", "per-experiment test + effect ≥ 0.1 log2 in both lines, same direction", "342 RBPs", "Benjamini-Hochberg; chance level estimated by shuffling RBP labels (~12)", ""],
    ["Do RBP functional groups differ from other RBPs?", "Mann-Whitney", "~20 groups", "Benjamini-Hochberg", ""],
    ["Do lncRNAs and mRNAs differ in unspliced RNA at the same splicing speed?", "Mann-Whitney per bin; rank regression", "10 bins", "Benjamini-Hochberg", ""],
    ["Do snoRNA hosts differ from mRNAs and lincRNAs (Fig. 5)?", "Mann-Whitney or Fisher", "8", "Benjamini-Hochberg", "as a group not more sensitive to NMD than other lncRNAs"],
    ["Does GC content of the RNA explain the class effect?", "class effect with GC matched, or after removing the GC trend; agreement between lines (Spearman, bootstrap 95% CI)", "2 variants", "none (sensitivity analysis)", "not explained by GC"],
    ["Which removals of NMD proteins disrupted NMD?", "change of 12 known NMD targets (DESeq2)", "8 removals", "DESeq2 adjusted p per gene", "3 of 8 disrupted NMD"],
    ["Do lncRNAs rise more than mRNAs after removal of NMD proteins?", "Mann-Whitney per expression bin, each experiment separately", "8 × 5 bins", "Benjamini-Hochberg within each experiment", "rise not specific to NMD"],

    ["Do nuclear surveillance RBPs change lncRNA categories more than other RBPs?", "Mann-Whitney", "8", "Benjamini-Hochberg", ""],
    ["Do neighbouring genes change together more than random pairs?", "Mann-Whitney vs expression-matched background", "10", "Benjamini-Hochberg (all p < 10⁻¹⁰)", ""],
    ["Do lncRNAs more often than mRNAs have one dominant regulator?", "Fisher", "1", "none needed", "negative result"],
    ["Termination / 3′ end theme (pre-specified)", "permutation test vs random RBP groups (10,000)", "1 main + 1 held-out", "none needed (planned in advance)", "negative result"],
    ["Do RBP-RBP similarity networks agree between data types?", "Mantel permutation test (1,000)", "2", "none", "networks do not replicate across lines; not interpreted"],
], columns=["Question", "Test", "Number of tests", "Correction for multiple testing", "Note"])

# ================================================================ v3: nowe tabele
PUB = Path("results/figures_pub")
gcs = read("results/phase4/followups/gc_summary.tsv")
gcd = read("results/phase4/followups/gc_dependence_per_kd.tsv")
t_gc, gc_rho_B, gc_rho_C = None, float("nan"), float("nan")
if gcs is not None:
    vm = {"B": "Main comparison (expression, length, splicing, stability)", "C": "Same, with GC content also matched",
          "B_gc": "Main comparison after removing the GC trend from every experiment"}
    t_gc = gcs.assign(wariant=gcs["wariant"].map(vm))[["wariant", "geny_lncRNA_K562", "mediana_class_effect", "rho_miedzy_liniami", "CI95"]]
    t_gc.columns = ["Version of the comparison", "lncRNAs compared (K562)", "Median class effect (log2)", "Agreement between lines (rho)", "95% CI"]
    gi = gcs.set_index("wariant")["rho_miedzy_liniami"]
    gc_rho_B, gc_rho_C = float(gi.get("B", float("nan"))), float(gi.get("C", float("nan")))
n_gcdep = int((gcd["rho_lfc_GC_mRNA"].abs() > 0.2).sum()) if gcd is not None else 0
n_gcall = len(gcd) if gcd is not None else 0

nmd = read("results/phase4/followups/nmd_skuteczne.tsv")
nmdw = read("results/figures_pub/figS2/c_NMD_wyciszenia.tsv")
t_nmd = None
if nmd is not None and nmdw is not None:
    m_ = nmd.merge(nmdw[["kd_experiment", "spadek_mRNA_RBP"]], on="kd_experiment", how="left")
    m_ = m_.sort_values(["skuteczne_NMD", "srednia_roznica"], ascending=[False, False])
    t_nmd = pd.DataFrame({"Removal": m_["wyciszenie"].values,
                          "Disrupted NMD": m_["skuteczne_NMD"].astype(bool).values,
                          "RBP mRNA change (log2)": m_["spadek_mRNA_RBP"].values,
                          "Known NMD targets raised (of 12)": m_["cele_NMD_w_gore"].fillna(0).astype(int).values,
                          "lncRNA minus mRNA, mature RNA (log2)": m_["srednia_roznica"].values,
                          "Bins with lncRNAs significantly higher (of 5)": m_["przedzialy_lnc>mRNA_q<0.05"].astype(int).values})

kp_ = read("results/figures_pub/fig4/d_KCNQ1OT1_profil.tsv")
t_kprof = None
if kp_ is not None:
    seg_ = [c for c in kp_.columns if c.startswith("odc")]
    t_kprof = kp_.assign(Removal=kp_["RBP"] + " " + kp_["linia"],
                         Sample=kp_["warunek"].map({"kontrola": "control", "wyciszenie": "RBP removed"}))[["Removal", "Sample"] + seg_]
    t_kprof.columns = ["Removal", "Sample"] + [f"seg. {i + 1}" for i in range(len(seg_))]

CHANGES = """<div class="changes"><p class="ft">What changed since version 2</p><ul>
<li><strong>Nonsense-mediated decay (NMD).</strong> Version 2 said that removing NMD proteins raises lncRNAs. After checking which removals actually disrupted NMD, we found a similar rise also after knockouts that leave NMD intact, so the rise is not specific to NMD (Figures 4b, 4c).</li>
<li><strong>snoRNA host genes</strong> as a group are not more sensitive to NMD than other lncRNAs; only individual hosts (ZFAS1, SNHG5) respond strongly (Result 4).</li>
<li><strong>Control correction.</strong> It changes the result for only a few RBP–lncRNA pairs from the literature, not “many” (Figure S3).</li>
<li><strong>New check: GC content</strong> of the RNA does not explain the class effect (Result 1).</li>
<li><strong>KCNQ1OT1.</strong> In HepG2 its signal fades gradually along the gene; K562 makes a shorter transcript already in control cells (Result 5).</li>
<li><strong>XIST and CIZ1.</strong> Published studies report no change in XIST levels after loss of CIZ1; our K562 result differs (Result 6).</li>
<li><strong>Terminology.</strong> lncRNAs overlapping a protein-coding gene on the other strand are now called NAT (natural antisense transcripts), as in the figures prepared for publication.</li>
<li><strong>Literature.</strong> Key findings are now compared with published work (numbers in square brackets; list at the end). Locus figures now also show the gene's level in each replicate.</li>
</ul></div>"""

REFS = """<ol class="refs">
<li>Van Nostrand EL et al. A large-scale binding and functional map of human RNA-binding proteins. <em>Nature</em> 2020;583:711–719.</li>
<li>Schlackow M et al. Distinctive patterns of transcription and RNA processing for human lincRNAs. <em>Mol Cell</em> 2017;65:25–38.</li>
<li>Flynn RA et al. Antisense RNA polymerase II divergent transcripts are P-TEFb dependent and substrates for the RNA exosome. <em>PNAS</em> 2011;108:10460–10465.</li>
<li>Estell C et al. A restrictor complex of ZC3H4, WDR82, and ARS2 integrates with PNUTS to control unproductive transcription. <em>Mol Cell</em> 2023;83:2222–2239.</li>
<li>Russo M et al. Restrictor synergizes with Symplekin and PNUTS to terminate extragenic transcription. <em>Genes Dev</em> 2023;37:1017–1040.</li>
<li>Tanu T et al. hnRNPH1-MTR4 complex-mediated regulation of NEAT1v2 stability is critical for IL8 expression. <em>RNA Biol</em> 2021;18(sup1):537–547.</li>
<li>Lykke-Andersen S et al. Human nonsense-mediated RNA decay initiates widely by endonucleolysis and targets snoRNA host genes. <em>Genes Dev</em> 2014;28:2498–2517.</li>
<li>Colombo M et al. Transcriptome-wide identification of NMD-targeted human mRNAs reveals extensive redundancy between SMG6- and SMG7-mediated degradation pathways. <em>RNA</em> 2017;23:189–201.</li>
<li>Smith CM, Steitz JA. Classification of gas5 as a multi-small-nucleolar-RNA (snoRNA) host gene and a member of the 5′-terminal oligopyrimidine gene family reveals common features of snoRNA host genes. <em>Mol Cell Biol</em> 1998;18:6897–6909.</li>
<li>Ridings-Figueroa R et al. The nuclear matrix protein CIZ1 facilitates localization of Xist RNA to the inactive X-chromosome territory. <em>Genes Dev</em> 2017;31:876–888.</li>
<li>Sunwoo H et al. Repeat E anchors Xist RNA to the inactive X chromosomal compartment through CDKN1A-interacting protein (CIZ1). <em>PNAS</em> 2017;114:10654–10659.</li>
<li>Turvey GL et al. Epigenetic deprogramming by disruption of CIZ1-RNA nuclear assemblies in early-stage breast cancers. <em>J Cell Biol</em> 2025;224:e202409123.</li>
<li>Korostowski L, Sedlak N, Engel N. The Kcnq1ot1 long non-coding RNA affects chromatin conformation and expression of Kcnq1, but does not regulate its imprinting in the developing heart. <em>PLoS Genet</em> 2012;8:e1002956.</li>
<li>Mancini-DiNardo D et al. Elongation of the Kcnq1ot1 transcript is required for genomic imprinting of neighboring genes. <em>Genes Dev</em> 2006;20:1268–1282.</li>
<li>He S et al. DNA break induces rapid transcription repression mediated by proteasome-dependent RNAPII removal. <em>Cell Rep</em> 2024;43:114420.</li>
<li>Kosicki M, Tomberg K, Bradley A. Repair of double-strand breaks induced by CRISPR-Cas9 leads to large deletions and complex rearrangements. <em>Nat Biotechnol</em> 2018;36:765–771.</li>
<li>Garbulowski M et al. Comprehensive analysis of the RBP regulome reveals functional modules and drug candidates in liver cancer. <em>Sci Rep</em> 2026;16:19626.</li>
</ol>"""

NEW_READ = ("Gene model on top; below, RNA-seq signal on the gene's strand (reads per million, mean of replicates) in control cells (grey) "
            "and after removal of the RBP (coloured line; blue K562, orange HepG2), one track per experiment. On the right, the level of the gene "
            "in each replicate (control vs removal), with its change (log2), adjusted p value and percentile among all removal experiments in that cell line.")
OLD_LAYOUT = "Older layout: tracks as in Figure 5e, without the replicate panel; numbers are given in the label of each track."

# ================================================================ tekst
P = lambda x: f"{x:.2f}"
rho, lo, hi, pm = rt.iloc[0]["rho"], rt.iloc[0]["ci_lo"], rt.iloc[0]["ci_hi"], rt.iloc[0]["p"]
rho2, lo2, hi2, n2 = rt.iloc[1]["rho"], rt.iloc[1]["ci_lo"], rt.iloc[1]["ci_hi"], int(rt.iloc[1]["n"])

LOG2 = '''<div class="tbl narrow"><table><thead><tr><th>Change (log2)</th><th>Change (%)</th></tr></thead><tbody>
<tr><td class="n">+0.1</td><td class="n">+7%</td></tr><tr><td class="n">+0.4</td><td class="n">+32%</td></tr><tr><td class="n">+1</td><td class="n">+100% (doubling)</td></tr>
<tr><td class="n">−0.1</td><td class="n">−7%</td></tr><tr><td class="n">−0.4</td><td class="n">−24%</td></tr><tr><td class="n">−1</td><td class="n">−50% (halving)</td></tr></tbody></table></div>'''

body = f'''
<div class="col">
<h1>How do long non-coding RNAs respond when RNA-binding proteins are removed?</h1>
<p class="sub">Re-analysis of {n_kd} ENCODE experiments in two human cell lines. Results report, version 3 (3 October 2026).</p>
<p class="answer">When one RNA-binding protein is removed, long non-coding RNAs (lncRNAs) change differently from protein-coding RNAs (mRNAs) that resemble them. The direction and size of this difference depend on which protein was removed, and the same protein produces a similar difference in two unrelated cell lines. Seven known properties of lncRNAs do not explain it, and neither does direct binding of the protein. Nor does the GC content of the RNA. Two processes stand out: the splicing machinery and the nuclear RNA exosome, which removes most strongly the lncRNAs that share a promoter region with a protein-coding gene.</p>
{CHANGES}

<h2 id="q">The questions and the logic</h2>
<p>RNA-binding proteins (RBPs) process, transport, stabilise and degrade RNA. The ENCODE consortium removed hundreds of RBPs one at a time and sequenced all RNA afterwards. These data have been analysed mostly for mRNAs. We asked three questions:</p>
<ol>
<li><strong>Do lncRNAs respond to the removal of an RBP differently from comparable mRNAs?</strong></li>
<li><strong>If they do, is that only because lncRNAs are known to differ from mRNAs</strong> (lower expression, slower splicing, faster decay, more nuclear location), or is there more to it?</li>
<li><strong>Which RBPs, processes and individual lncRNAs are involved?</strong></li>
</ol>
<h3>The class effect</h3>
<p>For each experiment we calculate one number, the <em>class effect</em>. Each lncRNA is paired with the 10 mRNAs most similar to it in expression level, gene length, splicing efficiency and stability. Example: after removing RBP X, a lncRNA falls by 20% while its 10 look-alike mRNAs fall by 5% on average; this lncRNA fell about 15% more than comparable mRNAs. We repeat this for every lncRNA we can measure (typically 1,000–4,000) and take the median. A class effect of zero means lncRNAs behaved like comparable mRNAs; a negative value means they ended up lower; a positive value means they ended up higher. Changes are on the log2 scale:</p>
{LOG2}
<h3>The main evidence: agreement between two cell lines</h3>
<p>A single class effect could be noise. {n_both} RBPs were removed in both cell lines ({int(rt.iloc[0]["n"])} of them with enough measurable lncRNAs in both to be compared), K562 (leukaemia-derived) and HepG2 (liver-cancer-derived), in independent experiments. Each of these RBPs therefore has two class effects, one per cell line. If the class effect reflects what the RBP does, RBPs with strongly negative class effects in K562 should also be among the most negative in HepG2, and likewise for positive ones. We measure this with the Spearman rank correlation, rho: rank the RBPs by class effect in each cell line and compare the two rankings (1 = identical rankings, 0 = unrelated). Throughout this report, “agreement between the two cell lines” means this rho.</p>
<h3>How we test explanations</h3>
<p>If a known property X of lncRNAs (for example their low stability) caused the class effect, then comparing each lncRNA only with mRNAs that have the same value of X should shrink the class effect towards zero, or destroy its agreement between the cell lines. If neither happens, X does not explain the difference.</p>

<h2 id="data">The data</h2>
<p>{n_kd} experiments in which one of {n_rbp} RBPs was removed, each in two replicates with matched control cells. shRNA lowers the RBP's mRNA; CRISPR cuts the RBP's gene. We also used maps of where RBPs bind RNA (eCLIP), RNA production and decay rates (Bru-seq, BruChase), RNA levels in nucleus and cytoplasm, and tissue expression (GTEx).</p>
{table(t_data)}

<h2 id="batch">A technical problem: shared control samples</h2>
<p>Each experiment is compared with control cells grown at the same time, and about ten experiments share one control sample. Any quirk of that control is copied into all of them. This was the strongest pattern in the raw data. For each experiment we subtracted, gene by gene, the typical change seen in the other experiments sharing its control (“control-corrected” data).</p>
</div>
<div class="wide">{figure("fig1", FIGS["fig1"], 1, "Shared control samples dominated the raw data; the correction removes their influence.",
 "Panels a and b: the six strongest patterns shared by many experiments (principal components). Each bar is the share of a pattern explained simply by which control sample was used, before (grey) and after (blue) the correction; the dashed line is chance level. Panel c: class effects of eight well-studied RBPs before (open) and after (filled) the correction.",
 "Before correction the control sample explained 25–68% of the strongest patterns; afterwards no more than chance. Known biology survives: removing MTREX, which delivers RNAs to the nuclear exosome, still raises lncRNAs in both lines. Some apparent effects were artefacts of a shared control (EXOSC10 in K562).")}</div>
<div class="col">

<h2 id="main">Result 1: lncRNAs respond differently from comparable mRNAs, and this is a property of the RBP</h2>
</div>
<div class="wide">{figure("fig2", FIGS["fig2"], 2, "Class effects are small but reproducible between cell lines.",
 f"Panel a: how many experiments had each class effect (dotted lines: ±0.1 log2, about ±7%). Panel b: each dot is one RBP removed in both cell lines; its horizontal position is its class effect in K562 and its vertical position its class effect in HepG2. A class effect summarises all measured lncRNAs relative to their look-alike mRNAs, so each dot is one RBP, not one gene. Coloured dots: the 35 RBPs with a clear effect in both lines. Panel c: agreement between the lines in different versions of the analysis.",
 f"The ranking of RBPs by class effect in K562 predicts the ranking in HepG2: rho = {P(rho)} (95% confidence interval {P(lo)}–{P(hi)}; p = {pm:.0e}; none of 10,000 random re-pairings of RBPs between the lines reached this value). Every correction we applied increased the agreement, so the corrections removed noise rather than biology.")}</div>
<div class="col">
<h3>How strong and how robust is this?</h3>
<p>Class effects are mostly small: typically about 4% (0.056 log2). For individual experiments we tested whether the class effect differs from zero (Wilcoxon test on the per-lncRNA differences, corrected for the number of experiments). About 70% are significant, and 23–30% are both significant and at least 7%:</p>
{table(t_sig)}
<p class="note">This per-experiment test treats lncRNAs as independent, but genes are correlated, so its p values are too optimistic. The agreement between independent experiments in two cell lines is therefore our main evidence.</p>
<p>The agreement survives every change in the analysis we tried, including using only shRNA or only CRISPR experiments:</p>
{table(t_rob, {"Typical class effect (log2)": "{:.3f}"})}
<h3>Does the GC content of the RNA explain it?</h3>
<p>The proportion of G and C nucleotides in an RNA affects how efficiently it is captured during sequencing, and ENCODE noted that changes in these experiments partly depend on it [1]. Our matching did not include GC content, so we checked it separately. In {n_gcdep} of {n_gcall} experiments the changes of mRNAs follow their GC content (|rho| &gt; 0.2), and lncRNAs have slightly less GC than mRNAs (median about 48.5% vs 50%). We repeated the comparison with GC content matched as well, and separately after removing the GC trend from every experiment. The agreement between the two cell lines did not fall; it rose slightly. GC content added noise but did not create the effect:</p>
{table(t_gc, {"Median class effect (log2)": "{:.3f}", "Agreement between lines (rho)": "{:.3f}"}) if t_gc is not None else ""}
<p><strong>35 RBPs</strong> have a clear class effect in both lines and in the same direction: at least 0.1 log2 (7%) and statistically significant in each line. After removing 12 of them lncRNAs end up higher than comparable mRNAs; after removing 23, lower. For the other 307 RBPs the effect is small, seen in only one line, or opposite in the two lines. That does not mean they leave lncRNAs untouched; their effects are too small to call with confidence. Shuffling RBPs between the lines gives about 12 such RBPs by chance, so roughly one in three of the 35 may be a false positive. They are listed at the end.</p>

<h2 id="notexpl">Result 2: known properties of lncRNAs do not explain the difference</h2>
<p>“The difference” here means the class effect and its agreement between cell lines. We added properties to the comparison one after another, so that the last version requires the reference mRNAs to match the lncRNA in all seven properties at once: expression, length, splicing efficiency, stability, location in the cell, tissue specificity and promoter type.</p>
</div>
<div class="wide">{figure("fig3", FIGS["fig3"], 3, "Adding RNA properties, removing bound genes or removing splicing and NMD RBPs does not remove the class effect.",
 "Panels a and b: open circles are the comparison before a property was added, filled circles after; panel a shows agreement between the lines, panel b the typical size of the class effect. Panel c: for RBPs with binding maps (eCLIP), whether RNAs the RBP binds change more after its removal than similar RNAs it does not bind (each dot one RBP; positive = bound genes respond more). Panel d: agreement between lines for all RBPs and after leaving out all RBPs involved in splicing or nonsense-mediated decay, with 95% confidence intervals.",
 f"With all seven properties matched, the class effect is 0.058 log2 and the agreement between lines 0.29, practically unchanged. Splicing efficiency and stability explain about 14% of the effect size; the other properties explain nothing. Bound genes do not respond more strongly than unbound ones, and the class effect is the same on unbound genes. This agrees with ENCODE's own conclusion that many changes after RBP removal are indirect [1]. Leaving out the {int(rt.iloc[0]['n']) - n2} splicing and NMD RBPs, the remaining {n2} RBPs still agree between lines (rho = {P(rho2)}, 95% CI {P(lo2)}–{P(hi2)}).")}</div>
<div class="col">
{table(t_feat, {"Typical class effect before": "{:.3f}", "Typical class effect after": "{:.3f}"}, "w")}
<p>We also checked that lncRNAs are not under-represented in the binding maps: at the same expression level they are bound at least as often as mRNAs. And the response profiles of lncRNAs are not noisier than those of mRNAs with the same expression. When we compare the full response of the same RBP on hundreds of individual genes between the two lines, the RBP is recognisable at least as well from lncRNAs as from mRNAs:</p>
{table(t_prof, {"Same RBP in both lines (median r)": "{:.3f}", "Different RBPs (median r)": "{:.3f}", "% best match is itself": "{:.1f}", "% itself in top 5%": "{:.1f}"}, "w")}
<p class="note">“Best match is itself”: for an RBP in K562, the most similar RBP profile in HepG2 is the same RBP; chance level is about 0.3%.</p>

<h2 id="splice">Result 3: splicing and nonsense-mediated decay</h2>
<p>Of all groups of RBPs with a known function, only the splicing machinery (spliceosome) pushes lncRNAs consistently in one direction: down relative to mRNAs. To find out why, we measured unspliced RNA (reads in introns) and mature RNA (reads in exons) separately in 63 spliceosome removal experiments (30 in K562, 33 in HepG2), without subtracting mRNAs from lncRNAs.</p>
</div>
<div class="wide">{figure("fig4", FIGS["fig4"], 4, "Losing the spliceosome disturbs mRNA processing more than lncRNA processing.",
 "Panel a: functional groups of RBPs (number in brackets); coloured dot = median class effect of the group, grey = all other RBPs; red = significant after correction. Panels b and c: absolute changes for mRNAs (grey) and lncRNAs (green) in five expression bins; b: unspliced RNA after spliceosome removal, c: mature RNA after removal of NMD proteins; solid lines K562, dashed HepG2. Panel d: unspliced RNA after spliceosome removal in five bins of splicing speed measured in control cells; stars mark significant lncRNA–mRNA differences within a bin.",
 "Unspliced RNA builds up much more in mRNAs (0.15–0.33 log2) than in lncRNAs (0.01–0.18), at every expression level and in both lines; the apparent “lncRNA decrease” is mainly an mRNA increase. lncRNAs are spliced more slowly to begin with (5–7 times more unspliced RNA in control cells), which explains part of this (panel d), but at the same splicing speed lncRNAs still build up less unspliced RNA. A likely reason is that unspliced lncRNA precursors are degraded in the nucleus rather than accumulating. Panel c pools all removals of NMD proteins; Figures 4b and 4c below show why this result needs care.")}</div>
<div class="col">
{table(t_spl, {"Unspliced RNA increase, mRNA": "{:.3f}", "Unspliced RNA increase, lncRNA": "{:.3f}", "q": "{:.0e}"}, "w")}
<details><summary>All functional groups</summary>{table(t_cat, {"Median class effect": "{:.3f}", "Median of all other RBPs": "{:.3f}", "q": "{:.3f}"})}</details>
<h3>Is the rise of lncRNAs after removal of NMD proteins specific to NMD?</h3>
<p>Panel c of Figure 4 pools eight experiments in which an NMD protein was removed. We first checked which of them actually disrupted NMD, using 12 genes known to be degraded by NMD (GAS5, SNHG1, SNHG5, SNHG6, SNHG8, ZFAS1, GADD45A, GADD45B, ATF4, SMG5, SMG7, UPF3B). If NMD is disrupted, they should go up.</p>
</div>
<div class="wide">{figure("nmdT", PUB / "figS2" / "c_NMD.png", "4b", "Which removals of NMD proteins actually disrupted NMD?",
 "Each column is one removal experiment (letter in brackets: s = shRNA, C = CRISPR), each row a gene known to be degraded by NMD. Colour = change of the gene after the removal (log2; red = up, blue = down). Framed columns: UPF1.",
 "A removal that disrupts NMD should turn its column red. This happens for UPF1 and UPF2 in K562 and UPF2 in HepG2, hardly at all for the SMG knockouts (these proteins can stand in for one another [8]), and the opposite happens for UPF1 in HepG2, where the UPF1 mRNA fell by only about 30%.")}</div>
<div class="wide">{figure("nmdE", PUB / "fig3" / "d_nmd.png", "4c", "lncRNAs rise relative to mRNAs after removal of NMD proteins, whether or not NMD was disrupted.",
 "Each row is one removal. Large dot: mean difference between lncRNAs and mRNAs of the same expression in mature RNA (log2), averaged over five expression bins; small grey dots: the five bins. Filled dot: the removal disrupted NMD (Figure 4b); open dot: it did not. Colour: cell line (blue K562, orange HepG2). Right: in how many bins lncRNAs were significantly higher.",
 "After the strongest NMD disruption (UPF1 in K562) lncRNAs rise more than mRNAs in all five expression bins. But knockouts of SMG7, SMG8 and SMG9, which leave NMD intact, produce a similar rise. The rise is therefore not specific to NMD: either these knockouts weaken NMD in a way the 12 classic targets do not reveal, or lncRNAs rise relative to mRNAs after many mild disturbances. We no longer claim that NMD controls lncRNAs as a class.")}</div>
<div class="col">
{table(t_nmd, {"RBP mRNA change (log2)": "{:.2f}", "lncRNA minus mRNA, mature RNA (log2)": "{:.3f}"}, "w") if t_nmd is not None else ""}


<h2 id="sno">Result 4: snoRNA host genes are a distinct subgroup</h2>
<p>Some lncRNA genes carry small nucleolar RNAs (snoRNAs) in their introns. These snoRNA host genes kept standing out, so we compared their properties with other lncRNAs.</p>
</div>
<div class="wide">{figure("fig5", FIGS["fig5"], 5, "snoRNA host genes resemble mRNAs more than other lncRNAs.",
 "Violin plots show the distribution per gene class (line = median). a: RNA in nucleus vs cytoplasm (higher = more nuclear); b: tissue specificity across 52 tissues (0 = everywhere, 1 = one tissue); c: share of promoters with a CpG island (typical of always-on genes); d: change after removing NMD/EJC proteins, snoRNA hosts vs other lncRNAs. Brackets give corrected significance.",
 "Compared with lincRNAs, snoRNA hosts are less nuclear, broadly expressed and mostly have CpG-island promoters; in none of these do they differ significantly from mRNAs. All snoRNA host genes carry a 5′TOP motif, typical of mRNAs for ribosomal proteins [9], which fits this mRNA-like behaviour. As a group, however, they are not more sensitive to removal of NMD/EJC proteins than other lncRNAs (panel d, not significant in either line). NMD is known to target snoRNA host transcripts [7], but in our data only individual hosts respond strongly: removing exon junction complex proteins (MAGOH, EIF4A3) raises ZFAS1 and SNHG5 more than almost any other removal (98th–100th percentile; locus figure below). snoRNA hosts should be treated as a separate subgroup, not as typical lncRNAs.")}</div>
<div class="col">
{table(t_sno, {"% CpG-island promoter": "{:.0f}"})}
</div>
<div class="wide">{figure("locZ", PUB / "figS5" / "ZFAS1_EJC.png", "5e", "ZFAS1 rises after removal of exon junction complex proteins.",
 NEW_READ,
 "The rise is seen in all experiments, both cell lines and both removal methods. It fits NMD of the processed host transcript, but the exon junction complex also takes part in splicing, and the one UPF1 removal that clearly disrupted NMD (K562) raises ZFAS1 only modestly, so the mechanism is not proven.")}</div>
<div class="col">

<h2 id="surv">Result 5: the nuclear exosome removes lncRNAs, most of all divergent ones</h2>
<p>We classified lncRNAs by position relative to protein-coding genes: <em>divergent</em> (start within 2 kb of a protein-coding gene start, transcribed in the opposite direction from a shared promoter region), <em>NAT</em> (natural antisense transcript: overlapping a protein-coding gene on the other strand), <em>nearby</em> (within 10 kb) and <em>intergenic</em> (more than 10 kb from any protein-coding gene). For each category we calculated a class effect.</p>
</div>
<div class="wide">{figure("fig6", FIGS["fig6"], 6, "Removing the nuclear exosome raises divergent lncRNAs most.",
 "Panels a and b: class effect of each lncRNA category; boxes show all other RBPs, dots the nuclear surveillance RBPs (exosome subunits and MTREX); stars: corrected significance of surveillance vs other RBPs. Panel c: RBPs whose removal raises divergent lncRNAs most relative to intergenic ones (bar = mean of the two lines, dots = each line).",
 "In HepG2, removing the exosome raises divergent lncRNAs most (+0.38 log2), NAT less (+0.24) and intergenic least (+0.17). In K562 most exosome removals were ineffective (one did not lower the target mRNA, another changed only 19 genes), but the effective MTREX removal shows the same pattern. A study in HeLa cells found the same order after exosome removal: antisense and promoter-upstream transcripts were stabilised more than intergenic lncRNAs [2]. Removing NELFE (which pauses RNA polymerase near promoters) or PPP1R10/PNUTS (transcription termination) also raises divergent lncRNAs relative to intergenic ones in both lines, a hint that both production and degradation control these transcripts. Both proteins are known to restrain transcription near promoters and transcription that does not lead to a gene [3–5].")}</div>
<div class="col">
{table(t_sv, {"Median effect, surveillance": "{:.3f}", "Median effect, all other RBPs": "{:.3f}", "q": "{:.0e}"}, "w")}
</div>
<div class="wide">{figure("locK", PUB / "fig4" / "d_KCNQ1OT1.png", "6d", "KCNQ1OT1 becomes visible along its whole length once the exosome is removed.",
 "As in Figure 5e. KCNQ1OT1 is a long (about 90 kb), unspliced nuclear lncRNA, transcribed from right to left in this view.",
 "In control HepG2 cells KCNQ1OT1 is barely detectable; after removal of MTREX, EXOSC2 or EXOSC5 its signal appears along about 65 kb (99th–100th percentile of all removals). The RNA is produced but normally removed almost immediately by the exosome. The signal fades gradually towards the end of the gene (table below), as expected if transcription of lncRNAs often stops early, at many points along the gene [2]. In K562 the transcript ends after about 27 kb already in control cells, and MTREX removal raises only this short form: K562 seems to make a shorter KCNQ1OT1. Shorter, tissue-specific forms have been described in mouse heart [13], and the length of this RNA matters for its function [14].")}</div>
<div class="col"><p class="note">KCNQ1OT1 signal (reads per million) in ten equal segments of the gene, from its start (seg. 1) to its end (seg. 10).</p>
{table(t_kprof, None, "w") if t_kprof is not None else ""}</div>
<div class="wide">{figure("locN", LOCUS / "locus_NEAT1_MALAT1_exosome.png", "6e", "NEAT1, and in HepG2 also MALAT1, rise after exosome removal.",
 "Older layout: tracks as in Figure 5e, without the replicate panel; the shaded region is present only in the long NEAT1 isoform (NEAT1_2).",
 "NEAT1 rises in both lines (95th and 99th percentile), MALAT1 rises in HepG2 only. Signal also appears in the NEAT1_2-only region, which is otherwise almost invisible in these poly(A) libraries. Mature MALAT1 and NEAT1_2 lack a poly(A) tail, so these data mainly capture their polyadenylated forms or precursors. MTREX, together with hnRNPH1, is known to control the stability of the long NEAT1 isoform [6].")}</div>
<div class="col">

<h2 id="gene">Result 6: individual genes</h2>
<h3>Neighbours change together, but lncRNAs less tightly than protein-coding pairs</h3>
<p>For pairs of neighbouring genes we correlated their changes across about 400 removal experiments in each cell line (co-response, r). Background: the same gene paired with a random mRNA of the same expression on another chromosome.</p>
</div>
<div class="wide">{figure("fig7", FIGS["fig7"], 7, "Gene-level view: co-response of neighbours and RBPs that dominate the response of individual lncRNAs.",
 "Panel a: median co-response for each type of gene pair (bars) and its background (black line). Panel b: for each RBP, the number of genes for which it is the dominant regulator in both cell lines, for lncRNAs and for the same number of mRNAs with the same expression. “Dominant regulator”: the RBP whose removal changes the gene most unusually compared with all other removals.",
 "All neighbour types change together more than random pairs, and the strength of co-response for a given pair is similar in both lines (rho 0.48–0.62). lncRNAs that share a promoter region with a protein-coding gene (divergent pairs) co-respond less tightly than pairs of protein-coding genes sharing a promoter (0.05–0.09 vs 0.17–0.19), consistent with extra control of these lncRNAs after transcription, for example by the exosome. Overall, lncRNAs do not have a single dominant regulator more often than mRNAs (4.9% vs 3.9%, p = 0.43), but SRSF1 is the dominant regulator of 7 lncRNAs and of none of the matched mRNAs.")}</div>
<div class="col">
{table(t_nb, {"Median r": "{:.3f}", "Median r, background": "{:.3f}", "% r > 0.3": "{:.1f}", "% r > 0.3, background": "{:.1f}"}, "w")}
<p class="note">The “nearby” category includes lncRNAs inside introns of a protein-coding gene on the same strand (sense-intronic); their reads partly come from the host gene's precursor, which inflates co-response.</p>
<h3>lncRNAs with a specific, reproducible regulator</h3>
<p>22 lncRNAs have the same dominant regulator in both cell lines, with a strong change in both. Dominance is how many times stronger the response to the dominant RBP is than to the second strongest; values near 1 mean “strongest”, not “exclusive”.</p>
{table(t_spec, {"Dominance K562": "{:.2f}", "Dominance HepG2": "{:.2f}"}, "w")}
<h3>Case studies</h3>
</div>
<div class="wide">{figure("locNO", PUB / "fig5" / "c_NORAD.png", "7c", "NORAD falls after removal of the same three RBPs in both cell lines.",
 "As in Figure 5e.", "SRSF1, RCC2 and EIF3D lower NORAD by 55–65% evenly along the transcript; in each line they are the strongest NORAD-lowering removals of all (0–1st percentile). None of them has eCLIP peaks in NORAD, so the effect is probably indirect. PUM1 and PUM2, the proteins NORAD is known to bind, do not change NORAD.")}</div>
<div class="wide">{figure("locX", PUB / "figS5" / "XIST_CIZ1.png", "7d", "XIST rises after knockout of CIZ1 (K562).",
 "As in Figure 5e.", "CIZ1, known to anchor XIST to the inactive X chromosome, is the removal that raises XIST most of all (+0.57 log2, 100th percentile). Published work describes CIZ1 as anchoring XIST to the inactive X and reports no change in XIST levels when CIZ1 is lost, in mouse fibroblasts [10, 11] and in human breast cancer cells [12]. Our observation therefore differs from previous studies; it may be specific to K562 or indirect, and needs independent confirmation.")}</div>
<div class="wide">{figure("locNN", LOCUS / "locus_NEAT1_NONO.png", "7e", "A cautionary case: NONO and NEAT1.",
 "Older layout: tracks as in Figure 5e, without the replicate panel; the bottom track shows NONO binding sites (eCLIP, K562).", "NONO binds almost exclusively the NEAT1_2-only region, which poly(A) RNA-seq barely captures. The known effect of NONO on NEAT1_2 can therefore be neither confirmed nor excluded with these data.")}</div>
<div class="wide">{figure("locO", PUB / "figS5" / "OIP5-AS1_HuR.png", "7f", "A negative case: HuR and OIP5-AS1.",
 "As in Figure 5e.", "OIP5-AS1 does not change after knockout of HuR (ELAVL1), a protein it is known to bind, and HuR eCLIP shows no peaks in OIP5-AS1, consistent with OIP5-AS1 acting on HuR rather than the reverse. Most RBP–lncRNA pairs from the literature behaved like this: binding or regulation by the lncRNA does not imply that the RBP controls the lncRNA's level.")}</div>
<div class="col">

<h2 id="method">Methodological lessons</h2>
<ul class="find">
<li><strong>CRISPR knockouts lower neighbouring genes up to at least 50 kb away</strong> (Figure S1); shRNA does not. A DNA cut can switch off transcription hundreds of kilobases around it [15], and its repair often deletes several kilobases [16]; either could contribute. We excluded such genes from all analyses.</li>
<li><strong>ENCODE quality audits do not flag ineffective knockdowns</strong>; all 876 experiments carry the same general warnings. We applied our own criteria.</li>
<li><strong>For CRISPR knockouts, the level of the RBP's mRNA is not a measure of success</strong>: in several knockouts it did not fall, yet thousands of genes changed.</li>
<li><strong>Poly(A) RNA-seq barely sees non-polyadenylated lncRNAs</strong> such as mature MALAT1 and NEAT1_2.</li>
<li><strong>The control correction changes the result for only a few RBP–lncRNA pairs from the literature</strong>, but for some it matters: RBM15 → XIST (−0.61 log2, p = 6·10⁻²⁴ before; −0.02 after; Figure S3).</li>
<li><strong>Changes after RBP removal often depend on the GC content of the RNA</strong> ({n_gcdep} of {n_gcall} experiments), as ENCODE noted [1]. It does not explain the class effect, but comparisons between groups of genes should take it into account.</li>
<li><strong>Removing an NMD protein does not always disrupt NMD.</strong> Single knockouts of SMG5, SMG7, SMG8 or SMG9 hardly changed known NMD targets, and UPF1 removal in HepG2 lowered rather than raised them (Figure 4b). Checking known targets is a simple test worth doing before interpreting such experiments.</li>
</ul>
</div>
<div class="wide">{figure("figS1", PUB / "figS1" / "a_sasiedzi.png", "S1", "CRISPR, but not shRNA, lowers genes near the removed RBP's gene.",
 "Change of protein-coding genes at a given position relative to the removed RBP's gene, compared with similar genes elsewhere, after CRISPR (blue) or shRNA (pink): violin = distribution, white dot and bar = median and middle half. Stars: change different from zero. Below each category: % of genes clearly lower (CRISPR | shRNA).",
 "After CRISPR, protein-coding neighbours drop by 0.1–0.3 log2, even 10–50 kb away; shRNA lowers only genes that share sequence with its target.")}</div>
<div class="col">
{table(t_cis, {c: "{:.1f}" if c.startswith("%") else "{:.2f}" for c in t_cis.columns}, "w")}

<h2 id="neg">What did not work, or was not supported</h2>
<ul class="find">
<li><strong>RBP–RBP similarity networks</strong> built from gene-level responses do not agree between the two cell lines (rho 0.05–0.07), neither for lncRNAs nor for mRNAs. Profiles of single RBPs replicate only weakly (r ≈ 0.1–0.2), and similarities between different RBPs are too weak to survive. We do not interpret these networks. A published network analysis of the same data estimated a very low signal-to-noise ratio [17].</li>
<li><strong>A unifying “transcription termination and 3′ end” theme</strong>, suggested by several individual observations, was tested with a test planned in advance on a group of 33 RBPs defined from annotation, and was not supported (p = 0.33; without the RBPs that suggested it, p = 0.62).</li>
<li><strong>Location in the cell and tissue specificity</strong> do not explain the class effect.</li>
<li><strong>Most RBP–lncRNA pairs from the literature</strong> (HNRNPU/HNRNPK → XIST, PUM → NORAD, IGF2BP → H19, HuR → OIP5-AS1) are not reproduced as changes in lncRNA levels (Figure S3).</li>
<li><strong>A specific effect of NMD on lncRNAs</strong>: the rise of lncRNAs after removal of NMD proteins also follows knockouts that leave NMD intact (Figure 4c).</li>
<li><strong>snoRNA hosts as a class sensitive to NMD</strong>: not supported (Figure 5d); only individual hosts respond.</li>
</ul>
</div>
<div class="wide">{figure("figS3", PUB / "figS3" / "podgladS3.png", "S3", "RBP–lncRNA pairs from the literature, before and after the control correction.",
 "Left: each point is one pair in one experiment; horizontal position = change of the lncRNA compared with its own control (standard analysis), vertical = after the control correction; filled = significant in the standard analysis; the grey band marks changes close to zero after correction. Right: position of each pair among all removals in that cell line (0 = strongest decrease, 100 = strongest increase); the line joins the two cell lines; grey zones = the most extreme 5%.",
 "For most pairs the correction changes little (points near the diagonal), but some effects disappear, such as RBM15 → XIST. Most pairs from the literature are not exceptional among about 400 removals; exceptions are MTREX → NEAT1 in both lines and PUM1 → NORAD and STAU1 → TUG1 in HepG2.")}</div>
<div class="col">

<h2 id="limits">Limitations</h2>
<p>Two cancer cell lines. Small effects for most RBPs. Sequencing measures relative, not absolute, amounts, so a class effect can come from changes in the reference mRNAs as well as in lncRNAs; we checked absolute changes for splicing and NMD. About one in three of the 35 RBPs may be false positives. Splicing and stability measurements come from few control samples. Removing an RBP changes many genes, so direct and indirect effects cannot be separated. Several exosome removals in K562 and several removals of NMD proteins were ineffective (Figure 4b). GC content of the RNA affects individual experiments; it does not explain the class effect. The rise of XIST after CIZ1 knockout contradicts published data and may be specific to K562.</p>

<h2 id="stats">Statistical tests used</h2>
{table(TESTS, cls_="w")}

<h2 id="rbps">The 35 RBPs with a clear class effect in both cell lines</h2>
<p>Class effects in log2. Groups are from the ENCODE annotation, completed from general knowledge for eight RBPs added later (to be verified against the literature). The last two columns list the lncRNAs that change most specifically after this RBP's removal, in the same direction in both lines.</p>
{table(t35, {"Class effect K562": "{:.2f}", "Class effect HepG2": "{:.2f}"}, "w")}

<h2 id="refs">Selected literature</h2>
{REFS}

<details><summary>Glossary</summary><dl>
<dt>RBP</dt><dd>RNA-binding protein.</dd>
<dt>lncRNA, mRNA</dt><dd>Long non-coding RNA (does not code for protein, longer than 200 nucleotides); messenger RNA (codes for protein).</dd>
<dt>Removal experiment</dt><dd>shRNA knockdown (the RBP's mRNA is destroyed) or CRISPR knockout (the RBP's gene is cut).</dd>
<dt>Class effect</dt><dd>Median over all measured lncRNAs of: change of the lncRNA minus mean change of its 10 most similar mRNAs.</dd>
<dt>Agreement between cell lines (rho)</dt><dd>Spearman rank correlation of the class effects of the same RBPs in K562 and HepG2.</dd>
<dt>log2 change</dt><dd>+1 = doubling, −1 = halving, ±0.1 ≈ ±7%.</dd>
<dt>q, adjusted p value</dt><dd>p value corrected for the number of tests (Benjamini-Hochberg); below 0.05 = significant.</dd>
<dt>Percentile among knockdowns</dt><dd>Position of one removal's effect on a gene among all removals in that cell line (0 = strongest decrease, 100 = strongest increase).</dd>
<dt>Divergent, NAT (natural antisense), intergenic lncRNA</dt><dd>Defined by position relative to protein-coding genes (see Result 5).</dd>
<dt>Spliceosome, NMD, EJC, exosome</dt><dd>Splicing machinery; nonsense-mediated decay (cytoplasmic RNA quality control); exon junction complex (marks spliced RNA, works with NMD); nuclear RNA-degrading complex (with its helper MTREX).</dd>
<dt>eCLIP</dt><dd>Map of where a given RBP binds RNA.</dd>
<dt>GC content</dt><dd>Share of G and C nucleotides in an RNA; affects how efficiently it is captured during sequencing.</dd>
<dt>5′TOP motif</dt><dd>A run of pyrimidines at the very start of an RNA, typical of mRNAs for ribosomal proteins and translation factors.</dd>
<dt>Premature termination</dt><dd>RNA polymerase stopping before it reaches the end of the gene, leaving a shorter RNA.</dd>
</dl></details>
</div>'''

CSS = '''
:root{--ink:#1d2733;--muted:#5b6675;--paper:#fff;--panel:#f4f6f8;--rule:#dde3ea;--lnc:#2a9d8f;--mrna:#7f7f7f}
*{box-sizing:border-box}html{-webkit-text-size-adjust:100%}
body{margin:0;background:var(--paper);color:var(--ink);font-family:"Source Serif 4",Georgia,serif;font-size:18px;line-height:1.62}
main{padding:2.5rem 1.25rem 4rem}.col{max-width:42rem;margin:0 auto}.wide{max-width:72rem;margin:2.25rem auto}
h1,h2,h3,.ft,.lead,th,.note,summary,dt{font-family:"IBM Plex Sans","Segoe UI",Helvetica,Arial,sans-serif}
h1{font-size:2.1rem;line-height:1.18;font-weight:600;margin:0 0 .6rem}.sub{color:var(--muted);margin:0 0 2rem}
.answer{font-size:1.25rem;line-height:1.5;border-left:4px solid var(--lnc);padding:.2rem 0 .2rem 1.1rem;margin:0 0 2rem}
h2{font-size:1.35rem;font-weight:600;margin:3.2rem 0 .8rem;padding-top:.8rem;border-top:1px solid var(--rule)}
h3{font-size:1.05rem;font-weight:600;margin:1.8rem 0 .4rem}p{margin:0 0 1rem}
ul.find,ol{padding-left:1.2rem;margin:0 0 1.4rem}ul.find li,ol li{margin:0 0 .6rem}strong{font-weight:600}
figure{margin:0}figure img{display:block;width:100%;height:auto;border:1px solid var(--rule);border-radius:6px;background:#fff}
figcaption{max-width:52rem;margin:.9rem auto 0;font-size:.97rem;color:#2c3744}.ft{font-weight:600;margin-bottom:.4rem}.lead{font-weight:600}
.tbl{overflow-x:auto;margin:1.2rem 0 1.8rem;border:1px solid var(--rule);border-radius:6px}.tbl.narrow{max-width:22rem}
.tbl.w{width:min(64rem,calc(100vw - 2.5rem));position:relative;left:50%;transform:translateX(-50%)}
table{border-collapse:collapse;width:100%;font-size:.84rem;font-family:"IBM Plex Sans","Segoe UI",Helvetica,Arial,sans-serif}
th{text-align:left;font-weight:600;background:var(--panel);padding:.5rem .65rem;border-bottom:1px solid var(--rule);vertical-align:bottom}
td{padding:.42rem .65rem;border-bottom:1px solid #eef1f4;vertical-align:top}td.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
.note{font-size:.88rem;color:var(--muted)}details{border:1px solid var(--rule);border-radius:6px;padding:.7rem 1rem;margin:1rem 0}
summary{cursor:pointer;font-weight:600}dt{font-weight:600;font-size:.95rem;margin-top:.7rem}dd{margin:.1rem 0 0}
.changes{background:var(--panel);border-radius:6px;padding:.8rem 1.1rem;margin:0 0 2.2rem;font-size:.95rem}.changes ul{margin:.3rem 0 0;padding-left:1.1rem}.changes li{margin:0 0 .4rem}
ol.refs{font-size:.88rem;padding-left:1.4rem}ol.refs li{margin:0 0 .35rem}
@media (max-width:640px){body{font-size:17px}h1{font-size:1.7rem}}
'''
doc = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>lncRNA responses to RBP removal: results report v3</title>
<link href="https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;600&family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body><main>{body}</main></body></html>'''
OUT.mkdir(parents=True, exist_ok=True)
out = OUT / "rbp_lncRNA_report_v3.html"
out.write_text(doc, encoding="utf-8")
missing = [k for k, v in {**FIGS, **{p.name: p for p in LOCUS.glob("locus_*.png")}}.items() if not Path(v).exists()]
print(f"Zapisano: {out} ({out.stat().st_size / 1e6:.1f} MB); brakujace ryciny: {missing or 'brak'}")
