"""
Tworzy scripts/70_report_v3.py z scripts/56_report_html.py (raport v2 pozostaje bez zmian).
Zmiany v3: ramka zmian, poprawione twierdzenia (NMD, gospodarze snoRNA, korekta kontroli), test GC, skutecznosc wyciszen NMD,
profil KCNQ1OT1, kontekst z literatury z lista prac, ryciny locus w nowym stylu, ryciny S1 i S3, testy, ograniczenia, slowniczek.
"""
from pathlib import Path

SRC, DST = Path("scripts/56_report_html.py"), Path("scripts/70_report_v3.py")
s = SRC.read_text(encoding="utf-8")

NEW_DATA = r'''# ================================================================ v3: nowe tabele
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

'''

GC_SECTION = r'''<h3>Does the GC content of the RNA explain it?</h3>
<p>The proportion of G and C nucleotides in an RNA affects how efficiently it is captured during sequencing, and ENCODE noted that changes in these experiments partly depend on it [1]. Our matching did not include GC content, so we checked it separately. In {n_gcdep} of {n_gcall} experiments the changes of mRNAs follow their GC content (|rho| &gt; 0.2), and lncRNAs have slightly less GC than mRNAs (median about 48.5% vs 50%). We repeated the comparison with GC content matched as well, and separately after removing the GC trend from every experiment. The agreement between the two cell lines did not fall; it rose slightly. GC content added noise but did not create the effect:</p>
{table(t_gc, {"Median class effect (log2)": "{:.3f}", "Agreement between lines (rho)": "{:.3f}"}) if t_gc is not None else ""}
'''

NMD_SECTION = r'''
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
'''

KPROF = r'''
<div class="col"><p class="note">KCNQ1OT1 signal (reads per million) in ten equal segments of the gene, from its start (seg. 1) to its end (seg. 10).</p>
{table(t_kprof, None, "w") if t_kprof is not None else ""}</div>'''

S3_FIG = r'''</div>
<div class="wide">{figure("figS3", PUB / "figS3" / "podgladS3.png", "S3", "RBP–lncRNA pairs from the literature, before and after the control correction.",
 "Left: each point is one pair in one experiment; horizontal position = change of the lncRNA compared with its own control (standard analysis), vertical = after the control correction; filled = significant in the standard analysis; the grey band marks changes close to zero after correction. Right: position of each pair among all removals in that cell line (0 = strongest decrease, 100 = strongest increase); the line joins the two cell lines; grey zones = the most extreme 5%.",
 "For most pairs the correction changes little (points near the diagonal), but some effects disappear, such as RBM15 → XIST. Most pairs from the literature are not exceptional among about 400 removals; exceptions are MTREX → NEAT1 in both lines and PUM1 → NORAD and STAU1 → TUG1 in HepG2.")}</div>
<div class="col">
'''

NEW_TESTS = r'''    ["Does GC content of the RNA explain the class effect?", "class effect with GC matched, or after removing the GC trend; agreement between lines (Spearman, bootstrap 95% CI)", "2 variants", "none (sensitivity analysis)", "not explained by GC"],
    ["Which removals of NMD proteins disrupted NMD?", "change of 12 known NMD targets (DESeq2)", "8 removals", "DESeq2 adjusted p per gene", "3 of 8 disrupted NMD"],
    ["Do lncRNAs rise more than mRNAs after removal of NMD proteins?", "Mann-Whitney per expression bin, each experiment separately", "8 × 5 bins", "Benjamini-Hochberg within each experiment", "rise not specific to NMD"],
'''

R = [
 ('Raport v2, czesc B: tekst w prostym jezyku (po angielsku), tabele z plikow wynikow i ryciny w jednym pliku HTML.',
  'Raport v3 (z 56_report_html.py przez 69_make_report_v3.py): tekst w prostym jezyku (po angielsku), tabele i ryciny w jednym pliku HTML.'),
 ('OUT = Path("results/phase4/report_v2")', 'OUT = Path("results/phase4/report_v3")'),
 ('# ================================================================ tekst', NEW_DATA + '# ================================================================ tekst'),
 ('Results report, version 2 (3 October 2026).</p>', 'Results report, version 3 (3 October 2026).</p>'),
 ('A few processes stand out: the splicing machinery, nonsense-mediated decay and the nuclear RNA exosome.</p>',
  'Nor does the GC content of the RNA. Two processes stand out: the splicing machinery and the nuclear RNA exosome, which removes most strongly the lncRNAs that share a promoter region with a protein-coding gene.</p>\n{CHANGES}'),
 ('<p><strong>35 RBPs</strong> have a clear class effect', GC_SECTION + '<p><strong>35 RBPs</strong> have a clear class effect'),
 ('the class effect is the same on unbound genes.',
  'the class effect is the same on unbound genes. This agrees with ENCODE\'s own conclusion that many changes after RBP removal are indirect [1].'),
 ('Losing the spliceosome disturbs mRNA processing more than lncRNA processing; losing NMD raises lncRNAs.',
  'Losing the spliceosome disturbs mRNA processing more than lncRNA processing.'),
 ('Removing NMD proteins raises lncRNAs more than mRNAs of the same expression (K562; HepG2 has only two experiments).',
  'Panel c pools all removals of NMD proteins; Figures 4b and 4c below show why this result needs care.'),
 ('\n\n<h2 id="sno">Result 4', NMD_SECTION + '\n\n<h2 id="sno">Result 4'),
 ('snoRNA host genes resemble housekeeping genes and are sensitive to NMD.', 'snoRNA host genes resemble mRNAs more than other lncRNAs.'),
 ('Compared with other lncRNAs, snoRNA hosts are more cytoplasmic, broadly expressed and mostly have CpG-island promoters, as mRNAs do. Removing exon junction complex proteins (MAGOH, EIF4A3) raises ZFAS1 and SNHG5 more than almost any other removal (98th–100th percentile; locus figures below). snoRNA hosts should be treated as a separate subgroup, not as typical lncRNAs.',
  'Compared with lincRNAs, snoRNA hosts are less nuclear, broadly expressed and mostly have CpG-island promoters; in none of these do they differ significantly from mRNAs. All snoRNA host genes carry a 5′TOP motif, typical of mRNAs for ribosomal proteins [9], which fits this mRNA-like behaviour. As a group, however, they are not more sensitive to removal of NMD/EJC proteins than other lncRNAs (panel d, not significant in either line). NMD is known to target snoRNA host transcripts [7], but in our data only individual hosts respond strongly: removing exon junction complex proteins (MAGOH, EIF4A3) raises ZFAS1 and SNHG5 more than almost any other removal (98th–100th percentile; locus figure below). snoRNA hosts should be treated as a separate subgroup, not as typical lncRNAs.'),
 ('LOCUS / "locus_ZFAS1_EJC.png"', 'PUB / "figS5" / "ZFAS1_EJC.png"'),
 ('"Gene model on top; below, RNA-seq signal on the gene\'s strand (reads per million, mean of two replicates) in control cells (grey) and after removal of the RBP (line), for each experiment. Labels give the change of the gene, its significance, the change after control correction and its percentile among all removal experiments in that cell line."',
  'NEW_READ'),
 ('and removing the core NMD factor UPF1 does not reproduce the rise, so the mechanism is not proven.',
  'and the one UPF1 removal that clearly disrupted NMD (K562) raises ZFAS1 only modestly, so the mechanism is not proven.'),
 ('but the effective MTREX removal shows the same pattern.',
  'but the effective MTREX removal shows the same pattern. A study in HeLa cells found the same order after exosome removal: antisense and promoter-upstream transcripts were stabilised more than intergenic lncRNAs [2].'),
 ('a hint that both production and degradation control these transcripts.',
  'a hint that both production and degradation control these transcripts. Both proteins are known to restrain transcription near promoters and transcription that does not lead to a gene [3–5].'),
 ('LOCUS / "locus_KCNQ1OT1_exosome.png"', 'PUB / "fig4" / "d_KCNQ1OT1.png"'),
 ('"As in Figure 5e. KCNQ1OT1 is a long, unspliced nuclear lncRNA."',
  '"As in Figure 5e. KCNQ1OT1 is a long (about 90 kb), unspliced nuclear lncRNA, transcribed from right to left in this view."'),
 ('The RNA is produced but normally removed almost immediately by the exosome.")}</div>',
  'The RNA is produced but normally removed almost immediately by the exosome. The signal fades gradually towards the end of the gene (table below), as expected if transcription of lncRNAs often stops early, at many points along the gene [2]. In K562 the transcript ends after about 27 kb already in control cells, and MTREX removal raises only this short form: K562 seems to make a shorter KCNQ1OT1. Shorter, tissue-specific forms have been described in mouse heart [13], and the length of this RNA matters for its function [14].")}</div>' + KPROF),
 ('"As in Figure 5e; the shaded region is present only in the long NEAT1 isoform (NEAT1_2).",',
  '"Older layout: tracks as in Figure 5e, without the replicate panel; the shaded region is present only in the long NEAT1 isoform (NEAT1_2).",'),
 ('so these data mainly capture their polyadenylated forms or precursors.")}</div>',
  'so these data mainly capture their polyadenylated forms or precursors. MTREX, together with hnRNPH1, is known to control the stability of the long NEAT1 isoform [6].")}</div>'),
 ('LOCUS / "locus_NORAD.png"', 'PUB / "fig5" / "c_NORAD.png"'),
 ('LOCUS / "locus_XIST_CIZ1.png"', 'PUB / "figS5" / "XIST_CIZ1.png"'),
 ('The literature describes CIZ1\'s role in XIST localisation rather than abundance.',
  'Published work describes CIZ1 as anchoring XIST to the inactive X and reports no change in XIST levels when CIZ1 is lost, in mouse fibroblasts [10, 11] and in human breast cancer cells [12]. Our observation therefore differs from previous studies; it may be specific to K562 or indirect, and needs independent confirmation.'),
 ('"As in Figure 5e; the bottom track shows NONO binding sites (eCLIP, K562).",',
  '"Older layout: tracks as in Figure 5e, without the replicate panel; the bottom track shows NONO binding sites (eCLIP, K562).",'),
 ('LOCUS / "locus_OIP5-AS1_HuR.png"', 'PUB / "figS5" / "OIP5-AS1_HuR.png"'),
 ('(Figure S1; known side effect of Cas9 DNA cuts); shRNA does not.',
  '(Figure S1); shRNA does not. A DNA cut can switch off transcription hundreds of kilobases around it [15], and its repair often deletes several kilobases [16]; either could contribute.'),
 ('<li><strong>Many apparent RBP–lncRNA effects disappear after the control correction</strong>, e.g. RBM15 → XIST (−0.61 log2, p = 6·10⁻²⁴ before; −0.02 after).</li>',
  '<li><strong>The control correction changes the result for only a few RBP–lncRNA pairs from the literature</strong>, but for some it matters: RBM15 → XIST (−0.61 log2, p = 6·10⁻²⁴ before; −0.02 after; Figure S3).</li>\n'
  '<li><strong>Changes after RBP removal often depend on the GC content of the RNA</strong> ({n_gcdep} of {n_gcall} experiments), as ENCODE noted [1]. It does not explain the class effect, but comparisons between groups of genes should take it into account.</li>\n'
  '<li><strong>Removing an NMD protein does not always disrupt NMD.</strong> Single knockouts of SMG5, SMG7, SMG8 or SMG9 hardly changed known NMD targets, and UPF1 removal in HepG2 lowered rather than raised them (Figure 4b). Checking known targets is a simple test worth doing before interpreting such experiments.</li>'),
 ('FIGS["figS1"], "S1"', 'PUB / "figS1" / "a_sasiedzi.png", "S1"'),
 ('"Median change of genes at a given position relative to the removed RBP\'s gene, compared with similar genes elsewhere, after CRISPR (purple) or shRNA (green).",',
  '"Change of protein-coding genes at a given position relative to the removed RBP\'s gene, compared with similar genes elsewhere, after CRISPR (blue) or shRNA (pink): violin = distribution, white dot and bar = median and middle half. Stars: change different from zero. Below each category: % of genes clearly lower (CRISPR | shRNA).",'),
 ('are not reproduced as changes in lncRNA levels.</li>',
  'are not reproduced as changes in lncRNA levels (Figure S3).</li>\n'
  '<li><strong>A specific effect of NMD on lncRNAs</strong>: the rise of lncRNAs after removal of NMD proteins also follows knockouts that leave NMD intact (Figure 4c).</li>\n'
  '<li><strong>snoRNA hosts as a class sensitive to NMD</strong>: not supported (Figure 5d); only individual hosts respond.</li>'),
 ('We do not interpret these networks.</li>',
  'We do not interpret these networks. A published network analysis of the same data estimated a very low signal-to-noise ratio [17].</li>'),
 ('\n<h2 id="limits">Limitations</h2>', S3_FIG + '\n<h2 id="limits">Limitations</h2>'),
 ('Several exosome removals in K562 were ineffective.</p>',
  'Several exosome removals in K562 and several removals of NMD proteins were ineffective (Figure 4b). GC content of the RNA affects individual experiments; it does not explain the class effect. The rise of XIST after CIZ1 knockout contradicts published data and may be specific to K562.</p>'),
 ('["Do snoRNA hosts differ from mRNAs and lincRNAs (Fig. 5)?", "Mann-Whitney or Fisher", "8", "Benjamini-Hochberg", ""],',
  '["Do snoRNA hosts differ from mRNAs and lincRNAs (Fig. 5)?", "Mann-Whitney or Fisher", "8", "Benjamini-Hochberg", "as a group not more sensitive to NMD than other lncRNAs"],\n' + NEW_TESTS),
 ('<details><summary>Glossary</summary><dl>',
  '<h2 id="refs">Selected literature</h2>\n{REFS}\n\n<details><summary>Glossary</summary><dl>'),
 ('<dt>eCLIP</dt><dd>Map of where a given RBP binds RNA.</dd>\n</dl></details>',
  '<dt>eCLIP</dt><dd>Map of where a given RBP binds RNA.</dd>\n'
  '<dt>GC content</dt><dd>Share of G and C nucleotides in an RNA; affects how efficiently it is captured during sequencing.</dd>\n'
  '<dt>5′TOP motif</dt><dd>A run of pyrimidines at the very start of an RNA, typical of mRNAs for ribosomal proteins and translation factors.</dd>\n'
  '<dt>Premature termination</dt><dd>RNA polymerase stopping before it reaches the end of the gene, leaving a shorter RNA.</dd>\n'
  '</dl></details>'),
 ('@media (max-width:640px)',
  '.changes{background:var(--panel);border-radius:6px;padding:.8rem 1.1rem;margin:0 0 2.2rem;font-size:.95rem}.changes ul{margin:.3rem 0 0;padding-left:1.1rem}.changes li{margin:0 0 .4rem}\n'
  'ol.refs{font-size:.88rem;padding-left:1.4rem}ol.refs li{margin:0 0 .35rem}\n@media (max-width:640px)'),
 ('<title>lncRNA responses to RBP removal: results report v2</title>', '<title>lncRNA responses to RBP removal: results report v3</title>'),
 ('out = OUT / "rbp_lncRNA_report_v2.html"', 'out = OUT / "rbp_lncRNA_report_v3.html"'),
]
for a, b in R:
    n = s.count(a)
    assert n == 1, f"nie znaleziono dokladnie raz ({n}): {a[:70]}"
    s = s.replace(a, b)
DST.write_text(s, encoding="utf-8")
print(f"Zapisano {DST} ({len(R)} zmian)")
