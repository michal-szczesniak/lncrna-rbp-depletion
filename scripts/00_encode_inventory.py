#!/usr/bin/env python3
"""
Faza 0: inwentaryzacja danych ENCODE.

Co robi:
  1. Pobiera listy eksperymentow (API /search/) dla testow z config/inventory.yaml.
  2. Dla testow "file_level_assays" pobiera pelne metadata.tsv (poziom plikow).
  3. Buduje tabele knockdownow z przypisanymi kontrolami, strandowoscia i laboratorium.
  4. Sprawdza kompletnosc plikow (kwantyfikacje genowe, bigWig, BAM) dla knockdownow i kontroli.
  5. Sprawdza obecnosc czynnikow nadzoru jadrowego i nakladanie z TimeLapse-seq.
  6. Zapisuje tabele, surowe odpowiedzi API i raport summary.md.

Uruchomienie (na wezle z dostepem do internetu, srodowisko rbp-py):
  python scripts/00_encode_inventory.py --config config/inventory.yaml --out data/meta
"""
import argparse
import datetime as dt
import io
import json
import re
import sys
import time
from pathlib import Path

import pandas as pd
import requests
import yaml
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

BASE = "https://www.encodeproject.org"

EXP_FIELDS = [
    "accession", "assay_title", "status", "target.label", "biosample_ontology.term_name",
    "lab.title", "possible_controls", "possible_controls.accession", "control_type", "perturbed",
    "replicates.library.strand_specificity",
    "replicates.library.biosample.subcellular_fraction_term_name",
    "replicates.library.nucleic_acid_term_name",
    "date_released", "description",
]

# Kanoniczne nazwy kolumn metadata.tsv -> mozliwe nazwy po normalizacji
FILE_COLS = {
    "file_accession": ["file_accession"],
    "file_format": ["file_format"],
    "file_type": ["file_type"],
    "output_type": ["output_type"],
    "assembly": ["file_assembly", "assembly"],
    "experiment": ["experiment_accession"],
    "assay": ["assay"],
    "cell": ["biosample_term_name"],
    "target": ["experiment_target"],
    "bio_rep": ["biological_replicate_s", "biological_replicates"],
    "tech_rep": ["technical_replicate_s", "technical_replicates"],
    "size": ["size"],
    "lab": ["lab"],
    "md5": ["md5sum"],
    "url": ["file_download_url"],
    "annotation": ["genome_annotation"],
    "controlled_by": ["controlled_by"],
    "file_status": ["file_status"],
    "s3_uri": ["s3_uri"],
    "run_type": ["run_type"],
    "read_length": ["read_length"],
}

KD_ASSAYS = ["shRNA RNA-seq", "CRISPR RNA-seq", "CRISPRi RNA-seq"]
ACC_RE = re.compile(r"ENC[A-Z]{2}\d{3}[A-Z]{3}")


# ----------------------------------------------------------------------------- API

def make_session():
    s = requests.Session()
    retry = Retry(total=5, backoff_factor=1.5, status_forcelist=(429, 500, 502, 503, 504),
                  allowed_methods=("GET",))
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.headers.update({"User-Agent": "rbp-classes-inventory/0.1"})
    return s


def api_get(session, path, params, accept, sleep):
    """GET z obsluga 404 (ENCODE zwraca 404, gdy wyszukiwanie nie ma wynikow)."""
    r = session.get(BASE + path, params=params, headers={"Accept": accept}, timeout=900)
    time.sleep(sleep)  # ENCODE prosi o <= 10 zapytan/s
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r


def search_experiments(session, params_extra, sleep):
    params = [("type", "Experiment"), ("status", "released"), ("format", "json"), ("limit", "all")]
    params += [("field", f) for f in EXP_FIELDS]
    params += params_extra
    r = api_get(session, "/search/", params, "application/json", sleep)
    return [] if r is None else r.json().get("@graph", [])


def fetch_metadata_tsv(session, params_extra, sleep):
    params = [("type", "Experiment"), ("status", "released")] + params_extra
    r = api_get(session, "/metadata/", params, "*/*", sleep)
    if r is None or not r.text.strip():
        return pd.DataFrame()
    return pd.read_csv(io.StringIO(r.text), sep="\t", dtype=str, low_memory=False)


def chunks(seq, n):
    seq = list(seq)
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# ----------------------------------------------------------------------------- parsing

def get_path(obj, path):
    """Plaska lista wartosci pod sciezka 'a.b.c' w zagniezdzonych slownikach i listach."""
    cur = [obj]
    for part in path.split("."):
        nxt = []
        for c in cur:
            if isinstance(c, dict) and part in c and c[part] is not None:
                v = c[part]
                nxt.extend(v if isinstance(v, list) else [v])
        cur = nxt
    return cur


def uniq_join(vals):
    out = []
    for v in vals:
        s = str(v)
        if s and s not in out:
            out.append(s)
    return ";".join(out)


def control_accessions(e):
    accs = [a for a in get_path(e, "possible_controls.accession") if isinstance(a, str)]
    if not accs:  # zapasowo: lista sciezek '/experiments/ENCSR.../'
        for v in get_path(e, "possible_controls"):
            if isinstance(v, str):
                accs += ACC_RE.findall(v)
    return uniq_join(accs)


def flatten_experiment(e):
    return {
        "experiment": e.get("accession", ""),
        "assay": e.get("assay_title", ""),
        "cell": uniq_join(get_path(e, "biosample_ontology.term_name")),
        "target": uniq_join(get_path(e, "target.label")),
        "lab": uniq_join(get_path(e, "lab.title")),
        "control_type": e.get("control_type", "") or "",
        "perturbed": e.get("perturbed", ""),
        "possible_controls": control_accessions(e),
        "strand_specificity": uniq_join(get_path(e, "replicates.library.strand_specificity")),
        "subcellular_fraction": uniq_join(
            get_path(e, "replicates.library.biosample.subcellular_fraction_term_name")),
        "nucleic_acid": uniq_join(get_path(e, "replicates.library.nucleic_acid_term_name")),
        "date_released": e.get("date_released", ""),
        "description": (e.get("description", "") or "").replace("\t", " ").replace("\n", " "),
    }


def normalize_file_table(df):
    if df.empty:
        return df
    norm = {c: re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_") for c in df.columns}
    df = df.rename(columns=norm)
    out = pd.DataFrame(index=df.index)
    missing = []
    for canon, cands in FILE_COLS.items():
        hit = next((c for c in cands if c in df.columns), None)
        if hit is None:
            out[canon] = ""
            missing.append(canon)
        else:
            out[canon] = df[hit].fillna("")
    if missing:
        print(f"  [uwaga] brak kolumn w metadata.tsv: {', '.join(missing)}", file=sys.stderr)
    out["size"] = pd.to_numeric(out["size"], errors="coerce").fillna(0).astype("int64")
    return out


def canon_target(label, synonyms):
    t = str(label).strip().upper()
    return synonyms.get(t, t)


def is_stranded(value):
    """True / False / None (nieznane)."""
    vals = [v.strip().lower() for v in str(value).split(";") if v.strip()]
    if not vals:
        return None
    if all(v == "unstranded" for v in vals):
        return False
    if any(v == "unstranded" for v in vals):
        return None  # mieszane repliki
    return True


# ----------------------------------------------------------------------------- per-experiment file summary

def summarize_files(files, assembly):
    """Kompletnosc plikow na eksperyment (tylko released, wybrany genom)."""
    f = files[(files["file_status"] == "released") & (files["assembly"] == assembly)].copy()
    rows = []
    for exp, g in f.groupby("experiment"):
        gq = g[g["output_type"] == "gene quantifications"]
        tq = g[g["output_type"] == "transcript quantifications"]
        bam = g[(g["file_format"] == "bam") & (g["output_type"] == "alignments")]
        bw = g[g["file_format"] == "bigWig"]
        bw_types = set(bw["output_type"])
        rows.append({
            "experiment": exp,
            "gene_quant_reps": gq["bio_rep"].nunique(),
            "gene_quant_files": len(gq),
            "transcript_quant_files": len(tq),
            "annotation": uniq_join(sorted(set(gq["annotation"]) - {""})),
            "bam_files": len(bam),
            "bam_gb": round(bam["size"].sum() / 1e9, 2),
            "bigwig_unique_stranded": ("plus strand signal of unique reads" in bw_types
                                       and "minus strand signal of unique reads" in bw_types),
            "bigwig_output_types": uniq_join(sorted(bw_types)),
            "bigwig_gb": round(bw["size"].sum() / 1e9, 2),
        })
    return pd.DataFrame(rows)


# ----------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config/inventory.yaml")
    ap.add_argument("--out", default="data/meta")
    ap.add_argument("--sleep", type=float, default=0.3, help="przerwa miedzy zapytaniami [s]")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    cells, assembly = cfg["cells"], cfg["assembly"]
    synonyms = {k.upper(): v.upper() for k, v in cfg.get("synonyms", {}).items()}
    out = Path(args.out)
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    s = make_session()
    cell_params = [("biosample_ontology.term_name", c) for c in cells]
    started = dt.datetime.now().isoformat(timespec="seconds")

    # 1. Eksperymenty -------------------------------------------------------
    exp_rows = []
    plan = [(a, cell_params) for a in cfg["file_level_assays"] + cfg["experiment_level_assays"]]
    plan += [(a, []) for a in cfg.get("cell_free_assays", [])]
    plan += [("TF ChIP-seq", cell_params + [("target.label", t) for t in cfg.get("tf_chip_targets", [])])]
    for assay, extra in plan:
        print(f"[search] {assay}")
        graph = search_experiments(s, [("assay_title", assay)] + extra, args.sleep)
        slug = re.sub(r"[^A-Za-z0-9]+", "_", assay).strip("_")
        json.dump(graph, open(raw / f"search_{slug}.json", "w"))
        exp_rows += [flatten_experiment(e) for e in graph]
        print(f"         {len(graph)} eksperymentow")
    exps = pd.DataFrame(exp_rows).drop_duplicates("experiment")

    # 2. Pliki dla testow rdzeniowych --------------------------------------
    file_tabs = []
    for assay in cfg["file_level_assays"]:
        print(f"[metadata] {assay}")
        df = fetch_metadata_tsv(s, [("assay_title", assay)] + cell_params, args.sleep)
        slug = re.sub(r"[^A-Za-z0-9]+", "_", assay).strip("_")
        if not df.empty:
            df.to_csv(raw / f"metadata_{slug}.tsv", sep="\t", index=False)
        file_tabs.append(normalize_file_table(df))
        print(f"           {len(df)} plikow")

    # 3. Knockdowny i kontrole ---------------------------------------------
    kd_all = exps[exps["assay"].isin(KD_ASSAYS)].copy()
    ctrl_ids = set(a for v in kd_all["possible_controls"] for a in v.split(";") if a)
    is_ctrl = (kd_all["experiment"].isin(ctrl_ids) | (kd_all["control_type"] != "")
               | kd_all["target"].str.contains("non-specific", case=False, na=False))
    kd = kd_all[~is_ctrl].copy()
    ctrl = kd_all[is_ctrl].copy()
    if kd.empty:
        exps.to_csv(out / "experiments_all.tsv", sep="\t", index=False)
        sys.exit("[blad] nie znaleziono eksperymentow knockdownu - sprawdz nazwy testow w konfiguracji "
                 "i surowe odpowiedzi w data/meta/raw/")

    # kontrole spoza wynikow wyszukiwania - dociagamy po akcesjach
    missing_ctrl = sorted(ctrl_ids - set(exps["experiment"]))
    if missing_ctrl:
        print(f"[controls] dociagam {len(missing_ctrl)} kontroli spoza wynikow wyszukiwania")
        for ch in chunks(missing_ctrl, 40):
            acc = [("accession", a) for a in ch]
            ctrl = pd.concat([ctrl, pd.DataFrame([flatten_experiment(e)
                                                  for e in search_experiments(s, acc, args.sleep)])])
            file_tabs.append(normalize_file_table(fetch_metadata_tsv(s, acc, args.sleep)))

    files = pd.concat([t for t in file_tabs if not t.empty], ignore_index=True) \
        if any(not t.empty for t in file_tabs) else pd.DataFrame(columns=list(FILE_COLS))
    files = files.drop_duplicates("file_accession")
    files.to_csv(out / "files_file_level.tsv", sep="\t", index=False)

    fsum = summarize_files(files, assembly) if not files.empty else pd.DataFrame(
        columns=["experiment", "gene_quant_reps"])
    fsum.to_csv(out / "files_per_experiment.tsv", sep="\t", index=False)
    reps = dict(zip(fsum["experiment"], fsum["gene_quant_reps"]))

    kd["target_canon"] = kd["target"].map(lambda t: canon_target(t, synonyms))
    kd["is_kd_lab"] = kd["lab"].str.lower().str.contains(cfg["kd_lab_keyword"].lower(), na=False)
    kd["stranded"] = kd["strand_specificity"].map(is_stranded)
    kd["kd_gene_quant_reps"] = kd["experiment"].map(reps).fillna(0).astype(int)

    def ctrl_min_reps(v):
        accs = [a for a in v.split(";") if a]
        return min((int(reps.get(a, 0)) for a in accs), default=0)

    kd["n_controls"] = kd["possible_controls"].map(lambda v: len([a for a in v.split(";") if a]))
    kd["ctrl_gene_quant_reps_min"] = kd["possible_controls"].map(ctrl_min_reps)
    kd["complete"] = (kd["kd_gene_quant_reps"] >= 2) & (kd["n_controls"] >= 1) & (kd["ctrl_gene_quant_reps_min"] >= 2)
    kd["usable"] = kd["is_kd_lab"] & (kd["stranded"] != False) & kd["complete"]  # noqa: E712
    kd = kd.merge(fsum.drop(columns=["gene_quant_reps"], errors="ignore"), on="experiment", how="left")
    kd.sort_values(["cell", "target_canon"]).to_csv(out / "kd_experiments.tsv", sep="\t", index=False)
    ctrl.merge(fsum, on="experiment", how="left").to_csv(out / "kd_controls.tsv", sep="\t", index=False)

    # 4. Czynniki nadzoru i TimeLapse --------------------------------------
    usable = kd[kd["usable"]]
    pres = []
    groups = dict(cfg["surveillance_factors"])
    groups["timelapse_overlap"] = cfg.get("timelapse_kd", [])
    for grp, genes in groups.items():
        for gene in genes:
            g = gene.upper()
            row = {"group": grp, "gene": g}
            for c in cells:
                hit = usable[(usable["target_canon"] == g) & (usable["cell"] == c)]
                row[c] = uniq_join(hit["experiment"]) or "-"
            pres.append(row)
    pres = pd.DataFrame(pres)
    pres.to_csv(out / "surveillance_presence.tsv", sep="\t", index=False)

    # 5. Pozostale testy ----------------------------------------------------
    counts = (exps.assign(cell=exps["cell"].replace("", "in vitro"))
              .groupby(["assay", "cell"]).size().rename("n_experiments").reset_index())
    counts.to_csv(out / "assay_counts.tsv", sep="\t", index=False)
    frac = exps[exps["assay"].isin(["polyA plus RNA-seq", "total RNA-seq"]) & (exps["subcellular_fraction"] != "")]
    frac.to_csv(out / "fractionation.tsv", sep="\t", index=False)
    exps.to_csv(out / "experiments_all.tsv", sep="\t", index=False)

    # 6. Lista plikow do pobrania w fazie 0 ---------------------------------
    keep_exp = set(usable["experiment"]) | set(a for v in usable["possible_controls"] for a in v.split(";") if a)
    rel = files[(files["file_status"] == "released") & (files["assembly"] == assembly)]
    core = rel[rel["experiment"].isin(keep_exp) & (rel["output_type"] == "gene quantifications")].copy()
    core["set"] = "kd_gene_quant"
    bru = rel[rel["assay"].isin(["Bru-seq", "BruChase-seq", "BruUV-seq"]) & (rel["file_format"] != "bam")].copy()
    bru["set"] = "bru_nonbam"
    dl = pd.concat([core, bru], ignore_index=True)
    dl.to_csv(out / "files_to_download_phase0.tsv", sep="\t", index=False)

    # 7. Raport --------------------------------------------------------------
    kd_bam_gb = kd.loc[kd["usable"], "bam_gb"].fillna(0).sum()
    kd_bw_gb = kd.loc[kd["usable"], "bigwig_gb"].fillna(0).sum()
    lines = [
        "# Inwentaryzacja ENCODE (faza 0)", "",
        f"Uruchomiono: {started}. Genom: {assembly}. Linie: {', '.join(cells)}.", "",
        "## Knockdowny RBP", "",
        f"- Eksperymenty knockdownu (bez kontroli): {len(kd)}; kontrole: {len(ctrl)}",
        f"- W tym z laboratorium '{cfg['kd_lab_keyword']}': {int(kd['is_kd_lab'].sum())}",
        f"- Strandowosc: tak {int((kd['stranded'] == True).sum())}, nie {int((kd['stranded'] == False).sum())}, "  # noqa: E712
        f"nieznana {int(kd['stranded'].isna().sum())}",
        f"- Komplet kwantyfikacji (>=2 repliki KD i kontroli): {int(kd['complete'].sum())}",
        f"- **Uzyteczne (lab + strandowane + komplet): {len(usable)}**",
    ]
    for c in cells:
        lines.append(f"  - {c}: {int((usable['cell'] == c).sum())}")
    lines += [
        f"- Wersje anotacji w kwantyfikacjach: {uniq_join(sorted(set(';'.join(kd['annotation'].fillna('')).split(';')) - {''}))}",
        f"- Uzyteczne z bigWig unikalnych odczytow na obu niciach: {int(usable['bigwig_unique_stranded'].fillna(False).sum())}",
        f"- Objetosc dla uzytecznych: BAM {kd_bam_gb:.0f} GB, bigWig {kd_bw_gb:.0f} GB (bez kontroli)", "",
        "## Czynniki nadzoru i TimeLapse-seq (uzyteczne knockdowny)", "",
        "| Grupa | Gen | " + " | ".join(cells) + " |",
        "| --- | --- | " + " | ".join("---" for _ in cells) + " |",
    ]
    for _, r in pres.iterrows():
        if any(r[c] != "-" for c in cells) or r["group"] == "timelapse_overlap":
            lines.append(f"| {r['group']} | {r['gene']} | " + " | ".join(r[c] for c in cells) + " |")
    lines += ["", "(Pokazano tylko geny obecne w zestawie oraz wszystkie geny TimeLapse.)", "",
              "## Pozostale testy", "", "| Test | Linia | Eksperymenty |", "| --- | --- | --- |"]
    for _, r in counts.iterrows():
        lines.append(f"| {r['assay']} | {r['cell']} | {r['n_experiments']} |")
    lines += ["", f"Frakcjonowanie (RNA-seq z frakcja subkomorkowa): {len(frac)} eksperymentow.", "",
              "## Pliki do pobrania w fazie 0", "",
              f"- Kwantyfikacje genowe KD + kontroli: {int((dl['set'] == 'kd_gene_quant').sum())} plikow, "
              f"{dl.loc[dl['set'] == 'kd_gene_quant', 'size'].sum() / 1e9:.1f} GB",
              f"- Bru/BruChase/BruUV bez BAM: {int((dl['set'] == 'bru_nonbam').sum())} plikow, "
              f"{dl.loc[dl['set'] == 'bru_nonbam', 'size'].sum() / 1e9:.1f} GB",
              "", "Typy plikow Bru (do decyzji, co pobrac):", ""]
    for (a, ot), g in bru.groupby(["assay", "output_type"]):
        lines.append(f"- {a} / {ot}: {len(g)} plikow, {g['size'].sum() / 1e9:.1f} GB")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    json.dump({"started": started, "base": BASE, "assembly": assembly, "cells": cells,
               "n_kd": len(kd), "n_usable": len(usable), "n_controls": len(ctrl)},
              open(out / "run_info.json", "w"), indent=2)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
