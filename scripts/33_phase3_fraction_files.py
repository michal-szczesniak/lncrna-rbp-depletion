"""
Faza 3, krok 3.3A: pliki eksperymentow frakcjonowania (lokalizacja subkomorkowa RNA) z ENCODE.

Z inwentaryzacji (data/meta/fractionation.tsv) bierzemy eksperymenty RNA-seq z frakcja subkomorkowa w K562 i HepG2,
pobieramy dla nich metadane plikow z API ENCODE i wybieramy kwantyfikacje genowe (GRCh38, najlepiej anotacja V29).
Wyjscie: data/meta/fraction_files.tsv (lista do pobrania) i podsumowanie: linia x frakcja x typ RNA.
"""
import io
import re
import time

import pandas as pd
import requests

BASE = "https://www.encodeproject.org"
COLS = {"file_accession": "file_accession", "file_format": "file_format", "output_type": "output_type",
        "file_assembly": "assembly", "experiment_accession": "experiment", "assay": "assay",
        "biological_replicate_s": "bio_rep", "size": "size", "md5sum": "md5", "file_download_url": "url",
        "genome_annotation": "annotation", "file_status": "file_status", "s3_uri": "s3_uri"}


def metadata_for(accs, session):
    params = [("type", "Experiment"), ("status", "released")] + [("accession", a) for a in accs]
    r = session.get(BASE + "/metadata/", params=params, headers={"Accept": "*/*"}, timeout=600)
    r.raise_for_status()
    time.sleep(0.3)
    return pd.read_csv(io.StringIO(r.text), sep="\t", dtype=str)


def main(fetch=metadata_for):
    fr = pd.read_csv("data/meta/fractionation.tsv", sep="\t", dtype=str).fillna("")
    fr = fr[fr["cell"].isin(["K562", "HepG2"])]
    print(f"Eksperymenty frakcjonowania z inwentaryzacji: {len(fr)}")
    s = requests.Session()
    tabs = [fetch(list(fr["experiment"].iloc[i:i + 40]), s) for i in range(0, len(fr), 40)]
    m = pd.concat(tabs, ignore_index=True)
    m.columns = [re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_") for c in m.columns]
    m = m.rename(columns=COLS)[[c for c in COLS.values() if c in m.rename(columns=COLS).columns]].fillna("")
    m = m[(m["output_type"] == "gene quantifications") & (m["file_status"] == "released") & (m["assembly"] == "GRCh38")]
    m["pref"] = (m["annotation"] == "V29").astype(int)
    best = m.groupby("experiment")["pref"].transform("max")
    m = m[m["pref"] == best].drop(columns="pref")             # V29, jesli eksperyment ja ma; inaczej to, co jest
    m = m.merge(fr[["experiment", "cell", "subcellular_fraction", "nucleic_acid", "assay"]].rename(columns={"assay": "assay_fr"}),
                on="experiment", how="left")
    m["assay"] = m["assay"].where(m["assay"] != "", m["assay_fr"])
    m["set"] = "fractionation"
    m.drop(columns="assay_fr").to_csv("data/meta/fraction_files.tsv", sep="\t", index=False)
    pd.set_option("display.width", 200)
    print("\nKwantyfikacje genowe wedlug linii, frakcji i typu RNA (eksperymenty / pliki / anotacja):")
    t = m.groupby(["cell", "subcellular_fraction", "nucleic_acid"]).agg(
        eksperymenty=("experiment", "nunique"), pliki=("file_accession", "size"),
        anotacja=("annotation", lambda x: ",".join(sorted(set(x)))))
    print(t.to_string())
    missing = sorted(set(fr["experiment"]) - set(m["experiment"]))
    print(f"\nEksperymenty bez kwantyfikacji genowych GRCh38: {missing if missing else 'brak'}")
    print(f"Plikow do pobrania: {len(m)}, {pd.to_numeric(m['size'], errors='coerce').sum() / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
