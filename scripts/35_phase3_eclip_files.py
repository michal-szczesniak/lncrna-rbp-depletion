"""
Faza 3, krok 3.4A: pliki pikow eCLIP z ENCODE dla K562 i HepG2.

Pobiera metadane eksperymentow eCLIP (pliki bed narrowPeak, GRCh38), wypisuje dostepne typy plikow i wybiera
dla kazdego eksperymentu piki z obu replik biologicznych (pliki laczace repliki "1, 2"; jesli jest kilka typow,
preferowane te z "IDR" lub "replicated" w nazwie). Sprawdza, ile RBP ma zarowno eCLIP, jak i knockdown w tej samej linii.
Wyjscie: data/meta/eclip_files.tsv
"""
import io
import re
import time

import pandas as pd
import requests

BASE = "https://www.encodeproject.org"
COLS = {"file_accession": "file_accession", "file_format": "file_format", "file_type": "file_type", "output_type": "output_type",
        "file_assembly": "assembly", "experiment_accession": "experiment", "biosample_term_name": "cell",
        "experiment_target": "target", "biological_replicate_s": "bio_rep", "size": "size", "md5sum": "md5",
        "file_download_url": "url", "file_status": "file_status", "s3_uri": "s3_uri"}


def fetch_metadata():
    """Dwa kroki: (1) numery eksperymentow eCLIP z wyszukiwarki, (2) metadane plikow dla tych numerow."""
    sess = requests.Session()
    params = [("type", "Experiment"), ("assay_title", "eCLIP"), ("status", "released"),
              ("biosample_ontology.term_name", "K562"), ("biosample_ontology.term_name", "HepG2"),
              ("limit", "all"), ("format", "json"), ("field", "accession")]
    r = sess.get(BASE + "/search/", params=params, headers={"Accept": "application/json"}, timeout=600)
    r.raise_for_status()
    accs = [e["accession"] for e in r.json().get("@graph", [])]
    print(f"Wyszukiwarka ENCODE: {len(accs)} eksperymentow eCLIP w K562 i HepG2", flush=True)
    tabs = []
    for i in range(0, len(accs), 40):
        p = [("type", "Experiment"), ("status", "released")] + [("accession", a) for a in accs[i:i + 40]]
        rr = sess.get(BASE + "/metadata/", params=p, headers={"Accept": "*/*"}, timeout=600)
        rr.raise_for_status()
        tabs.append(pd.read_csv(io.StringIO(rr.text), sep="\t", dtype=str))
        time.sleep(0.3)
    return pd.concat(tabs, ignore_index=True) if tabs else pd.DataFrame()


def main(fetch=fetch_metadata):
    m = fetch()
    print(f"Surowa odpowiedz: {len(m)} wierszy (plikow)")
    if m.empty:
        raise SystemExit("Brak danych z API - prosze przyslac ten wynik")
    m.columns = [re.sub(r"[^a-z0-9]+", "_", c.lower()).strip("_") for c in m.columns]
    m = m.rename(columns=COLS)[[c for c in COLS.values() if c in m.rename(columns=COLS).columns]].fillna("")
    print("   typy plikow:", m["file_type"].value_counts().head(8).to_dict())
    print("   linie komorkowe:", m["cell"].value_counts().to_dict())
    m = m[(m["file_type"] == "bed") & (m["assembly"] == "GRCh38") & (m["file_status"] == "released")
          & m["cell"].isin(["K562", "HepG2"])].copy()
    m["target"] = m["target"].str.replace(r"-human$", "", regex=True)
    m["reps"] = m["bio_rep"].str.replace(" ", "")
    pd.set_option("display.width", 200)
    print(f"Eksperymenty eCLIP (K562, HepG2): {m['experiment'].nunique()}; pliki bed narrowPeak: {len(m)}")
    print("\nTypy plikow (typ wyjscia x repliki):")
    print(m.groupby(["output_type", "reps"]).size().to_string())

    both = m[m["reps"].str.contains(",")].copy()
    both["pref"] = both["output_type"].str.contains("IDR|replicated", case=False).astype(int)
    both = both.sort_values(["experiment", "pref"], ascending=[True, False]).drop_duplicates("experiment")
    no_both = sorted(set(m["experiment"]) - set(both["experiment"]))
    both["assay"] = "eCLIP"
    both["set"] = "eclip_peaks"
    both.drop(columns=["pref"]).to_csv("data/meta/eclip_files.tsv", sep="\t", index=False)
    print(f"\nWybrane pliki (piki z obu replik): {len(both)}; typ: {both['output_type'].value_counts().to_dict()}")
    print(f"Eksperymenty bez pliku z obu replik: {len(no_both)}")

    kd = pd.read_csv("data/meta/phase2_kds.tsv", sep="\t", dtype=str)
    print("\nRBP z eCLIP i knockdownem w tej samej linii:")
    for cell in ["K562", "HepG2"]:
        e = set(both.loc[both["cell"] == cell, "target"])
        k = set(kd.loc[kd["cell"] == cell, "target_canon"])
        print(f"   {cell}: eCLIP {len(e)}, knockdown {len(k)}, oba {len(e & k)}")
    print(f"\nPlikow do pobrania: {len(both)}, {pd.to_numeric(both['size'], errors='coerce').sum() / 1e6:.0f} MB")


if __name__ == "__main__":
    main()
