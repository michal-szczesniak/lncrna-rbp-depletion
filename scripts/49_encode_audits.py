"""
Audyty ENCODE dla wszystkich knockdownow: czy ENCODE oflagowal eksperymenty (np. te, w ktorych mRNA RBP nie spadlo)?

Pobiera z API ENCODE pole "audit" kazdego eksperymentu (poziomy ERROR, NOT_COMPLIANT, WARNING, INTERNAL_ACTION),
zestawia kategorie flag i porownuje je ze skutecznoscia wyciszenia (spadek mRNA RBP) oraz z lista 35 RBP.
Wyjscie: results/phase4/audits/encode_audits.tsv (jeden wiersz na eksperyment i flage), audit_summary.tsv
"""
import time
from pathlib import Path

import pandas as pd
import requests

BASE = "https://www.encodeproject.org"
OUT = Path("results/phase4/audits")
OUT.mkdir(parents=True, exist_ok=True)
LEVELS = ["ERROR", "NOT_COMPLIANT", "WARNING", "INTERNAL_ACTION"]
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 70)


def fetch(accs):
    params = [("type", "Experiment"), ("format", "json"), ("limit", "all")] + [("accession", a) for a in accs]
    r = requests.get(BASE + "/search/", params=params, headers={"Accept": "application/json"}, timeout=600)
    r.raise_for_status()
    graph = r.json().get("@graph", [])
    if graph and not any("audit" in e for e in graph):          # zapasowo: pelny rekord kazdego eksperymentu
        graph = []
        for a in accs:
            rr = requests.get(f"{BASE}/experiments/{a}/", params={"format": "json"}, headers={"Accept": "application/json"}, timeout=300)
            if rr.ok:
                graph.append(rr.json())
            time.sleep(0.2)
    return graph


def main(fetch_fn=fetch):
    summ = pd.read_csv("results/phase2/deseq2/summary_all.tsv", sep="\t")
    hits = set(pd.read_csv("results/phase2/classes/rbp_hits.tsv", sep="\t", index_col=0).index)
    accs = summ["kd_experiment"].tolist()
    rows, seen = [], set()
    for i in range(0, len(accs), 50):
        for e in fetch_fn(accs[i:i + 50]):
            acc = e.get("accession")
            seen.add(acc)
            for lvl in LEVELS:
                for a in (e.get("audit") or {}).get(lvl, []):
                    rows.append({"kd_experiment": acc, "poziom": lvl, "kategoria": a.get("category", ""),
                                 "szczegol": str(a.get("detail", ""))[:200]})
        time.sleep(0.3)
    au = pd.DataFrame(rows, columns=["kd_experiment", "poziom", "kategoria", "szczegol"])
    au.to_csv(OUT / "encode_audits.tsv", sep="\t", index=False)
    print(f"Eksperymenty sprawdzone w API: {len(seen)} z {len(accs)}; flag lacznie: {len(au)}")

    print("\n1. Kategorie flag (liczba eksperymentow z dana flaga)")
    t = au.drop_duplicates(["kd_experiment", "poziom", "kategoria"]).groupby(["poziom", "kategoria"]).size().rename("eksperymenty")
    print(t.sort_values(ascending=False).to_string())

    s = summ.set_index("kd_experiment")
    s["slabe_wyciszenie"] = s["target_lfc"] > -0.5
    worst = au.assign(r=au["poziom"].map({l: i for i, l in enumerate(LEVELS)})).sort_values("r").drop_duplicates("kd_experiment")
    s["najpowazniejsza_flaga"] = worst.set_index("kd_experiment")["poziom"].reindex(s.index).fillna("brak")
    s["flagi"] = au.groupby("kd_experiment")["kategoria"].agg(lambda x: "; ".join(sorted(set(x)))).reindex(s.index).fillna("")
    print("\n2. Najpowazniejsza flaga a skutecznosc wyciszenia (spadek mRNA RBP > -0.5 = slabe)")
    print(pd.crosstab([s["method"], s["slabe_wyciszenie"]], s["najpowazniejsza_flaga"]).to_string())

    print("\n3. Eksperymenty ze slabym wyciszeniem: flagi ENCODE (pierwsze 40)")
    w = s[s["slabe_wyciszenie"]].sort_values("target_lfc", ascending=False)
    print(w[["target", "cell", "method", "target_lfc", "n_padj05", "najpowazniejsza_flaga", "flagi"]].head(40).to_string())

    print("\n4. 35 RBP z listy: flagi ENCODE")
    h = s[s["target"].isin(hits)].sort_values(["target", "cell"])
    print(h[["target", "cell", "method", "target_lfc", "najpowazniejsza_flaga", "flagi"]].to_string())
    s[["target", "cell", "method", "target_lfc", "n_padj05", "slabe_wyciszenie", "najpowazniejsza_flaga", "flagi"]].to_csv(
        OUT / "audit_summary.tsv", sep="\t")
    print(f"\nZapisano: {OUT}/encode_audits.tsv, {OUT}/audit_summary.tsv")


if __name__ == "__main__":
    main()
