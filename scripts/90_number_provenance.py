"""
90_number_provenance.py - rejestr liczb manuskryptu (audyt).
Dla kazdej liczby z tekstu (lista wbudowana ponizej, wyciagnieta z manuskryptu 19.10 wieczorem, po poprawkach z audytu; rev 351) szuka, gdzie wystepuje w projekcie:
  logi (logs/, results/**/*.log|txt), tabele wynikow (results/**/*.tsv|tsv.gz|csv) i stale w skryptach (scripts/).
Zgodnosc = wartosc z pliku zaokraglona do tylu miejsc, ile ma liczba w tekscie. Trafienia sa sortowane wg zgodnosci
kontekstu (slowa kluczowe ze zdania). Wersje oprogramowania sa sprawdzane bezposrednio (Python i R).
Nic nie zmienia w projekcie. Wynik: results/audit/number_provenance.tsv (albo ~/rbp_audit/, gdy brak miejsca).
Ponowne uruchomienie po zmianach w tekscie lub danych = weryfikacja (rozbieznosci = liczby bez trafien).
"""
import gzip, io, math, os, re, subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

NUMBERS = """N001	860	0	number	result	
N002	473	0	number	result	experiment K562 HepG2 RNA
N003	0.31	2	number	result	rho spearman
N004	342	0	number	result	rho spearman
N005	0.26	2	number	result	gc rho spearman
N006	0.37	2	number	result	gc mrna rho spearman
N007	7		word	result	gc
N008	2		word	design	K562
N009	860	0	number	result	
N010	3		word	design	
N011	860	0	number	result	
N012	473	0	number	result	crispr shrna experiment K562 HepG2 RNA
N013	8		word	result	control shared
N014	10		word	result	control shared
N015	0.6666666666666666		word	result	control shared
N016	10		word	parameter	lncrna mrna splic
N017	1252	0	number	result	lncrna K562 HepG2
N018	1008	0	number	result	lncrna K562 HepG2
N019	0.25	2	number	result	rho
N020	4		word	result	lncrna K562
N021	0.3333333333333333		word	result	lncrna K562 HepG2
N022	0.056	3	number	result	class median
N023	4	0	percent	result	class median
N024	342	0	number	result	
N025	0.31	2	number	result	class rho spearman
N026	95	0	percent	result	class rho spearman
N027	0.20	2	number	result	rho spearman
N028	0.41	2	number	result	rho spearman
N029	10000	0	number	result	pair
N030	0.26	2	number	result	control rho shared
N031	0.26	2	number	result	control rho shared batch
N032	0.27	2	number	result	control median shared batch virtual
N033	0.29	2	number	result	rho
N034	0.017	3	number	result	class lncrna median rho
N035	0.04	2	number	result	median rho
N036	20	0	number	result	median rho
N037	3		word	result	class lncrna median mrna
N038	308	0	number	result	specific method
N039	342	0	number	result	specific method
N040	34	0	number	result	rho specific method
N041	0.18	2	number	result	rho method
N042	95	0	percent	result	rho method
N043	-0.18	2	number	result	rho method
N044	0.50	2	number	result	rho method
N045	0.25	2	number	result	mrna rho
N046	0.38	2	number	result	mrna rho
N047	0.37	2	number	result	gc rho intron
N048	0.1	1	number	parameter	class gc robust
N049	95	0	percent	result	chance shuffl
N050	1	0	number	result	chance shuffl
N051	9	0	number	result	chance shuffl
N052	17		word	result	class
N053	4		word	result	gc chance shuffl
N054	15		word	result	lncrna mrna splic
N055	9		word	result	lncrna mrna splic MTREX STA
N056	2		word	result	MTREX STAU1 lncrna mrna splic
N057	35	0	number	result	gc
N058	12	0	number	result	
N059	35	0	number	result	
N060	0.3333333333333333		word	result	
N061	18	0	number	result	lncrna
N062	10		word	result	lncrna
N063	7		word	result	efficien lncrna mrna splic
N064	0.36	2	number	result	rho
N065	0.31	2	number	result	rho
N066	0.14285714285714285		word	result	rho stabil
N067	66	0	number	result	eclip depletion
N068	0.02	2	number	result	bound lncrna median mrna depletion
N069	0.05	2	number	result	bound lncrna median mrna depletion
N070	9	0	number	result	bound lncrna mrna
N071	12	0	percent	result	bound lncrna mrna
N072	14	0	percent	result	lncrna
N073	47	0	number	result	lncrna
N074	342	0	number	result	lncrna
N075	11	0	percent	result	lncrna mrna
N076	36	0	number	result	lncrna mrna
N077	342	0	number	result	lncrna mrna
N078	26	0	percent	result	lncrna mrna
N079	21	0	percent	result	mrna
N080	107	0	number	result	mrna
N081	2000	0	number	result	mrna chance depletion
N082	0.3	1	percent	result	mrna
N083	63	0	number	result	class splic
N084	4		word	result	lncrna mrna K562 HepG2
N085	5		word	result	lncrna mrna K562 HepG2
N086	8		word	result	nmd
N087	12		word	result	nmd
N088	3		word	result	UPF1
N089	3		word	result	K562 UPF1 UPF2
N090	5		word	result	lncrna mrna depletion K562
N091	95	0	number	result	SMG7 SMG8 percentile
N092	93	0	number	result	SMG7 SMG8 nmd percentile
N093	89	0	number	result	gc nmd percentile percent UPF1
N094	88	0	number	result	gc nmd percentile percent UPF1
N095	95	0	number	result	percentile percent UPF1
N096	7		word	result	class nmd
N097	8		word	result	class nmd
N098	98	0	number	result	percentile depletion percent MAGOH EIF4A3 ZFAS1 SNHG5
N099	100	0	number	result	percentile depletion percent MAGOH EIF4A3 ZFAS1 SNHG5
N100	10	0	number	parameter	intergenic lncrna
N101	0.38	2	number	result	class divergent intergenic lncrna median mrna MTREX
N102	0.24	2	number	result	class divergent intergenic lncrna median mrna
N103	0.17	2	number	result	class intergenic lncrna median mrna
N104	0.12	2	number	result	class intergenic lncrna
N105	19	0	number	result	depletion EXOSC9 MTREX
N106	0.36	2	number	result	MTREX divergent intergenic lncrna mrna
N107	0.23	2	number	result	MTREX divergent intergenic lncrna mrna
N108	2		word	result	class divergent intergenic lncrna top NELFE MTREX
N109	0.58	2	number	result	divergent lncrna depletion K562 HepG2 MTREX
N110	0.36	2	number	result	divergent lncrna mrna depletion K562 HepG2 MTREX
N111	0.46	2	number	result	divergent lncrna percentile depletion percent K562 HepG2 NELFE
N112	99.8	1	number	result	divergent lncrna percentile depletion percent K562 HepG2 NELFE
N113	2		word	literature	NELF
N114	0.11	2	number	result	divergent intergenic lncrna depletion
N115	0.13	2	number	result	divergent intergenic lncrna depletion
N116	76	0	number	result	lncrna mrna regulator shared
N117	2		word	result	lncrna mrna regulator shared
N118	24	0	number	result	control depletion kb MTREX
N119	90	0	number	result	control depletion kb MTREX
N120	0.48	2	number	result	pair rho
N121	0.54	2	number	result	pair rho
N122	2		word	result	divergent lncrna
N123	2	0	number	result	regulator chance
N124	11	0	number	result	regulator chance
N125	23	0	percent	result	lncrna mrna regulator chance
N126	14	0	percent	result	lncrna mrna regulator chance
N127	3		word	result	lncrna mrna regulator shared
N128	2		word	result	class lncrna chance
N129	22	0	number	result	lncrna regulator
N130	5	0	number	result	depletion
N131	7		word	result	SRSF1 lncrna splic
N132	0	0	number	result	percentile depletion percent NORAD
N133	1	0	number	result	percentile depletion percent NORAD
N134	22	0	number	result	NORAD lncrna regulator
N135	0.5		word	result	EIF3D RCC2 SRSF1
N136	20	0	number	result	pair
N137	18	0	number	result	pair
N138	5	0	percent	result	lncrna pair
N139	6		word	result	lncrna pair
N140	3		word	result	lncrna pair snorna host UPF1
N141	10	0	number	result	crispr shrna
N142	50	0	number	result	crispr shrna
N143	30	0	percent	result	crispr mrna
N144	0.2		word	result	crispr mrna
N145	374	0	number	result	mrna rho spearman experiment
N146	815	0	number	result	mrna rho spearman experiment
N147	0.2	1	number	result	gc mrna rho spearman experiment
N148	0.5		word	result	mrna
N149	18	0	number	result	class gc
N150	35	0	number	result	class gc
N151	2		word	design	
N152	33	0	number	result	control lncrna
N153	17	0	number	result	gc robust splic
N154	9		word	result	gc robust splic tau MTREX STAU1
N155	2		word	result	MTREX STAU1 gc lncrna robust splic
N156	0.5		word	result	EIF3D NORAD RCC2 SRSF1
N157	4		word	result	
N158	2		word	design	
N159	17	0	number	result	gc robust
N160	35	0	number	result	
N161	4		word	result	gc robust chance
N162	0.3333333333333333		word	result	gc robust
N163	860	0	number	result	K562
N164	441	0	number	result	K562 replic
N165	419	0	number	result	K562 replic
N166	473	0	number	parameter	experiment K562 HepG2 RNA
N167	478	0	number	result	K562 replic
N168	2		word	design	K562 replic
N169	16		word	result	
N170	10	0	number	parameter	control
N171	10	0	number	parameter	
N172	50	0	number	parameter	
N173	10	0	number	result	K562 control median shared
N174	8	0	number	result	K562 control median shared
N175	1	0	number	result	K562 control median shared
N176	27	0	number	result	K562 control median shared
N177	20	0	number	result	K562 control shared
N178	25	0	number	result	K562 control shared
N179	815	0	number	result	K562 class shared
N180	421	0	number	result	K562 class
N181	394	0	number	result	K562 class
N182	4		word	parameter	K562 control shared
N183	321	0	number	parameter	control shared batch
N184	6		word	parameter	
N185	2	0	number	parameter	divergent
N186	10	0	number	parameter	GENCODE intergenic
N187	10	0	number	parameter	intergenic
N188	4		word	result	divergent lncrna
N189	6	0	number	parameter	efficien splic
N190	0.1	1	number	parameter	fraction specific top
N191	52	0	number	result	median
N192	1	0	number	parameter	median
N193	0.5	1	number	parameter	
N194	500	0	number	parameter	
N195	10		word	parameter	lncrna mrna
N196	4291	0	number	parameter	lncrna K562 HepG2
N197	4326	0	number	parameter	lncrna nuclear tau K562 HepG2
N198	1	0	number	parameter	gc lncrna mrna percentile
N199	99	0	number	parameter	gc lncrna mrna percentile
N200	1252	0	number	result	K562 lncrna mrna percentile
N201	1008	0	number	result	K562 lncrna mrna
N202	20	0	number	parameter	lncrna mrna
N203	5		word	parameter	lncrna mrna pseudo
N204	10000	0	number	parameter	bootstrap pair permut spearman
N205	95	0	percent	parameter	bootstrap pair
N206	0.1	1	number	parameter	class
N207	0.05	2	number	parameter	class
N208	1000	0	number	parameter	robust chance shuffl
N209	50	0	number	parameter	crispr robust shrna
N210	30	0	percent	parameter	crispr mrna shrna
N211	5	0	number	parameter	crispr mrna shrna
N212	20	0	number	parameter	ISPR mrna shrna
N213	0.05	2	number	parameter	class
N214	0.2	1	number	parameter	gc spearman
N215	20	0	number	parameter	class gc median
N216	17	0	number	result	gc robust
N217	35	0	number	result	
N218	10	0	number	parameter	control
N219	5		word	parameter	control efficien splic
N220	12	0	number	result	GAS5 SNHG1 SNHG5 SNHG6 SNHG8 nmd
N221	0.05	2	number	parameter	median
N222	0.1	1	number	parameter	median
N223	3		word	parameter	ATF4 D45A GADD45B SMG5 SMG7 UPF3B median
N224	5		word	parameter	lncrna mrna nmd
N225	3	0	number	parameter	
N226	3	0	number	parameter	
N227	5	0	percent	parameter	bound robust
N228	0.05	2	number	parameter	
N229	5		word	parameter	robust
N230	1.4826	4	number	result	median robust
N231	50	0	number	parameter	median
N232	3		word	parameter	regulator
N233	10		word	parameter	regulator
N234	20	0	number	parameter	
N235	90	0	percent	parameter	
N236	0.05	2	number	parameter	lncrna regulator
N237	22	0	number	result	lncrna regulator
N238	5	0	number	parameter	lncrna regulator
N239	1000	0	number	parameter	lncrna mrna permut
N240	10000	0	number	parameter	permut
N241	33	0	number	result	
N242	4		word	parameter	permut
N243	8		word	result	
N244	20	0	number	result	lncrna pair
N245	18	0	number	result	lncrna pair
N246	4		word	result	pair
N247	3.11	2	number	version	
N248	3.0	1	number	version	
N249	2.4	1	number	version	
N250	1.17	2	number	version	
N251	1.9	1	number	version	
N252	3.11	2	number	version	
N253	4.5	1	number	version	
N254	1.50	2	number	version	
N255	1.32	2	number	version	
N256	1.18	2	number	version	
N257	6		word	parameter	control shared
N258	2		word	design	HNRNPK NONO class lncrna
N259	10		word	parameter	class lncrna median mrna
N260	0.1	1	number	parameter	class median
N261	342	0	number	result	class
N262	17	0	number	result	gc robust
N263	18	0	number	result	gc robust
N264	95	0	percent	parameter	bootstrap permut rho spearman
N265	10000	0	number	result	bootstrap permut rho spearman
N266	95	0	percent	result	bootstrap rho spearman
N267	20	0	number	result	lncrna median mrna pseudo
N268	95	0	percent	result	NAT class divergent intergenic lncrna
N269	5	0	percent	parameter	bound robust
N270	5	0	percent	parameter	bound robust top
N271	5		word	parameter	bound top
N272	0.05	2	number	parameter	
N273	5	0	percent	parameter	bound top
N274	30	0	number	result	K562 lncrna mrna splic
N275	33	0	number	result	K562 lncrna
N276	5		word	parameter	bound control
N277	5		word	parameter	efficien splic
N278	5		word	parameter	K562 nmd
N279	15	0	number	result	class
N280	10		word	parameter	regulator shared
N281	3		word	parameter	lncrna regulator
N282	10		word	parameter	lncrna regulator
N283	22	0	number	result	lncrna regulator
N284	50	0	number	parameter	crispr lncrna
N285	3		word	result	NORAD regulator
N286	0.05	2	number	parameter	crispr shrna
N287	12	0	number	result	nmd
N288	5	0	percent	parameter	lncrna"""

MAX_BYTES = 30e6          # pomijamy wieksze pliki tabel
MAX_TABLES_PER_DIR = 40   # katalogi z wieloma tabelami (wyniki per eksperyment) pomijamy
TOP = 3
WORDS_APPROX = {0.5: (0.45, 0.62), 1/3: (0.28, 0.40), 2/3: (0.60, 0.72), 0.2: (0.17, 0.23), 1/7: (0.12, 0.17), 2.0: (1.7, 2.3)}
COUNT_COL = re.compile(r"(^n$|^n_|_n$|count|liczb|^num|ile_|geny|genes|n_)", re.I)
tok_re = re.compile(r"(?<![\w.])[-−]?\d+(?:\.\d+)?(?:[eE][-−]?\d+)?(%?)")

def parse(spec):
    sid, val, dec, kind, cls, kw = (spec.split("\t") + [""] * 6)[:6]
    d = {"id": sid, "raw": val, "dec": int(dec) if dec.strip().isdigit() else 0, "kind": kind, "cls": cls,
         "kw": [k.lower() for k in kw.split()], "approx": None, "sci": False, "ok": True}
    try:
        v = float(val)
    except ValueError:
        d["ok"] = False; d["v"] = None; return d
    d["v"] = v
    if kind == "word" and (v < 1 or v == 2.0 and val.startswith("2.0")):
        d["approx"] = WORDS_APPROX.get(min(WORDS_APPROX, key=lambda k: abs(k - v)))
    if "e" in val.lower():
        d["sci"] = True
    return d

SPECS = [parse(l) for l in NUMBERS.strip().split("\n")]

def match_vals(arr, s, pct_token=False):
    """arr: numpy array of floats -> boolean mask"""
    v, dec = s["v"], s["dec"]
    with np.errstate(all="ignore"):
        if s["approx"]:
            lo, hi = s["approx"]; return (arr >= lo) & (arr <= hi)
        if s["sci"]:
            return (arr > 0) & (np.abs(np.log10(np.where(arr > 0, arr, 1)) - math.log10(v)) < 0.08)
        if s["kind"] == "percent":
            m1 = (np.abs(arr) <= 1) & np.isclose(np.round(arr * 100, dec), v, atol=1e-9)
            m2 = (np.abs(arr) > 1) & np.isclose(np.round(arr, dec), v, atol=1e-9)
            return m1 | m2
        return np.isclose(np.round(arr, dec), v, atol=1e-9) | np.isclose(np.round(np.abs(arr), dec), v, atol=1e-9) & (dec > 0)

def kwscore(text, s):
    t = text.lower()
    return sum(1 for k in s["kw"] if k and k in t)

if not (Path("scripts").is_dir() and Path("results").is_dir()):
    sys.exit("Uruchom z katalogu projektu rbp_classes (musza byc scripts/ i results/).")
hits = {s["id"]: [] for s in SPECS}
root = Path(".")

# ---------------------------------------------------------------- logi i pliki tekstowe
logfiles = list(Path("logs").rglob("*")) + [p for p in Path("results").rglob("*") if p.suffix in (".log", ".txt")]
logfiles = [p for p in logfiles if p.is_file() and p.stat().st_size < 20e6]
print(f"logi: {len(logfiles)} plikow", flush=True)
for p in logfiles:
    try:
        lines = p.read_text(errors="replace").splitlines()
    except Exception:
        continue
    for i, line in enumerate(lines):
        toks = [(m.group(0), m.group(1)) for m in tok_re.finditer(line)]
        if not toks:
            continue
        arr = np.array([float(t.rstrip("%").replace("−", "-")) for t, _ in toks])
        ctx = " ".join(lines[max(0, i - 2): i + 1]) + " " + str(p)
        for s in SPECS:
            if not s["ok"] or s["cls"] == "version":
                continue
            if s["v"] is not None and s["dec"] == 0 and not s["approx"] and not s["sci"] and s["v"] < 13 and kwscore(ctx, s) == 0:
                continue  # male liczby calkowite tylko z kontekstem
            m = match_vals(arr, s)
            if m.any():
                hits[s["id"]].append((kwscore(ctx, s) + 1, f"LOG {p}:{i+1}: {line.strip()[:160]}"))

# ---------------------------------------------------------------- tabele
tabs = [p for p in Path("results").rglob("*") if p.is_file() and (p.name.endswith((".tsv", ".tsv.gz", ".csv")))]
per_dir = {}
for p in tabs:
    per_dir[p.parent] = per_dir.get(p.parent, 0) + 1
tabs = [p for p in tabs if p.stat().st_size < MAX_BYTES and per_dir[p.parent] <= MAX_TABLES_PER_DIR]
print(f"tabele: {len(tabs)} plikow (pominiete katalogi z >{MAX_TABLES_PER_DIR} tabelami: "
      f"{', '.join(str(d) for d, n in per_dir.items() if n > MAX_TABLES_PER_DIR) or 'brak'})", flush=True)
for k, p in enumerate(tabs, 1):
    try:
        df = pd.read_csv(p, sep="," if p.name.endswith(".csv") else "\t", low_memory=False)
    except Exception:
        continue
    if df.empty:
        continue
    labcol = next((c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])), None)
    num = df.select_dtypes("number")
    for c in num.columns:
        arr = num[c].to_numpy(dtype=float)
        fin = np.isfinite(arr)
        if not fin.any():
            continue
        for s in SPECS:
            if not s["ok"] or s["cls"] in ("version",) or s["approx"]:
                continue
            if s["dec"] == 0 and not s["sci"] and s["v"] < 100 and not COUNT_COL.search(str(c)):
                continue
            if s["dec"] == 1 and s["kind"] != "percent":
                continue  # 0.1, 0.5 itp. sa zbyt pospolite w tabelach
            m = match_vals(arr, s) & fin
            if m.any():
                idx = np.flatnonzero(m)
                r = idx[0]
                lab = str(df.iloc[r][labcol])[:40] if labcol is not None else str(df.index[r])
                ctx = f"{p} {c} {lab}"
                hits[s["id"]].append((kwscore(ctx, s) + (2 if len(idx) <= 3 else 0),
                                      f"TAB {p} | kol. {c} | wiersz {lab} | wartosc {arr[r]:.6g} | trafien w kolumnie: {len(idx)}"))
    if k % 50 == 0:
        print(f"   ... {k}/{len(tabs)} tabel", flush=True)

# ---------------------------------------------------------------- stale w skryptach (parametry)
scripts = [p for p in Path("scripts").rglob("*") if p.suffix in (".py", ".R", ".sh", ".yaml", ".yml")] + list(Path("config").rglob("*.y*ml"))
src = {}
for p in scripts:
    try:
        src[p] = p.read_text(errors="replace").splitlines()
    except Exception:
        pass
def literal_forms(s):
    v = s["v"]
    forms = set()
    if v is None:
        return forms
    if float(v).is_integer():
        iv = int(v)
        forms |= {str(iv), f"{iv:_}"}
        if iv >= 1000 and iv % 1000 == 0:
            forms |= {f"{iv//1000}e3", f"{iv//1000}_000"}
        if s["kind"] != "percent" and iv < 1000:
            forms |= {f"{iv * 1000}", f"{iv * 1000:_}", f"{iv}e3", f"{iv}_000", f"{iv} * 1000", f"{iv}*1000"}
    else:
        forms |= {s["raw"], s["raw"].lstrip("0")}
    if s["kind"] == "percent":
        forms |= {f"{v/100:g}", f"{v/100:g}".lstrip("0")}
    return forms
for s in SPECS:
    if s["cls"] != "parameter" or not s["ok"]:
        continue
    pats = [re.compile(r"(?<![\w.])" + re.escape(f) + r"(?![\w.])") for f in literal_forms(s)]
    for p, lines in src.items():
        for i, line in enumerate(lines):
            if "=" not in line and "(" not in line:
                continue
            if any(pt.search(line) for pt in pats):
                hits[s["id"]].append((kwscore(line + " " + str(p), s) + (1 if re.search(r"^[A-Z_]{3,}\s*=", line.strip()) else 0),
                                      f"SKRYPT {p}:{i+1}: {line.strip()[:140]}"))

# ---------------------------------------------------------------- wersje oprogramowania
vers = {}
for mod in ["pandas", "numpy", "scipy", "sklearn", "pyBigWig", "matplotlib"]:
    try:
        vers[mod] = __import__(mod).__version__
    except Exception as e:  # noqa: BLE001
        vers[mod] = f"brak ({e.__class__.__name__})"
vers["python"] = sys.version.split()[0]
try:
    r = subprocess.run(["Rscript", "-e", 'cat(R.version$major, ".", R.version$minor, " ", as.character(packageVersion("DESeq2")), " ", as.character(packageVersion("apeglm")), " ", as.character(packageVersion("data.table")), sep="")'],
                       capture_output=True, text=True, timeout=120)
    vers["R / DESeq2 / apeglm / data.table"] = r.stdout.strip() or r.stderr.strip()[:100]
except Exception as e:  # noqa: BLE001
    vers["R"] = f"brak ({e.__class__.__name__}); uruchom z envs/rbp-r/bin w PATH"
vline = "; ".join(f"{k} {v}" for k, v in vers.items())

# ---------------------------------------------------------------- zapis
rows = []
for s in SPECS:
    h = sorted(hits[s["id"]], key=lambda x: -x[0])
    if s["cls"] == "version":
        best = [vline]
    else:
        best = [b for _, b in h[:TOP]]
    rows.append({"id": s["id"], "value": s["raw"], "class": s["cls"], "n_hits": len(h),
                 "best1": best[0] if len(best) > 0 else "", "best2": best[1] if len(best) > 1 else "",
                 "best3": best[2] if len(best) > 2 else ""})
out = pd.DataFrame(rows)
for d in [Path("results/audit"), Path.home() / "rbp_audit"]:
    try:
        d.mkdir(parents=True, exist_ok=True)
        out.to_csv(d / "number_provenance.tsv", sep="\t", index=False)
        print(f"Zapisano {d / 'number_provenance.tsv'}")
        break
    except OSError as e:
        print(f"   nie mozna zapisac w {d}: {e}")
print("Wersje:", vline)
none = out[(out["n_hits"] == 0) & (out["class"] != "version")]
print(f"Liczby z trafieniami: {(out['n_hits'] > 0).sum()} z {len(out)}; bez trafien: {len(none)}")
print(none.groupby("class")["id"].apply(lambda x: " ".join(x)).to_string())
