#!/usr/bin/env python3
"""Round 5 E5: external semantic criterion via rule-based German weather-slot extraction.

Motivation (reviewer Major #5): decoded-text metrics (BLEU/chrF/WER) all consume the same
BT decode and cannot tell whether donor replay conveys target-equivalent meaning. We extract
factual slots from references and decoded hypotheses with a frozen rule-based extractor and
compute multiset slot-F1 per item for five slot families:
  NUM      - German number words (temperature/quantity content), incl. minus/plus sign carrier
  TEMP     - temperature mentions: number word followed by 'grad' (value includes sign)
  TIME     - days, months, day-parts, relative days (heute/morgen/uebermorgen/wochenende)
  PLACE    - curated PHOENIX location lexicon (regions, rivers, coasts, countries, compass+land)
  EVENT    - weather events/qualities (regen, schnee, sonne, gewitter, sturm, nebel, frost, ...)
Outputs per-item slot-F1 per family and an all-slot micro-F1, for the four canonical
systems under the original evaluator, plus GT under the six reconstructions for context.
"""
from __future__ import annotations
import json, re
from pathlib import Path
from collections import Counter

ROOT = Path("/ssd/xkb4/RCP")
CELLS = ROOT / "revision_20260728_canonical_rebuild/outputs/evaluations/cells"
OUT = ROOT / "revision_20260729_round5/results/e5_slot_f1.json"

NUM_WORDS = set("""null eins zwei drei vier fünf sechs sieben acht neun zehn elf zwölf dreizehn vierzehn
fünfzehn sechzehn siebzehn achtzehn neunzehn zwanzig einundzwanzig zweiundzwanzig dreiundzwanzig
vierundzwanzig fünfundzwanzig sechsundzwanzig siebenundzwanzig achtundzwanzig neunundzwanzig dreißig
vierzig fünfzig dreißiger vierziger zwanziger""".split())
TIME_WORDS = set("""montag dienstag mittwoch donnerstag freitag samstag sonntag januar februar märz april
mai juni juli august september oktober november dezember morgen übermorgen heute gestern abend nacht
vormittag nachmittag mittag mitternacht wochenende wochenanfang wochenmitte tag nacht tagen abend
morgens abends nachts vormittags nachmittags mittags werktag feiertag""".split())
PLACE_WORDS = set("""deutschland norddeutschland süddeutschland ostdeutschland westdeutschland mitteldeutschland
nordsee ostsee alpen alpenrand alpenvorland bayern sachsen thüringen brandenburg hessen baden württemberg
niedersachsen schleswig holstein mecklenburg vorpommern rheinland pfalz saarland nordrhein westfalen
sachsen anhalt berlin hamburg münchen köln frankfurt stuttgart dresden leipzig hannover nürnberg bremen
rhein main donau elbe oder weser neckar mosel ruhr eifel harz schwarzwald erzgebirge fichtelgebirge
spessart rhön vogtland lausitz oberrhein niederrhein mittelrhein breisgau bodensee Allgäu allgäu
franken schwaben pfalz bergland mittelgebirge küste küsten nordseeküste ostseeküste insel inseln rügen
usedom fehmarn sylt helgoland borkum norderney wangerooge usedom frankreich spanien italien österreich
schweiz polen tschechien dänemark niederlande belgien luxemburg england skandinavien mitteleuropa europa
norden süden westen osten mitte nordosten nordwesten südosten südwesten nordhälfte südhälfte osthälfte
westhälfte nordwesthälfte südosthälfte nordosthälfte südwesthälfte osten süden westen norden
emsland münsterland sauerland siegerland westerwald taunus odenwald alb schwäbische fränkische
mecklenburgische seenplatte lüneburger heide nordfriesland ostfriesland friesland wattenmeer
peene schlei kieler bucht lübecker bucht fehmarnbelt moseltal rheintal donautal elbtal
oberland voralpen voralpenland gäu heide wald bergen gebirge höhen zügen räumen gebieten regionen""".split())
EVENT_WORDS = set("""regen schnee sonne wolken gewitter sturm wind nebel frost hitze schauer regenschauer
schneeschauer gewitterschauer blitz donner bewölkt wolkig heiter trocken nass kalt warm kühl heiß mild
freundlich sonnig trüb diesig schwül drückend stürmisch windig böig böen sturmböen orkan orkanböen
niederschlag niederschläge regenfälle schneefälle dauerregen regenwolken wolkenfelder auflockerung
sonnenschein wolkenloses klar klart klaren aufklaren heiterkeit glätte eis eisig gefroren tau tauwetter
schneematsch matsch nasskalt kühler wärmer milder frostig heiterem freundlichem freundliches wechselhaft
wechselhaftes sonnige sonnigen regnerisch regnerische trüben trübes wolkigen wolkige sonnigem
schönen schönes schönem schlechtem schlechtes schlechten unbeständig unbeständiges gleichmäßig""".split())
SIGN_WORDS = {"minus": "-", "plus": "+"}


def tokens(s: str) -> list[str]:
    return [t for t in re.split(r"\s+", s.strip().lower()) if t and t != "."]


def extract(text: str) -> dict[str, Counter]:
    t = tokens(text)
    slots = {"NUM": Counter(), "TEMP": Counter(), "TIME": Counter(), "PLACE": Counter(), "EVENT": Counter()}
    for i, w in enumerate(t):
        if w in NUM_WORDS:
            sign = t[i - 1] if i > 0 and t[i - 1] in SIGN_WORDS else ""
            slots["NUM"][sign + w] += 1
            if i + 1 < len(t) and t[i + 1] == "grad":
                slots["TEMP"][sign + w] += 1
        if w in TIME_WORDS:
            slots["TIME"][w] += 1
        if w in PLACE_WORDS:
            slots["PLACE"][w] += 1
        if w in EVENT_WORDS:
            slots["EVENT"][w] += 1
    return slots


def f1(a: Counter, b: Counter) -> tuple[float, float, float]:
    inter = sum((a & b).values())
    p = inter / sum(a.values()) if sum(a.values()) else 0.0
    r = inter / sum(b.values()) if sum(b.values()) else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return f, p, r


def score_cell(cell_path: Path) -> dict:
    d = json.load(open(cell_path))
    items = d["metrics"]["items"]
    fam_p = Counter(); fam_r = Counter(); fam_i = Counter()
    per_item = []
    for it in items:
        rs = extract(it["reference"])
        hs = extract(it["hypothesis"])
        row = {}
        for fam in rs:
            i = sum((rs[fam] & hs[fam]).values())
            p = sum(hs[fam].values()); r = sum(rs[fam].values())
            fam_i[fam] += i; fam_p[fam] += p; fam_r[fam] += r
            row[fam] = f1(hs[fam], rs[fam])[0]
        # all-slot micro F1 per item
        ra = sum(rs.values(), Counter()); ha = sum(hs.values(), Counter())
        row["ALL"] = f1(ha, ra)[0]
        per_item.append({"id": it["id"], **row})
    fam_f1 = {}
    for fam in ["NUM", "TEMP", "TIME", "PLACE", "EVENT"]:
        p = fam_i[fam] / fam_p[fam] if fam_p[fam] else 0.0
        r = fam_i[fam] / fam_r[fam] if fam_r[fam] else 0.0
        fam_f1[fam] = {"micro_f1": 2 * p * r / (p + r) if p + r else 0.0,
                       "precision": p, "recall": r,
                       "n_ref_slots": fam_r[fam], "n_hyp_slots": fam_p[fam]}
    all_i = sum(fam_i.values()); all_p = sum(fam_p.values()); all_r = sum(fam_r.values())
    p = all_i / all_p if all_p else 0.0; r = all_i / all_r if all_r else 0.0
    fam_f1["ALL"] = {"micro_f1": 2 * p * r / (p + r) if p + r else 0.0, "precision": p, "recall": r,
                     "n_ref_slots": all_r, "n_hyp_slots": all_p}
    mean_item_all = sum(x["ALL"] for x in per_item) / len(per_item)
    return {"family_micro": fam_f1, "mean_item_all_slot_f1": mean_item_all, "per_item": per_item}


def main():
    out = {}
    for cp, ev in [("cp0", "original"), ("cp1", "seed_101"), ("cp6", "seed_606")]:
        out[ev] = {}
        for sysname in ["GT-v1", "PT-v1", "TN-PTCOMP-v1", "TN-PURE-v1"]:
            res = score_cell(CELLS / f"{cp}_{sysname}.json")
            out[ev][sysname] = {k: v for k, v in res.items() if k != "per_item"}
            if ev == "original":
                Path(ROOT / "revision_20260729_round5/results").joinpath(
                    f"e5_per_item_{sysname}.json").write_text(json.dumps(res["per_item"]))
        print(f"== {ev}")
        for sysname, res in out[ev].items():
            fams = " ".join(f"{f}={res['family_micro'][f]['micro_f1']:.3f}" for f in ["NUM", "TEMP", "TIME", "PLACE", "EVENT", "ALL"])
            print(f"  {sysname:14s} {fams}")
    OUT.write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
