#!/usr/bin/env python3
"""
Generate 1,500 QA pairs from GeoJSON planetary feature data.
Outputs train/test JSONL splits for fine-tuning.

  python -m georag.generate_qa_dataset
  python -m georag.generate_qa_dataset --total 2000
"""
from __future__ import annotations

import argparse
import json
import random
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

from georag.config import (
    FEATURE_FILES,
    QA_OUTPUT_FULL,
    QA_OUTPUT_TEST,
    QA_OUTPUT_TRAIN,
    QA_SEED,
    QA_TOTAL_TARGET,
    QA_TRAIN_RATIO,
)

# --- helpers ---

BODY_ADJECTIVE: dict[str, str] = {
    "moon": "lunar",
    "mars": "Martian",
    "mercury": "Mercurian",
    "venus": "Venusian",
}

BODY_DISPLAY: dict[str, str] = {
    "moon": "the Moon",
    "mars": "Mars",
    "mercury": "Mercury",
    "venus": "Venus",
}

CATEGORY_DESCRIPTION: dict[str, str] = {
    "Crater": "an impact crater",
    "Mons": "a mountain (mons)",
    "Tholus": "a small dome-shaped hill (tholus)",
    "Vallis": "a valley (vallis)",
    "Patera": "a shallow, irregular crater (patera)",
    "Rupes": "a cliff or scarp (rupes)",
    "Planitia": "a large low-lying plain (planitia)",
    "Terra": "a large highland region (terra)",
    "Dorsum": "a ridge (dorsum)",
    "Catena": "a chain of craters (catena)",
    "Chaos": "a region of chaotic terrain (chaos)",
    "Fossa": "a long, narrow trough (fossa)",
    "Rima": "a narrow fissure or channel (rima)",
    "Lacus": "a small plain (lacus)",
    "Regio": "a broad region (regio)",
    "Sinus": "a bay-like feature (sinus)",
}


def _fmt_lat(lat: float) -> str:
    d = "N" if lat >= 0 else "S"
    return f"{abs(lat):.2f}°{d}"


def _fmt_lon(lon: float) -> str:
    d = "E" if lon >= 0 else "W"
    return f"{abs(lon):.2f}°{d}"


def _fmt_coords(feat: dict) -> str:
    return f"{_fmt_lat(feat['lat'])}, {_fmt_lon(feat['lon'])}"


def _hemisphere(lat: float) -> str:
    if lat > 30:
        return "northern"
    elif lat < -30:
        return "southern"
    return "equatorial"


def _cat_desc(cat: str) -> str:
    return CATEGORY_DESCRIPTION.get(cat, f"a surface feature ({cat.lower()})")


def _body_display(body: str) -> str:
    return BODY_DISPLAY.get(body, body.title())


def _body_adj(body: str) -> str:
    return BODY_ADJECTIVE.get(body, body.title())


# --- question templates ---
#
# each template is (question_fn, answer_fn), grouped by family
# so sampling stays balanced.

TEMPLATES: list[tuple[str, callable, callable]] = []


def _register(family: str):
    def decorator(pair_fn):
        q_fn, a_fn = pair_fn()
        TEMPLATES.append((family, q_fn, a_fn))
        return pair_fn
    return decorator


# feature type identification

@_register("type_identification")
def _tmpl_what_type():
    def q(f): return f"What type of feature is {f['name']}?"
    def a(f): return (
        f"{f['name']} is {_cat_desc(f['category'])} "
        f"on {_body_display(f['body'])}."
    )
    return q, a

@_register("type_identification")
def _tmpl_what_kind():
    def q(f): return f"What kind of geological feature is {f['name']} on {_body_display(f['body'])}?"
    def a(f): return (
        f"{f['name']} is classified as {_cat_desc(f['category'])} "
        f"located on {_body_display(f['body'])}."
    )
    return q, a

@_register("type_identification")
def _tmpl_describe():
    def q(f): return f"Describe the feature named {f['name']}."
    def a(f): return (
        f"{f['name']} is {_cat_desc(f['category'])} on the surface of "
        f"{_body_display(f['body'])}, situated at coordinates "
        f"{_fmt_coords(f)}."
    )
    return q, a

@_register("type_identification")
def _tmpl_classify():
    def q(f): return f"How is {f['name']} classified in planetary nomenclature?"
    def a(f): return (
        f"In the IAU planetary nomenclature system, {f['name']} is classified as "
        f"{_cat_desc(f['category'])} on {_body_display(f['body'])}."
    )
    return q, a


# location / coordinates

@_register("location")
def _tmpl_where():
    def q(f): return f"Where is {f['name']} located?"
    def a(f): return (
        f"{f['name']} is located on {_body_display(f['body'])} at "
        f"coordinates {_fmt_coords(f)}, in the "
        f"{_hemisphere(f['lat'])} hemisphere."
    )
    return q, a

@_register("location")
def _tmpl_coordinates():
    def q(f): return f"What are the coordinates of {f['name']}?"
    def a(f): return (
        f"The coordinates of {f['name']} on {_body_display(f['body'])} "
        f"are {_fmt_coords(f)}."
    )
    return q, a

@_register("location")
def _tmpl_hemisphere():
    def q(f): return f"In which hemisphere of {_body_display(f['body'])} is {f['name']}?"
    def a(f): return (
        f"{f['name']} is in the {_hemisphere(f['lat'])} hemisphere of "
        f"{_body_display(f['body'])}, at latitude {_fmt_lat(f['lat'])}."
    )
    return q, a

@_register("location")
def _tmpl_latitude():
    def q(f): return f"What is the latitude of {f['name']} on {_body_display(f['body'])}?"
    def a(f): return (
        f"The latitude of {f['name']} is {_fmt_lat(f['lat'])}."
    )
    return q, a

@_register("location")
def _tmpl_longitude():
    def q(f): return f"What is the longitude of {f['name']} on {_body_display(f['body'])}?"
    def a(f): return (
        f"The longitude of {f['name']} is {_fmt_lon(f['lon'])}."
    )
    return q, a


# body association

@_register("body_association")
def _tmpl_which_body():
    def q(f): return f"On which celestial body is {f['name']} found?"
    def a(f): return (
        f"{f['name']} is found on {_body_display(f['body'])}. "
        f"It is {_cat_desc(f['category'])}."
    )
    return q, a

@_register("body_association")
def _tmpl_is_on_body():
    def q(f): return f"Is {f['name']} on {_body_display(f['body'])}?"
    def a(f): return (
        f"Yes, {f['name']} is {_cat_desc(f['category'])} "
        f"located on {_body_display(f['body'])} at {_fmt_coords(f)}."
    )
    return q, a

@_register("body_association")
def _tmpl_tell_me_about():
    def q(f): return f"Tell me about {f['name']} on {_body_display(f['body'])}."
    def a(f): return (
        f"{f['name']} is {_cat_desc(f['category'])} on {_body_display(f['body'])}. "
        f"It is situated at {_fmt_coords(f)} "
        f"in the {_hemisphere(f['lat'])} hemisphere."
    )
    return q, a


# comparisons (pair-based, handled by the generator)

@_register("comparison")
def _tmpl_same_body():
    def q(f): return "__PAIR__"
    def a(f): return "__PAIR__"
    return q, a


# listing (aggregation, built from grouped data)──

@_register("listing")
def _tmpl_list_placeholder():
    def q(f): return "__LISTING__"
    def a(f): return "__LISTING__"
    return q, a


# yes/no factual

@_register("yes_no")
def _tmpl_is_crater():
    def q(f): return f"Is {f['name']} a crater?"
    def a(f):
        if f["category"] == "Crater":
            return f"Yes, {f['name']} is an impact crater on {_body_display(f['body'])}."
        return (
            f"No, {f['name']} is not a crater. It is {_cat_desc(f['category'])} "
            f"on {_body_display(f['body'])}."
        )
    return q, a

@_register("yes_no")
def _tmpl_is_on_mars():
    def q(f): return f"Is {f['name']} located on Mars?"
    def a(f):
        if f["body"] == "mars":
            return (
                f"Yes, {f['name']} is {_cat_desc(f['category'])} "
                f"located on Mars at {_fmt_coords(f)}."
            )
        return (
            f"No, {f['name']} is not on Mars. It is located on "
            f"{_body_display(f['body'])}."
        )
    return q, a

@_register("yes_no")
def _tmpl_is_in_southern():
    def q(f): return f"Is {f['name']} in the southern hemisphere of {_body_display(f['body'])}?"
    def a(f):
        hemi = _hemisphere(f["lat"])
        if hemi == "southern":
            return (
                f"Yes, {f['name']} is in the southern hemisphere of "
                f"{_body_display(f['body'])} at latitude {_fmt_lat(f['lat'])}."
            )
        return (
            f"No, {f['name']} is in the {hemi} region of "
            f"{_body_display(f['body'])} at latitude {_fmt_lat(f['lat'])}."
        )
    return q, a


# open-ended / summary

@_register("summary")
def _tmpl_summary():
    def q(f): return f"Give me a summary of {f['name']}."
    def a(f): return (
        f"{f['name']} is {_cat_desc(f['category'])} on {_body_display(f['body'])}. "
        f"It is located at {_fmt_coords(f)} in the {_hemisphere(f['lat'])} "
        f"hemisphere."
    )
    return q, a

@_register("summary")
def _tmpl_what_do_we_know():
    def q(f): return f"What do we know about the {_body_adj(f['body'])} feature {f['name']}?"
    def a(f): return (
        f"{f['name']} is {_cat_desc(f['category'])} identified on "
        f"{_body_display(f['body'])}. Its coordinates are {_fmt_coords(f)}, "
        f"placing it in the {_hemisphere(f['lat'])} hemisphere."
    )
    return q, a

@_register("summary")
def _tmpl_quick_facts():
    def q(f): return f"Provide quick facts about {f['name']}."
    def a(f): return (
        f"Quick facts — Name: {f['name']}; Body: {_body_display(f['body'])}; "
        f"Type: {f['category']}; Coordinates: {_fmt_coords(f)}; "
        f"Hemisphere: {_hemisphere(f['lat'])}."
    )
    return q, a


# proximity (spatial, assembled by the generator)────

@_register("proximity")
def _tmpl_nearby():
    def q(f): return "__PROXIMITY__"
    def a(f): return "__PROXIMITY__"
    return q, a


# --- data loading ---

def load_all_features() -> list[dict[str, Any]]:
    seen_ids: set[str] = set()
    features: list[dict] = []
    for path in FEATURE_FILES:
        if not path.exists():
            print(f"⚠  Skipping {path} (not found)")
            continue
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for feat in data:
            fid = feat.get("id", feat["name"])
            if fid not in seen_ids:
                seen_ids.add(fid)
                features.append(feat)
    print(f"✓ Loaded {len(features)} unique features from {len(FEATURE_FILES)} files")
    return features


# --- special generators (pairs, listings, proximity) ---

def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float,
                  body_radius_km: float = 1737.4) -> float:
    la1, la2, lo1, lo2 = map(math.radians, (lat1, lat2, lon1, lon2))
    dlat = la2 - la1
    dlon = lo2 - lo1
    a = math.sin(dlat / 2) ** 2 + math.cos(la1) * math.cos(la2) * math.sin(dlon / 2) ** 2
    return 2 * body_radius_km * math.asin(math.sqrt(a))

BODY_RADIUS_KM: dict[str, float] = {
    "moon": 1737.4,
    "mars": 3389.5,
    "mercury": 2439.7,
}


def generate_pair_questions(
    features: list[dict], rng: random.Random, count: int
) -> list[dict]:
    by_body: dict[str, list[dict]] = defaultdict(list)
    for f in features:
        by_body[f["body"]].append(f)

    pairs: list[dict] = []
    templates = [
        (
            "Are {a_name} and {b_name} both on {body}?",
            "Yes, both {a_name} and {b_name} are located on {body}. "
            "{a_name} is {a_desc} at {a_coords}, while {b_name} is {b_desc} at {b_coords}."
        ),
        (
            "Compare the locations of {a_name} and {b_name} on {body}.",
            "{a_name} is at {a_coords} in the {a_hemi} hemisphere, "
            "while {b_name} is at {b_coords} in the {b_hemi} hemisphere of {body}."
        ),
        (
            "What is the difference between {a_name} and {b_name} on {body}?",
            "{a_name} is {a_desc} at {a_coords}, whereas {b_name} is {b_desc} at {b_coords} on {body}."
        ),
    ]

    attempts = 0
    while len(pairs) < count and attempts < count * 10:
        attempts += 1
        body = rng.choice(list(by_body.keys()))
        if len(by_body[body]) < 2:
            continue
        a, b = rng.sample(by_body[body], 2)
        qt, at = rng.choice(templates)
        body_disp = _body_display(body)
        q = qt.format(a_name=a["name"], b_name=b["name"], body=body_disp)
        ans = at.format(
            a_name=a["name"], b_name=b["name"], body=body_disp,
            a_desc=_cat_desc(a["category"]), b_desc=_cat_desc(b["category"]),
            a_coords=_fmt_coords(a), b_coords=_fmt_coords(b),
            a_hemi=_hemisphere(a["lat"]), b_hemi=_hemisphere(b["lat"]),
        )
        pairs.append({"question": q, "answer": ans, "family": "comparison",
                       "features": [a["id"], b["id"]]})
    return pairs


def generate_listing_questions(
    features: list[dict], rng: random.Random, count: int
) -> list[dict]:
    by_body_cat: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for f in features:
        by_body_cat[(f["body"], f["category"])].append(f)

    listing_templates = [
        "Name some {cat_plural} on {body}.",
        "List a few {cat_plural} found on {body}.",
        "What are some notable {cat_plural} on {body}?",
        "Can you give examples of {cat_plural} on {body}?",
    ]

    CAT_PLURAL: dict[str, str] = {
        "Crater": "craters", "Mons": "mountains", "Tholus": "tholi",
        "Vallis": "valleys", "Patera": "paterae", "Rupes": "scarps",
        "Planitia": "plains", "Terra": "highland regions", "Dorsum": "ridges",
        "Catena": "crater chains", "Chaos": "chaotic terrains",
        "Fossa": "troughs", "Rima": "rilles", "Lacus": "small plains",
        "Regio": "regions", "Sinus": "bays",
    }

    items: list[dict] = []
    keys = [k for k, v in by_body_cat.items() if len(v) >= 3]
    attempts = 0
    while len(items) < count and attempts < count * 10:
        attempts += 1
        if not keys:
            break
        body, cat = rng.choice(keys)
        sample = rng.sample(by_body_cat[(body, cat)], min(5, len(by_body_cat[(body, cat)])))
        names = ", ".join(s["name"] for s in sample)
        cat_pl = CAT_PLURAL.get(cat, f"{cat.lower()} features")
        body_disp = _body_display(body)
        q = rng.choice(listing_templates).format(cat_plural=cat_pl, body=body_disp)
        a = f"Some {cat_pl} on {body_disp} include: {names}."
        items.append({"question": q, "answer": a, "family": "listing",
                       "features": [s["id"] for s in sample]})
    return items


def generate_proximity_questions(
    features: list[dict], rng: random.Random, count: int
) -> list[dict]:
    by_body: dict[str, list[dict]] = defaultdict(list)
    for f in features:
        by_body[f["body"]].append(f)

    prox_templates = [
        "What features are near {name} on {body}?",
        "Name a feature close to {name} on {body}.",
        "Which {adj} features are located near {name}?",
    ]

    items: list[dict] = []
    attempts = 0
    while len(items) < count and attempts < count * 20:
        attempts += 1
        body = rng.choice(list(by_body.keys()))
        if len(by_body[body]) < 10:
            continue
        anchor = rng.choice(by_body[body])
        radius = BODY_RADIUS_KM.get(body, 1737.4)
        neighbors = []
        for other in by_body[body]:
            if other["id"] == anchor["id"]:
                continue
            dist = _haversine_km(anchor["lat"], anchor["lon"],
                                  other["lat"], other["lon"], radius)
            if dist < 200:  # within 200 km
                neighbors.append((dist, other))
        if not neighbors:
            continue
        neighbors.sort(key=lambda x: x[0])
        nearest = neighbors[:3]
        body_disp = _body_display(body)
        q = rng.choice(prox_templates).format(
            name=anchor["name"], body=body_disp, adj=_body_adj(body))
        parts = [f"{n['name']} ({_cat_desc(n['category'])}, ~{d:.0f} km away)"
                 for d, n in nearest]
        a = (f"Features near {anchor['name']} on {body_disp} include: "
             + "; ".join(parts) + ".")
        items.append({"question": q, "answer": a, "family": "proximity",
                       "features": [anchor["id"]] + [n["id"] for _, n in nearest]})
    return items


# --- main generation pipeline ---

def generate_qa_dataset(features: list[dict], total: int = QA_TOTAL_TARGET,
                         seed: int = QA_SEED) -> list[dict]:
    rng = random.Random(seed)

    # separate single-feature templates from the special generators
    single_templates = [(fam, qf, af) for fam, qf, af in TEMPLATES
                        if qf(features[0]) not in ("__PAIR__", "__LISTING__", "__PROXIMITY__")]

    # budget: 70% single-feature, 10% pairs, 10% listing, 10% proximity
    n_single = int(total * 0.70)
    n_pair = int(total * 0.10)
    n_listing = int(total * 0.10)
    n_proximity = total - n_single - n_pair - n_listing

    # single-feature QA
    # round-robin families, random template within each
    fam_tmpls: dict[str, list] = defaultdict(list)
    for fam, qf, af in single_templates:
        fam_tmpls[fam].append((qf, af))

    families = list(fam_tmpls.keys())
    shuffled_features = features.copy()
    rng.shuffle(shuffled_features)

    single_items: list[dict] = []
    feat_idx = 0
    while len(single_items) < n_single:
        feat = shuffled_features[feat_idx % len(shuffled_features)]
        feat_idx += 1
        family = families[len(single_items) % len(families)]
        qf, af = rng.choice(fam_tmpls[family])
        single_items.append({
            "question": qf(feat),
            "answer": af(feat),
            "family": family,
            "features": [feat["id"]],
        })

    # multi-feature generators
    pair_items = generate_pair_questions(features, rng, n_pair)
    listing_items = generate_listing_questions(features, rng, n_listing)
    proximity_items = generate_proximity_questions(features, rng, n_proximity)

    # merge & shuffle
    all_items = single_items + pair_items + listing_items + proximity_items
    rng.shuffle(all_items)

    # Add sequential IDs
    for i, item in enumerate(all_items):
        item["id"] = i

    print(f"✓ Generated {len(all_items)} QA pairs")
    family_counts = defaultdict(int)
    for item in all_items:
        family_counts[item["family"]] += 1
    for fam, cnt in sorted(family_counts.items()):
        print(f"  {fam:25s}: {cnt:5d}")
    return all_items


# --- saving ---

def save_dataset(items: list[dict], train_ratio: float = QA_TRAIN_RATIO):
    rng = random.Random(QA_SEED + 1)
    shuffled = items.copy()
    rng.shuffle(shuffled)
    split = int(len(shuffled) * train_ratio)
    train, test = shuffled[:split], shuffled[split:]

    QA_OUTPUT_TRAIN.parent.mkdir(parents=True, exist_ok=True)

    with open(QA_OUTPUT_TRAIN, "w", encoding="utf-8") as f:
        for item in train:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(QA_OUTPUT_TEST, "w", encoding="utf-8") as f:
        for item in test:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(QA_OUTPUT_FULL, "w", encoding="utf-8") as f:
        json.dump(items, f, ensure_ascii=False, indent=2)

    print(f"✓ Saved {len(train)} train, {len(test)} test  →  {QA_OUTPUT_TRAIN.parent}")



def main():
    parser = argparse.ArgumentParser(description="Generate GeoRAG QA dataset")
    parser.add_argument("--total", type=int, default=QA_TOTAL_TARGET,
                        help="Total QA pairs to generate")
    parser.add_argument("--seed", type=int, default=QA_SEED)
    args = parser.parse_args()

    features = load_all_features()
    if not features:
        raise SystemExit("No features loaded — cannot generate QA dataset.")

    items = generate_qa_dataset(features, total=args.total, seed=args.seed)
    save_dataset(items)


if __name__ == "__main__":
    main()
