#!/usr/bin/env python3
"""Launch the full trait-audit battery: for a single model + species file,
post one job per trait-specific knowledge template (12 total) to the
running admin dashboard on localhost.

Each job asks the model, for every species in the species file, how much
it knows about a single phenotype (gram_staining, motility, ...). Together
with the existing species-level knowledge template (template3_knowlege),
this gives us 14 ratings per species — used to test whether the
species-level rating approximates the aggregate of the 13 trait-level
ratings.

Usage:
    python microbellm/templates/research/phenotype_analysis/sections/\
07k_knowledge_accuracy/trait_specific_pilot/launch_trait_audit_jobs.py \
        --model openai/gpt-4.1-nano \
        --species-file wa_with_gcount.txt

Optional:
    --admin-url   http://localhost:5051
    --include-species-level   also enqueue the species-level template
    --include-motility        also enqueue template1_knowledge_motility
    --dry-run                 print what would be submitted, don't POST
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path


TRAIT_SLUGS = [
    "gram_staining",
    "aerophilicity",
    "extreme_environment_tolerance",
    "biofilm_formation",
    "animal_pathogenicity",
    "biosafety_level",
    "health_association",
    "host_association",
    "plant_pathogenicity",
    "spore_formation",
    "hemolysis",
    "cell_shape",
]


def post_job(
    admin_url: str,
    species_file: str,
    model: str,
    system_template: str,
    user_template: str,
) -> dict:
    payload = json.dumps({
        "species_file": species_file,
        "model": model,
        "system_template": system_template,
        "user_template": user_template,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{admin_url.rstrip('/')}/api/create_and_run_job",
        data=payload,
        method="POST",
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="OpenRouter model slug, e.g. openai/gpt-4.1-nano")
    ap.add_argument("--species-file", default="wa_with_gcount.txt",
                    help="Species file name under data/ (default: wa_with_gcount.txt)")
    ap.add_argument("--admin-url", default="http://localhost:5051")
    ap.add_argument("--include-species-level", action="store_true",
                    help="Also enqueue templates/*/template3_knowlege")
    ap.add_argument("--include-motility", action="store_true",
                    help="Also enqueue templates/*/template1_knowledge_motility")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    slugs = list(TRAIT_SLUGS)
    if args.include_motility:
        slugs.insert(1, "motility")  # keeps gram_staining at index 0
    templates = [
        (f"templates/system/template1_knowledge_{s}.txt",
         f"templates/user/template1_knowledge_{s}.txt",
         s)
        for s in slugs
    ]
    if args.include_species_level:
        templates.append((
            "templates/system/template3_knowlege.txt",
            "templates/user/template3_knowlege.txt",
            "species_level",
        ))

    print(f"Model:        {args.model}")
    print(f"Species file: {args.species_file}")
    print(f"Admin:        {args.admin_url}")
    print(f"Jobs:         {len(templates)}")
    print()

    if args.dry_run:
        print("--- DRY RUN ---")
        for sys_t, usr_t, label in templates:
            print(f"  [{label:>34}] sys={sys_t}  usr={usr_t}")
        return

    results = []
    for i, (sys_t, usr_t, label) in enumerate(templates, 1):
        try:
            resp = post_job(
                admin_url=args.admin_url,
                species_file=args.species_file,
                model=args.model,
                system_template=sys_t,
                user_template=usr_t,
            )
        except urllib.error.HTTPError as e:
            body = e.read().decode("utf-8", errors="ignore")
            print(f"  [{i:>2}/{len(templates)}] {label:>34}  HTTP {e.code}: {body[:200]}")
            results.append({"label": label, "ok": False, "error": body})
            continue
        except Exception as e:
            print(f"  [{i:>2}/{len(templates)}] {label:>34}  ERR: {e}")
            results.append({"label": label, "ok": False, "error": str(e)})
            continue

        ok = bool(resp.get("success"))
        job_id = resp.get("job_id", "?")
        tag = "OK " if ok else "FAIL"
        print(f"  [{i:>2}/{len(templates)}] {label:>34}  {tag}  job_id={job_id}")
        results.append({"label": label, "ok": ok, "job_id": job_id})

        # Gentle delay so admin can bookkeep between job creations
        time.sleep(0.5)

    print()
    ok = sum(1 for r in results if r["ok"])
    print(f"Summary: {ok}/{len(templates)} jobs submitted successfully.")
    if ok < len(templates):
        print("Failures:")
        for r in results:
            if not r["ok"]:
                print(f"  - {r['label']}: {r.get('error', '')[:200]}")
        sys.exit(1)


if __name__ == "__main__":
    main()
