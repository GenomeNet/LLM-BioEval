#!/usr/bin/env python3
"""Generate trait-specific knowledge templates for the 12 phenotypes that
don't yet have one (motility already has template1_knowledge_motility.*).

Writes three files per phenotype:
  templates/system/template1_knowledge_<slug>.txt
  templates/user/template1_knowledge_<slug>.txt
  templates/validation/template1_knowledge_<slug>.json

All templates share the same structural wording (mirroring the motility
pilot) so that downstream aggregation remains an apples-to-apples
comparison — only the trait-specific context changes between files.

Run once from the repo root:
    python microbellm/templates/research/phenotype_analysis/sections/\
07k_knowledge_accuracy/trait_specific_pilot/generate_trait_templates.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Dict

REPO_ROOT = Path(__file__).resolve().parents[7]
TPL_ROOT = REPO_ROOT / "templates"

# (db_column, human_trait_name, what_to_look_for, moderate_examples,
#  extensive_examples)
TRAITS: Dict[str, Dict[str, str]] = {
    "gram_staining": {
        "name": "gram staining",
        "look_for": "cell-wall architecture and how the organism behaves under Gram's staining procedure",
        "moderate": "basic staining observations or cell-wall composition notes",
        "extensive": "detailed cell-wall chemistry (peptidoglycan thickness, teichoic acid content), membrane architecture, and consistent microscopic characterisation",
    },
    "aerophilicity": {
        "name": "aerophilicity",
        "look_for": "oxygen requirements and respiratory lifestyle (aerobic, anaerobic, facultative, aerotolerant)",
        "moderate": "basic oxygen-tolerance observations or growth-condition notes on standard media",
        "extensive": "detailed respiratory-metabolism studies (terminal oxidases, electron-transport chain components, anaerobic energy conservation, oxygen-sensitivity thresholds)",
    },
    "extreme_environment_tolerance": {
        "name": "extreme-environment tolerance",
        "look_for": "growth and survival under extreme conditions (temperature, pH, salinity, pressure, radiation, desiccation)",
        "moderate": "reports of survival at non-standard conditions or basic tolerance-range observations",
        "extensive": "mechanistic studies of stress responses (compatible solutes, heat-shock proteins, DNA-repair systems, membrane adaptations) and well-characterised tolerance boundaries",
    },
    "biofilm_formation": {
        "name": "biofilm formation",
        "look_for": "ability to form biofilms, surface adhesion, and biofilm architecture",
        "moderate": "basic observations of biofilm growth or adhesion assays on standard substrates",
        "extensive": "mechanistic studies (quorum sensing, exopolysaccharide composition, dispersal signalling) and reproducible experimental biofilm characterisation",
    },
    "animal_pathogenicity": {
        "name": "animal pathogenicity",
        "look_for": "pathogenic potential toward animals (including humans), disease manifestations, and virulence determinants",
        "moderate": "reports of clinical isolation or basic virulence observations",
        "extensive": "detailed virulence mechanisms (toxins, adhesins, immune evasion), defined infection models, and epidemiological characterisation",
    },
    "biosafety_level": {
        "name": "biosafety level",
        "look_for": "biosafety classification based on risk assessment, transmissibility, and disease severity",
        "moderate": "an assigned biosafety level from a recognised authority (ABSA, WHO, national lists)",
        "extensive": "detailed risk-group discussion including exposure routes, laboratory-acquired-infection history, and containment recommendations",
    },
    "health_association": {
        "name": "health association",
        "look_for": "association with host health (commensal, probiotic, or beneficial roles)",
        "moderate": "reports of presence in healthy microbiota or basic beneficial-role observations",
        "extensive": "detailed studies of commensal/probiotic mechanisms, host health outcomes, and reproducible health-association evidence",
    },
    "host_association": {
        "name": "host association",
        "look_for": "host-colonisation patterns and lifestyle (free-living, commensal, parasitic, symbiotic)",
        "moderate": "basic host-range observations or isolation-source notes",
        "extensive": "detailed host-interaction studies (colonisation factors, host-specificity determinants, ecological-niche characterisation)",
    },
    "plant_pathogenicity": {
        "name": "plant pathogenicity",
        "look_for": "pathogenic potential toward plants, plant-disease manifestations, and phytopathogenic mechanisms",
        "moderate": "reports of plant infection or isolation from diseased plants",
        "extensive": "detailed phytopathogenic-mechanism studies (effector proteins, type III secretion, host-susceptibility determinants) and characterised plant disease cycles",
    },
    "spore_formation": {
        "name": "spore formation",
        "look_for": "endospore formation, sporulation regulation, and spore properties",
        "moderate": "basic observations of spore formation under starvation conditions or morphological descriptions",
        "extensive": "detailed sporulation regulation (sigma-factor cascade, forespore/mother-cell differentiation), spore ultrastructure, and germination mechanisms",
    },
    "hemolysis": {
        "name": "hemolysis",
        "look_for": "hemolytic activity on blood agar (alpha, beta, gamma) and the underlying hemolysins",
        "moderate": "basic blood-agar observations or simple hemolysin descriptions",
        "extensive": "detailed hemolysin characterisation (gene identity, regulatory control, pore-forming mechanisms) and reproducible hemolytic-phenotype documentation",
    },
    "cell_shape": {
        "name": "cell shape",
        "look_for": "cellular morphology (bacillus, coccus, spirillum, other shapes) and shape-determining factors",
        "moderate": "basic light-microscopy descriptions",
        "extensive": "detailed shape-determinant studies (MreB/FtsZ roles, peptidoglycan architecture) and consistent microscopic/electron-microscopic characterisation",
    },
}

SYSTEM_TEMPLATE = """Determine your level of scientific knowledge specifically about the {name} phenotype of the given binomial species name, based on the depth of species-specific literature describing {look_for}:

- limited: Little or no species-specific literature about this organism's {name}. You cannot confidently state this phenotype except by generic inference from its genus or higher taxonomic rank.
- moderate: Some species-specific information on {name} is available, including {moderate}, or a small number of primary-literature reports.
- extensive: Comprehensive species-specific literature on {name}, including {extensive}, and consistent coverage across multiple independent publications.

If the species name is not a real or recognized species, or if you cannot meaningfully separate your knowledge of its {name} from a generic taxonomic assumption, respond with NA.
"""

USER_TEMPLATE = """Respond with a JSON object for {{binomial_name}} indicating your level of species-specific scientific knowledge about its {name} phenotype, in lowercase, in this format:

{{
    "knowledge_group": "<limited|moderate|extensive|NA>"
}}
"""


def validation_config(slug: str, name: str) -> dict:
    return {
        "template_info": {
            "name": f"template1_knowledge_{slug}",
            "type": "knowledge",
            "description": f"Trait-specific knowledge-level assessment template ({name}).",
            "version": "1.0",
            "purpose": (
                "Member of the trait-audit battery generated for the reviewer "
                "rebuttal: asks the model, for a single phenotype, how much "
                "species-specific knowledge it holds. Used alongside the "
                f"species-level knowledge-rating template to test whether "
                f"species-level confidence approximates the aggregate of "
                f"trait-specific confidences across all 13 phenotypes."
            ),
            "usage_context": {
                "when_to_use": (
                    f"Run together with the other 12 trait-specific knowledge "
                    f"templates and the species-level knowledge-rating "
                    f"template on a shared species sample; compare the "
                    f"species-level rating to aggregates (mean/max/mode) of "
                    f"the 13 trait-specific ratings."
                ),
                "typical_workflow": (
                    "(1) Pick a species sample (e.g. trait_audit_sample.txt). "
                    "(2) Run species-level template3_knowlege. "
                    "(3) Run all 13 trait-specific template1_knowledge_* "
                    "templates on the same sample + model. "
                    "(4) Aggregate the 13 trait ratings per species and "
                    "correlate with the species-level rating."
                ),
            },
            "interpretation_guide": {
                "limited": f"Only generic, taxonomy-derived expectation about {name}; no species-specific literature recalled.",
                "moderate": f"Some species-specific {name} observations or mechanistic notes recalled.",
                "extensive": f"Rich species-specific {name} literature recalled, including mechanistic, regulatory, and phenotypic detail.",
                "NA": "Cannot place the species in any of the above tiers (unrecognised name, or no meaningful trait-specific signal beyond taxonomic inference).",
            },
            "quality_indicators": {
                "high_quality_response": f"Clear categorisation that is coherent with other trait-specific ratings for the same species.",
                "low_quality_response": f"Categorisation appears to mirror taxonomic family-level guesses rather than species-specific recall.",
            },
        },
        "expected_response": {
            "format": "json",
            "required_fields": ["knowledge_group"],
            "optional_fields": [],
        },
        "field_definitions": {
            "knowledge_group": {
                "type": "string",
                "required": True,
                "description": f"Trait-specific knowledge level for {name} of the given organism.",
                "allowed_values": ["limited", "moderate", "extensive", "NA"],
                "validation_rules": {
                    "case_sensitive": False,
                    "trim_whitespace": True,
                    "normalize_mapping": {
                        "limited": ["limited", "minimal", "basic", "low", "little", "poor"],
                        "moderate": ["moderate", "medium", "intermediate", "fair", "some"],
                        "extensive": ["extensive", "comprehensive", "detailed", "high", "full", "complete", "thorough"],
                        "NA": ["na", "n/a", "none", "unknown", "not applicable"],
                    },
                },
                "validation_error_messages": {
                    "missing": "Required field 'knowledge_group' is missing from response",
                    "invalid_value": "Invalid knowledge level. Expected one of: limited, moderate, extensive, NA",
                    "wrong_type": "Field 'knowledge_group' must be a string",
                },
            },
        },
        "parsing_instructions": {
            "json_extraction": {
                "method": "regex",
                "pattern": "\\{.*\\}",
                "flags": ["DOTALL"],
            },
            "fallback_parsing": {
                "enabled": True,
                "method": "keyword_search",
                "keywords": ["knowledge_group", "knowledge level", "level"],
            },
        },
        "success_criteria": {
            "minimum_required_fields": 1,
            "require_all_mandatory": True,
            "allow_extra_fields": False,
        },
        "error_handling": {
            "on_parse_failure": "return_null",
            "on_validation_failure": "return_errors",
            "on_missing_required": "return_errors",
        },
    }


def main() -> None:
    sys_dir = TPL_ROOT / "system"
    usr_dir = TPL_ROOT / "user"
    val_dir = TPL_ROOT / "validation"
    for d in (sys_dir, usr_dir, val_dir):
        d.mkdir(parents=True, exist_ok=True)

    written = 0
    skipped = 0
    for slug, spec in TRAITS.items():
        name = spec["name"]
        sys_path = sys_dir / f"template1_knowledge_{slug}.txt"
        usr_path = usr_dir / f"template1_knowledge_{slug}.txt"
        val_path = val_dir / f"template1_knowledge_{slug}.json"

        # System
        sys_txt = SYSTEM_TEMPLATE.format(
            name=name,
            look_for=spec["look_for"],
            moderate=spec["moderate"],
            extensive=spec["extensive"],
        )
        sys_path.write_text(sys_txt)

        # User (note: str.format escapes {{ / }})
        usr_txt = USER_TEMPLATE.format(name=name)
        usr_path.write_text(usr_txt)

        # Validation JSON
        cfg = validation_config(slug, name)
        val_path.write_text(json.dumps(cfg, indent=2) + "\n")

        written += 3
        print(f"  wrote {slug:>34}  → system/user/validation")

    print(f"\nDone. {written} files written (12 traits × 3).")


if __name__ == "__main__":
    main()
