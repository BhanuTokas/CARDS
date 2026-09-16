"""Maps the original 43-class CORINE Land Cover (CLC) Level-3 nomenclature
used by BigEarthNet's own image folders to the collapsed 19-class
nomenclature the CounterConcept paper's FTW experiment (Section 4.5) uses
as its concept set -- extracted directly from the official BigEarthNet
class-list table (bigearth.eu/BigEarthNetListofClasses.pdf, Table I),
cross-checked against Sumbul et al.'s own stated counts in the
BigEarthNet-MM paper (arXiv:2105.07921): "Ten classes of the original CLC
nomenclature are maintained in the new nomenclature, 22 classes are
grouped into 9 new classes, and 11 classes are removed" -- this module's
own REMOVED_43_CLASSES has exactly 11 entries and BIGEARTHNET_43_TO_19 has
exactly 32 (10 identity + 22 grouped), verified directly below.

Names are kept in the PDF table's own exact spelling/punctuation (the
canonical source) -- `Datasets/bigearthnet_samples/`'s own folder names
differ in minor punctuation for at least one class ("Transitional woodland
or shrub" vs. this table's "Transitional woodland-shrub"); any loader
reading those folders needs to normalize against BIGEARTHNET_43_TO_19's
keys rather than assume an exact string match, see
`verify_local_folder_coverage` below.
"""

from __future__ import annotations

from pathlib import Path

# The 19-class nomenclature the CounterConcept paper's FTW experiment uses
# as its BigEarthNet concept set.
BIGEARTHNET_19_CLASSES: list[str] = [
    "Urban fabric",
    "Industrial or commercial units",
    "Arable land",
    "Permanent crops",
    "Pastures",
    "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas",
    "Broad-leaved forest",
    "Coniferous forest",
    "Mixed forest",
    "Natural grassland and sparsely vegetated areas",
    "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland-shrub",
    "Beaches, dunes, sands",
    "Inland wetlands",
    "Coastal wetlands",
    "Inland waters",
    "Marine waters",
]

# 32 of the 43 original CLC classes -> their 19-class equivalent (10 kept
# as-is under the same name, 22 grouped into the 9 non-identity targets).
BIGEARTHNET_43_TO_19: dict[str, str] = {
    "Continuous urban fabric": "Urban fabric",
    "Discontinuous urban fabric": "Urban fabric",
    "Industrial or commercial units": "Industrial or commercial units",
    "Non-irrigated arable land": "Arable land",
    "Permanently irrigated land": "Arable land",
    "Rice fields": "Arable land",
    "Vineyards": "Permanent crops",
    "Fruit trees and berry plantations": "Permanent crops",
    "Olive groves": "Permanent crops",
    "Annual crops associated with permanent crops": "Permanent crops",
    "Pastures": "Pastures",
    "Complex cultivation patterns": "Complex cultivation patterns",
    "Land principally occupied by agriculture, with significant areas of natural vegetation":
        "Land principally occupied by agriculture, with significant areas of natural vegetation",
    "Agro-forestry areas": "Agro-forestry areas",
    "Broad-leaved forest": "Broad-leaved forest",
    "Coniferous forest": "Coniferous forest",
    "Mixed forest": "Mixed forest",
    "Natural grassland": "Natural grassland and sparsely vegetated areas",
    "Sparsely vegetated areas": "Natural grassland and sparsely vegetated areas",
    "Moors and heathland": "Moors, heathland and sclerophyllous vegetation",
    "Sclerophyllous vegetation": "Moors, heathland and sclerophyllous vegetation",
    "Transitional woodland-shrub": "Transitional woodland-shrub",
    "Beaches, dunes, sands": "Beaches, dunes, sands",
    "Inland marshes": "Inland wetlands",
    "Peatbogs": "Inland wetlands",
    "Salt marshes": "Coastal wetlands",
    "Salines": "Coastal wetlands",
    "Water courses": "Inland waters",
    "Water bodies": "Inland waters",
    "Coastal lagoons": "Marine waters",
    "Estuaries": "Marine waters",
    "Sea and ocean": "Marine waters",
}

# The 11 CLC classes with no 19-class equivalent -- land-use-dependent
# (Airports, Port areas, ...) or requiring time-series data single-date
# imagery can't provide (Intertidal flats), per the BigEarthNet-MM paper's
# own stated rationale. Kept explicit rather than silently absent from
# BIGEARTHNET_43_TO_19, matching this project's own EXCLUDED_ATTRIBUTES
# convention (celeba_attributes.py).
REMOVED_43_CLASSES: set[str] = {
    "Road and rail networks and associated land",
    "Port areas",
    "Airports",
    "Mineral extraction sites",
    "Dump sites",
    "Construction sites",
    "Green urban areas",
    "Sport and leisure facilities",
    "Bare rock",
    "Burnt areas",
    "Intertidal flats",
}

assert len(REMOVED_43_CLASSES) == 11
assert len(BIGEARTHNET_43_TO_19) == 32
assert not (set(BIGEARTHNET_43_TO_19) & REMOVED_43_CLASSES)
assert set(BIGEARTHNET_43_TO_19.values()) == set(BIGEARTHNET_19_CLASSES)


def verify_local_folder_coverage(samples_root: Path) -> dict[str, list[str]]:
    """Diagnostic, not used at import time: compares
    `samples_root`'s own subfolder names against this module's 43-class
    keys (mapped + removed), so a naming mismatch (confirmed to exist for
    at least "Transitional woodland-shrub" vs. the local "Transitional
    woodland or shrub") is caught explicitly rather than silently
    dropping that concept everywhere downstream.

    Returns {"matched": [...], "unmatched_folders": [...], "missing_classes": [...]}.
    """
    known_43 = set(BIGEARTHNET_43_TO_19) | REMOVED_43_CLASSES
    folder_names = {p.name for p in samples_root.iterdir() if p.is_dir()}
    return {
        "matched": sorted(folder_names & known_43),
        "unmatched_folders": sorted(folder_names - known_43),
        "missing_classes": sorted(known_43 - folder_names),
    }
