from __future__ import annotations

FIELDS=["status","foreground_ratio","coherence_mean","coherence_p10","coherence_p50","coherence_p90",
        "ridge_frequency_median","ridge_frequency_p10","ridge_frequency_p90","frequency_coverage",
        "minutiae_density","orientation_concentration"]

def recognition_ready(quality:dict)->bool: return quality.get("status")=="usable"
def quality_fields(quality:dict)->dict: return {k:quality[k] for k in FIELDS if k in quality}
