from __future__ import annotations

def subject_disjoint_split(records:list[dict],test_subjects:set[str])->tuple[list[dict],list[dict]]:
    train=[]; test=[]
    for r in records: (test if str(r.get("subject_id")) in test_subjects else train).append(r)
    return train,test

def assert_subject_disjoint(train:list[dict],test:list[dict])->None:
    a={str(r.get("subject_id")) for r in train}; b={str(r.get("subject_id")) for r in test}; overlap=a&b
    if overlap: raise ValueError(f"Subject leakage detected: {sorted(overlap)[:10]}")
