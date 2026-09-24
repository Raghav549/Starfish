from adapters.xyt import parse_xyt
from matching import greedy_match
from metrics import summarize
from splits import assert_subject_disjoint

def test_xyt_parser():
    pts=parse_xyt("10 20 1.25 80\n30 40 2.0 70\n")
    assert pts[0]["x"]==10 and pts[0]["quality"]==80

def test_matching_and_metrics():
    truth=[{"x":10,"y":20,"type":"ending","angle":1.0},{"x":40,"y":50,"type":"bifurcation","angle":2.0}]
    pred=[{"x":11,"y":21,"type":"ending","angle":1.1},{"x":100,"y":100,"type":"ending","angle":0.5}]
    pairs=greedy_match(pred,truth,3)
    m=summarize(pred,truth,pairs)
    assert m["tp"]==1 and m["fp"]==1 and m["fn"]==1

def test_subject_disjoint():
    assert_subject_disjoint([{"subject_id":"a"}],[{"subject_id":"b"}])
