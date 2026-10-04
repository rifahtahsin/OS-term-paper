"""Unit tests on the textbook queue (Silberschatz et al., head at cylinder 53).

FCFS = 640 and SSTF = 236 are the totals given in the textbook.
SCAN / C-SCAN totals are hand-computed under the conventions in schedulers.py.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from schedulers import fcfs, sstf, scan, cscan, UP, DOWN

Q, HEAD = [98, 183, 37, 122, 14, 124, 65, 67], 53

def test_fcfs():
    order, seek, *_ = fcfs(Q, HEAD)
    assert order == Q and seek == 640

def test_sstf():
    order, seek, *_ = sstf(Q, HEAD)
    assert order == [65, 67, 37, 14, 98, 122, 124, 183] and seek == 236

def test_scan_down():   # 53->37->14->0 (53) then 0->183 (183)
    order, seek, *_ = scan(Q, HEAD, DOWN)
    assert order == [37, 14, 65, 67, 98, 122, 124, 183] and seek == 236

def test_scan_up():     # 53->199 (146) then 199->14 (185)
    order, seek, *_ = scan(Q, HEAD, UP)
    assert seek == 331

def test_cscan():       # 53->199 (146) + jump 199 + 0->37 (37)
    order, seek, *_ = cscan(Q, HEAD)
    assert order == [65, 67, 98, 122, 124, 183, 14, 37] and seek == 382

def test_empty_and_single():
    for f in (fcfs, sstf, scan, cscan):
        assert f([], 50)[1] == 0
        assert f([60], 50)[1] == 10
