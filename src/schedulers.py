"""Classical disk-scheduling algorithms: FCFS, SSTF, SCAN, C-SCAN.
Disk model (textbook convention, Silberschatz et al.):
  * cylinders numbered 0 .. MAX_CYL (default 199)
  * seek cost = total number of cylinders the head travels

Every scheduler returns (service_order, total_seek, final_head, final_direction).

Conventions (stated explicitly because textbooks differ):
  * SCAN  : sweeps in the current direction to the END of the disk, then reverses.
            If no request remains on the other side, the arm stops at the last
            request served (no pointless trip to the edge).
  * C-SCAN: sweeps toward MAX_CYL, travels to MAX_CYL, jumps back to cylinder 0,
            and continues upward. The return jump (MAX_CYL -> 0) IS counted in
            the seek total. If nothing remains below the start, no jump is made.
  * SSTF  : ties in distance are broken toward the LOWER cylinder.
"""
from __future__ import annotations

from typing import List, Sequence, Tuple

MAX_CYL = 199
UP, DOWN = +1, -1

Result = Tuple[List[int], int, int, int]


def _seek(path: Sequence[int]) -> int:
    return sum(abs(b - a) for a, b in zip(path[:-1], path[1:]))


def fcfs(requests: Sequence[int], head: int, direction: int = UP, max_cyl: int = MAX_CYL) -> Result:
    order = list(requests)
    path = [head] + order
    last = path[-1]
    new_dir = direction if len(path) < 2 or path[-1] == path[-2] else (UP if path[-1] > path[-2] else DOWN)
    return order, _seek(path), last, new_dir


def sstf(requests: Sequence[int], head: int, direction: int = UP, max_cyl: int = MAX_CYL) -> Result:
    pending = list(requests)
    order, pos = [], head
    while pending:
        # nearest request; tie -> lower cylinder
        idx = min(range(len(pending)), key=lambda i: (abs(pending[i] - pos), pending[i]))
        nxt = pending.pop(idx)
        order.append(nxt)
        pos = nxt
    path = [head] + order
    new_dir = direction if len(path) < 2 or path[-1] == path[-2] else (UP if path[-1] > path[-2] else DOWN)
    return order, _seek(path), pos, new_dir


def scan(requests: Sequence[int], head: int, direction: int = UP, max_cyl: int = MAX_CYL) -> Result:
    if not requests:
        return [], 0, head, direction
    if direction == UP:
        first = sorted(r for r in requests if r >= head)
        second = sorted((r for r in requests if r < head), reverse=True)
        edge = max_cyl
    else:
        first = sorted((r for r in requests if r <= head), reverse=True)
        second = sorted(r for r in requests if r > head)
        edge = 0
    order = first + second
    if second:  # must reverse -> travel to the disk edge first
        path = [head] + first + [edge] + second
        new_dir = -direction
    else:
        path = [head] + first
        new_dir = direction
    return order, _seek(path), path[-1], new_dir


def cscan(requests: Sequence[int], head: int, direction: int = UP, max_cyl: int = MAX_CYL) -> Result:
    """C-SCAN always services in the UP direction (direction arg ignored)."""
    if not requests:
        return [], 0, head, UP
    upper = sorted(r for r in requests if r >= head)
    lower = sorted(r for r in requests if r < head)
    order = upper + lower
    if lower:  # go to the end, jump to 0 (counted), continue upward
        path = [head] + upper + [max_cyl, 0] + lower
    else:
        path = [head] + upper
    return order, _seek(path), path[-1], UP


SCHEDULERS = {"FCFS": fcfs, "SSTF": sstf, "SCAN": scan, "CSCAN": cscan}
NAMES = list(SCHEDULERS)  # fixed order, also used as tie-break preference
