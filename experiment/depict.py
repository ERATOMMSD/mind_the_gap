import smtutil
import itertools
from typing import *

import treeutil
import util


def depict_bv(t: util.Tree, m: int) -> Tuple[str, List[str]]:
    vs = list(set(treeutil.get_variables(t)))
    vs.sort()
    if len(vs) == 0:
        res = treeutil.evaluate_bv(t, m, {})
        return f"[Could not draw because of no variable: {res}]\n", vs
    elif len(vs) > 2:
        return f"[Could not draw because of too many variables: {vs}\n", vs
    assert len(vs) <= 2
    rows = range(m)
    txt = ""
    if len(vs) == 1:
        vs.append("hoge")
        cols = [0]
    else:
        cols = range(m - 1, -1, -1)
        txt += f"^{vs[1]}\n"
    txt += (f">{vs[0]}\n")
    for col in cols:
        for row in rows:
            res = treeutil.evaluate_bv(t, m, {vs[0]: row, vs[1]: col})
            txt += "*" if res else "."
        txt += "\n"
    return txt, vs
