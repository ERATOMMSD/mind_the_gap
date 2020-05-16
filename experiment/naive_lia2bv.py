from typing import *
from util import Tree
import bv2lia


def naive_lia2bv_normalized_ineq(x_c: Dict[str, int], b: int, m: int) -> Any:
    if len(x_c) == 0:
        return "true" if b >= 0 else "false"
    elif len(x_c) == 1:
        coef, val = list(x_c.items())[0]
        if b // val > m - 1:
            return "true"
        else:
            return ["<=", coef, b // val]
    else:
        rev_x_c = {v: k for k, v in x_c.items()}
        val_big = max(rev_x_c)
        coef_big = rev_x_c[val_big]
        rest = {k: v for k, v in x_c.items() if k != coef_big}
        lits = []
        for i in range(b // val_big + 1):
            if i > m - 1:
                break
            lit = ["and", ["=", coef_big, i], naive_lia2bv_normalized_ineq(rest, b - i * val_big, m)]
            lits.append(lit)
        return ["or"] + lits


def naive_lia2bv_le_ineq(x_c: Dict[str, int], b: int, m: int) -> Any:
    return bv2lia.process_le_ineq(x_c, b, m, naive_lia2bv_normalized_ineq)


def naive_lia2bv_ineq(t: Tree, m: int) -> Any:
    return bv2lia.process_ineq(t, m, naive_lia2bv_le_ineq)


def naive_lia2bv_all_ineqs_in_tree(t: Tree, m: int) -> Any:
    return bv2lia.process_ineqs_in_tree(t, m, naive_lia2bv_ineq)
