import copy
import math
from fractions import Fraction
from typing import *

from util import lcm_many, compare_trees, Tree, SMTUtilOption, get_tag, get_coefs_fraction, convert_int2list, \
    find_subtrees_by_tags, get_lets


def look_into_floor(to_int: Tree, vars: Dict[str, Tuple[int, int]], lets: Dict[str, Any]) -> Tuple[
    Tree, List[int], int]:
    def get_bounds(coefs: Dict[str, Fraction], const: Fraction) -> Tuple[Fraction, Fraction]:
        val_min = Fraction(0)
        val_max = Fraction(0)
        assert set(vars.keys()) >= set(coefs.keys())
        for k in coefs.keys():
            if coefs[k] >= 0:
                val_min += coefs[k] * vars[k][0]
                val_max += coefs[k] * vars[k][1]
            else:
                val_min += coefs[k] * vars[k][1]
                val_max += coefs[k] * vars[k][0]
        val_min += const
        val_max += const
        return val_min, val_max

    assert get_tag(to_int[0]) in ["to_int", "floor"]
    content_to_int = to_int[1]
    coefs, const = get_coefs_fraction(content_to_int, lets)
    assert set(coefs.keys()) <= set(vars.keys())
    # TODO
    content_min_q, content_max_q = get_bounds(coefs, const)
    # content_min = int(math.ceil(content_min_q))
    content_min = int(math.floor(content_min_q))
    content_max = int(math.floor(content_max_q))
    values_to_int = list(range(content_min, content_max + 1))
    den = lcm_many([q.denominator for q in coefs.values()] + [const.denominator])
    num_coefs = {k: int(v * den) for k, v in coefs.items()}
    num_const = int(const * den)
    expression = ["+"] + [["*", k, convert_int2list(v)] for k, v in num_coefs.items()] + [num_const]  # type: ignore
    return expression, values_to_int, den


def reduce_floor_strategy1(t: Tree, to_int: Tree, parents: List[Tuple[Tree, int]], vars: Dict[str, Tuple[int, int]],
                           lets: Dict[str, Any]) -> Tree:
    expression, values_to_int, den = look_into_floor(to_int, vars, lets)
    segments: List[Any] = []
    for v in values_to_int:
        cond: List[Any] = []
        left = convert_int2list(v * den)
        right = convert_int2list((v + 1) * den - 1)
        cond.append(["<=", left, expression])
        cond.append(["<=", expression, right])
        for parent, id in parents:
            parent[id] = convert_int2list(v)
        seg = ["and"] + cond + [t]
        segments.append(copy.deepcopy(seg))
    return ["or"] + segments


def is_power_of_two(n):
    while(True):
        if n % 2 == 0:
            n = n // 2
        else:
            return n == 1
    assert False



def reduce_floor_strategy2(t: Tree, to_int: Tree, parents: List[Tuple[Tree, int]], vars: Dict[str, Tuple[int, int]],
                           lets: Dict[str, Any], counter: int, m: int) -> Tuple[Tree, Optional[Tuple[str, Tree]], bool]:
    # check the application condition
    expression, _, den = look_into_floor(to_int, vars, lets)
    for parent, id in parents:
        if parent[0] == "*":
            if type(parent[1]) is int or (type(parent[1]) is str and parent[1].lstrip("+-").isnumeric()):
                if int(parent[1]) % den == 0:
                    if den < m / 2:
                        if is_power_of_two(den):
                            pass
                        else:
                            return t, None, False
                    else:
                        return t, None, False
                else:
                    return t, None, False
            else:
                return t, None, False
        else:
            return t, None, False
    varname = f"var_strategy2_{counter}"
    replacing = ["-", expression, ["mod", expression, den]]
    assert den % m != 0
    for parent, id in parents:
        parent[1] = int(parent[1]) // den
        parent[2] = varname
    return t, (varname, replacing), True


def reduce_float_from_tree(t: Tree, vars: Dict[str, Tuple[int, int]], m: int, opt: SMTUtilOption,
                           force_strategy: Optional[str] = None) -> Any:
    def get_to_int_all(tree: Any, parent: Any = None, id: int = -1) -> List[Tuple[Any, Any, int]]:
        return find_subtrees_by_tags(["to_int", "floor"], tree, parent, id)

    counter = 0
    t = copy.deepcopy(t)
    replacements_floor: Dict[str, Tree] = {}
    while True:
        to_ints = get_to_int_all(t)
        if not to_ints:
            break
        to_ints_to_parents: List[Tuple[Tree, List[Tuple[Tree, int]]]] = []
        for to_int, parent, id in to_ints:
            added = False
            for k, v in to_ints_to_parents:
                if compare_trees(k, to_int):
                    v.append((parent, id))
                    added = True
                    break
            if not added:
                to_ints_to_parents.append((to_int, [(parent, id)]))
        lets = get_lets(t)
        to_int, parents = to_ints_to_parents[0]
        # force_strategy = "strategy1"  # strategy 2 has a bug!
        if force_strategy is None:
            force_strategy = opt.floor_reduce
        if force_strategy == "strategy1":
            t = reduce_floor_strategy1(t, to_int, parents, vars, lets)
        elif force_strategy == "strategy2":
            t, replacement, res_flag = reduce_floor_strategy2(t, to_int, parents, vars, lets, counter, m)
            counter += 1
            if res_flag == False:
                t = reduce_floor_strategy1(t, to_int, parents, vars, lets)
            if replacement is not None:
                replacements_floor[replacement[0]] = replacement[1]
        else:
            assert False
    return t, replacements_floor


def process_le_ineq(x_c: Dict[str, int], b: int, m: int, f: Callable[[Dict[str, int], int, int], Any]) -> Any:
    flipped: List[str] = []
    b_added = 0
    # flip
    for k in x_c.keys():
        if x_c[k] < 0:
            x_c[k] *= -1
            flipped.append(k)
            b_added += (m - 1) * x_c[k]
            b += (m - 1) * x_c[k]
    # print("flipped", x_c, b)
    if b < 0:
        return "false"
    # reduce zeors
    x_c = {k: v for k, v in x_c.items() if v != 0}
    # calc
    boxed = f(x_c, b, m)

    # print("calc", boxed)
    # flip again
    def help(tree: Any, parent: Any = None, id: int = -1) -> None:
        if type(tree) is str and tree in flipped:
            if parent is None:
                assert False
            else:
                parent[id] = ["-", m - 1, tree]  # type: ignore
        elif type(tree) is list:
            for i in range(len(tree)):
                help(tree[i], tree, i)

    help(boxed)
    return boxed


def process_ineq(t: Tree, m: int, f: Callable[[Dict[str, int], int, int], Any]) -> Any:
    l_coefs1, l_const1 = get_coefs_fraction(t[1], {})
    r_coefs1, r_const1 = get_coefs_fraction(t[2], {})
    l_coefs = {k: v.numerator for k, v in l_coefs1.items()}
    r_coefs = {k: v.numerator for k, v in r_coefs1.items()}
    l_const = l_const1.numerator
    r_const = r_const1.numerator
    vars = set(list(l_coefs.keys()) + list(r_coefs.keys()))
    coefs: Dict[str, int] = {}
    tag = get_tag(t[0])
    if tag == "<=":
        for k in vars:
            coefs[k] = l_coefs.get(k, 0) - r_coefs.get(k, 0)
        const = r_const - l_const
        # print(l_coefs, l_const, r_coefs, r_const)
        # print(coefs, const)
        t = f(coefs, const, m)
    elif tag == ">=":
        for k in vars:
            coefs[k] = r_coefs.get(k, 0) - l_coefs.get(k, 0)
        const = l_const - r_const
        t = f(coefs, const, m)
        pass
    elif tag == "<":
        for k in vars:
            coefs[k] = l_coefs.get(k, 0) - r_coefs.get(k, 0)
        const = r_const - l_const - 1
        t = f(coefs, const, m)
    elif tag == ">":
        for k in vars:
            coefs[k] = r_coefs.get(k, 0) - l_coefs.get(k, 0)
        const = l_const - r_const - 1
        t = f(coefs, const, m)
    else:
        assert False
    return t


def process_ineqs_in_tree(t: Tree, m: int, f: Callable[[Tree, int], Any]) -> Any:
    def help(tree: Tree) -> Any:
        if type(tree) is list:
            if type(tree[0]) is str:
                if get_tag(tree[0]) in ["<=", ">=", "<", ">"]:
                    return f(tree, m)
            return [help(i) for i in tree]
        else:
            return tree

    return help(t)

if __name__ == "__main__":
    pass