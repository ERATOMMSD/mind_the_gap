import functools
import math
from fractions import Fraction
from typing import *
from typing import List, Any, Dict, Tuple
import re

def implies(a: bool, b: bool) -> bool:
    return (not a) or b

def debug_print_list(x):
    if isinstance(x, list):
        return f"({' '.join([debug_print_list(y) for y in x])})"
    else:
        return str(x)

def debug_print_preds(xs):
    xx = []
    for x in xs:
        print(debug_print_list(x))
        xx.append(x)
    return xx

def debug_print_smtform(x):
    if type(x) is list:
        return "(" + " ".join([debug_print_smtform(y) for y in x]) + ")"
    elif type(x) is int:
        return str(x)
    elif type(x) is bool:
        return "true" if x else "false"
    elif type(x) is str:
        return x
    else:
        assert False


def compare_trees(x: Any, y: Any) -> bool:
    if type(x) != type(y):
        return False
    else:
        if type(x) == list:
            if len(x) == len(y):
                return all(compare_trees(x[i], y[i]) for i in range(len(x)))
            else:
                return False
        elif type(x) in [bool, int, str]:
            return x == y
        else:
            assert False
    assert False

def read_tree(s: str) -> Any:
    ss = s.replace("(", " ( ").replace(")", " ) ")
    tokens = ss.split(" ")
    tokens = [x for x in tokens if x != ""]
    def help(xs):
        if xs[0] == "(":
            # call manytimes for xs[1:]
            xs = xs[1:]
            temp = []
            while True:
                if xs[0] == ")":
                    return temp, xs[1:]
                else:
                    item, xs = help(xs)
                    temp.append(item)            
        else:
            return xs[0], xs[1:]
    res, _ = help(tokens)
    return res



T = TypeVar("T")
def argmax(xs: Iterable[T], f: Callable[[T], int]) -> Tuple[T, int]:
    currentmax = None
    currentmaxval = None
    for x in xs:
        val = f(x)
        if currentmaxval is None or currentmaxval < val:
            currentmax = x
            currentmaxval = val
    if currentmax is None or currentmaxval is None:
        raise Exception("Empty")
    else:
        return currentmax, currentmaxval


def lcm(a: int, b: int) -> int:
    return a * b // math.gcd(a, b)


def lcm_many(xs: List[int]) -> int:
    if len(xs) == 0:
        return 1
    elif len(xs) == 1:
        return abs(xs[0])
    else:
        return functools.reduce(lcm, xs)


def div_ceiling(a: int, b: int) -> int:
    assert a >= 0 and b >= 0
    if a % b == 0:
        return a // b
    else:
        return a // b + 1


Tree = Any


class SMTUtilOption():
    def __init__(self, floor_reduce, expand_floor_in_inteprolation_error, disable_euf, lia2bv):
        self.floor_reduce = floor_reduce
        self.expand_floor_in_inteprolation_error = expand_floor_in_inteprolation_error
        self.disable_euf = disable_euf
        self.lia2bv = lia2bv # boxing or naive
    def to_dict(self):
        return {"floor_reduce": self.floor_reduce,
        "expand_floor_in_interpolation_error": self.expand_floor_in_inteprolation_error,
        "disable_euf": self.disable_euf,
        "lia2bv": self.lia2bv}


def get_tag(x) -> str:
    return x


def get_coefs_fraction(tree: Tree, lets: Dict[str, Any]) -> Tuple[Dict[str, Fraction], Fraction]:
    def is_const(coefs: Dict[str, Fraction], const: Fraction) -> bool:
        for _, n in coefs.items():
            if n != 0:
                return False
        return True

    if type(tree) is str:
        if tree.lstrip("+-").isnumeric():  # type: ignore
            return get_coefs_fraction(Fraction(int(tree)), lets)  # type: ignore
        elif tree.startswith("var_"):
            return {tree: 1}, 0
        else:
            # division?
            sp = tree.split("/")
            if len(sp) == 2 and sp[0].lstrip("+-").isnumeric() and sp[1].isnumeric():
                return get_coefs_fraction(["/", sp[0], sp[1]], lets)
            assert False
    elif type(tree) is int:
        return {}, Fraction(tree)
    elif type(tree) is Fraction:
        return {}, tree
    elif type(tree) is list:
        tag = get_tag(tree[0])
        if tag == "*":
            temp_coefs: Dict[str, Fraction] = {}
            temp_const = Fraction(1)
            for i in tree[1:]:
                coefs, const = get_coefs_fraction(i, lets)
                if is_const(temp_coefs, temp_const):
                    val = temp_const
                    mul_coefs, mul_const = coefs, const
                elif is_const(coefs, const):
                    val = const
                    mul_coefs, mul_const = temp_coefs, temp_const
                else:
                    raise Exception("Nonlinear...")
                temp_coefs, temp_const = {k: val * v for k, v in mul_coefs.items()}, mul_const * val
            return temp_coefs, temp_const
        elif tag == "+":
            temp_coefs = {}
            temp_const = Fraction(0)
            for i in tree[1:]:
                coefs, const = get_coefs_fraction(i, lets)
                for k, v in coefs.items():
                    temp_coefs[k] = temp_coefs.get(k, 0) + v
                temp_const += const
            return temp_coefs, temp_const
        elif tag == "-":
            if len(tree) == 2:
                coefs, const = get_coefs_fraction(tree[1], lets)
                return {k: -v for k, v in coefs.items()}, -const
            elif len(tree) == 3:
                coefs_left, const_left = get_coefs_fraction(tree[1], lets)
                coefs_right, const_right = get_coefs_fraction(tree[2], lets)
                for k, v in coefs_right.items():
                    coefs_left[k] = coefs_left.get(k, 0) - v
                const_left -= const_right
                return coefs_left, const_left
            else:
                assert False
        elif tag == "/":
            if len(tree) == 3:
                coefs_left, const_left = get_coefs_fraction(tree[1], lets)
                coefs_right, const_right = get_coefs_fraction(tree[2], lets)
                if is_const(coefs_right, const_right):
                    return {k: v / const_right for k, v in coefs_left.items()}, const_left / const_right
                else:
                    raise Exception("righthand side is not constanta")
            else:
                assert False
        elif tag == "to_real":
            return get_coefs_fraction(tree[1], lets)
        else:
            raise Exception(f"Unexpected symbol {tag}")
    assert False


def convert_int2list(n: int) -> Any:
    if n >= 0:
        return n
    else:
        return ["-", -n]


# def find_subtrees_by_tags(tags: List[str], tree: Any, parent: Any = None, id: int = -1) -> List[Tuple[Any, Any, int]]:
#     if type(tree) is list:
#         res = []
#         if type(tree[0]) is str and get_tag(tree[0]) in tags:
#             return [(tree, parent, id)]
#         else:
#             res: List[Tuple[Any, Any, int]] = []
#             for i in range(len(tree)):
#                 res += find_subtrees_by_tags(tags, tree[i], tree, i)
#             return res
#     return []

def find_subtrees_by_tags(tags: List[str], tree: Any, parent: Any = None, id: int = -1) -> List[Tuple[Any, Any, int]]:
    def help(tags: List[str], tree: Any, parent: Any = None, id: int = -1) -> List[Tuple[Any, Any, int]]:
        if type(tree) is list:
            res = []
            if type(tree[0]) is str and get_tag(tree[0]) in tags:
                res.append((tree, parent, id))
            for i in range(len(tree)):
                res += help(tags, tree[i], tree, i)
            return res
        return []
    res = help(tags, tree, parent, id)
    res.reverse()
    return res


def get_lets(tree: Tree) -> Dict[str, Any]:
    if type(tree) is list:
        res: Dict[str, Any] = {}
        if type(tree[0]) is str and get_tag(tree[0]) == "let":
            assert len(tree) == 3
            for d in tree[1]:
                res[get_tag(d[0])] = d[1]
        for t in tree:
            res.update(get_lets(t))
        return res
    else:
        return {}



re_hex = re.compile(r"^(\+|\-)?0x(\d|[a-f])+$")

def count_terms(tree: Tree) -> int:
    if type(tree) is not list:
        return 1
    tag = get_tag(tree[0])
    if tag in ["and", "or", "not", "=>"]:
        return sum([count_terms(i) for i in tree[1:]])
    elif tag in ["=", "<=", ">=", ">", "<", "_", "true", "false", True, False] or tag.startswith("bv"):
        return 1
    else:
        assert False


def pick_minimum_priority(ss: Set[T], f: Callable[[T], int]):
    current = ss.pop()
    current_p = f(current)
    for x in ss:
        p = f(x)
        if p < current_p:
            current = x
    return current