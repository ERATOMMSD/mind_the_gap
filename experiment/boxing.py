from typing import List, Dict, Any, Callable, TypeVar

from util import div_ceiling, Tree, get_tag, get_coefs_fraction, convert_int2list
import bv2lia
import statistics

T = TypeVar("T")

def add_list_maximal(s: List[T], e: T, f_sup: Callable[[T, T], int]) -> None:
    """f_sup(x, y) returns +1 if x is superior, -1 if y is superior, or 0 otherwise"""
    drops: List[T] = []
    for i in s:
        res = f_sup(i, e)
        if res > 0:
            return  # if there is already superior element in s, do nothing
        elif res < 0:
            drops.append(i)
    for d in drops:
        s.remove(d)
    s.append(e)


def compare_dict(x: Dict[str, int], y: Dict[str, int]) -> int:
    n = 0
    xp = 0
    yp = 0
    for i in x.keys():
        n += 1
        if x[i] > y[i]:
            xp += 1
        if x[i] < y[i]:
            yp += 1
    if xp > 0 and yp == 0:
        return 1
    elif yp > 0 and xp == 0:
        return -1
    return 0


def i_d(d: List[str], n: int) -> List[Dict[str, int]]:
    if n <= 0:
        return []
    if len(d) == 0:
        assert False
    elif len(d) == 1:
        return [{d[0]: n}]
    else:
        res: List[Dict[str, int]] = []
        head = d[0]
        tail = d[1:]
        for i in range(1, n + 1):
            for j in i_d(tail, n - i):
                j.update({head: i})
                res.append(j)
        return res


def box_d_prime(x_c: Dict[str, int], b: int, m: int) -> Any:
    corners: List[Dict[str, int]] = []
    d = len(x_c)
    assert d >= 2
    l = b // (m // 2) + 1
    for pvec in i_d(list(x_c.keys()), ((d - 1) * (l + 1))):
        corner = {k: min(div_ceiling(pvec[k] * m // 2, x_c[k] * (d - 1)) - 1, m) for k in x_c.keys()}
        add_list_maximal(corners, corner, compare_dict)
    corners2: List[Dict[str, int]] = []
    for corner in corners:
        appending = {k: v for k, v in corner.items() if v < m}
        if len(appending) > 0:
            corners2.append(appending)
    if len(corners2) == 0:
        return "true"
    corners_tree = [["and"] + [["<=", k, v] for k, v in c.items()] for c in corners2]  # type: ignore
    tree = ["or"] + corners_tree  # type: ignore
    return tree


# def box_normalized_ineq_old(x_c: Dict[str, int], b: int, m: int) -> Any:
#     assert b >= 0
#     if len(x_c) == 1:
#         k = list(x_c)[0]
#         r = b // x_c[k]
#         if r < m - 1:
#             return ["<=", k, r]
#         else:
#             return "true"
#     elif len(x_c) == 0:
#         return "true"
#     assert len(x_c) >= 2
#     assert any([v > 0 for k, v in x_c.items()])
#     s = b // (m // 2)

#     def phi0() -> Any:
#         main = ["<=",  # type: ignore
#                 ["+"] + [["*", convert_int2list(c), x] for x, c in x_c.items()] + [convert_int2list(-(s - 2) * m // 2)],
#                 # type: ignore
#                 convert_int2list(m // 2 - 1)]  # type: ignore
#         box = box_d_prime(x_c, convert_int2list((s - 1) * m // 2 - 1), m)
#         return ["and", main, box]

#     def phi1() -> Any:
#         main = ["<=",  # type: ignore
#                 ["+"] + [["*", convert_int2list(c), x] for x, c in x_c.items()] + [convert_int2list(-(s - 1) * m // 2)],
#                 # type: ignore
#                 convert_int2list(m // 2 - 1)]  # type: ignore
#         box = box_d_prime(x_c, convert_int2list(s * m // 2 - 1), m)
#         return ["and", main, box]

#     def phi2() -> Any:
#         main = ["<=",  # type: ignore
#                 ["+"] + [["*", convert_int2list(c), x] for x, c in x_c.items()] + [convert_int2list(-s * m // 2)],
#                 # type: ignore
#                 convert_int2list(b % (m // 2))]  # type: ignore
#         box = box_d_prime(x_c, b, m)
#         return ["and", main, box]

#     if s == 0:
#         return phi2()
#     elif (b % m == m // 2 - 1 and s >= 1) or s == 1:
#         return ["or", phi2(), phi1()]
#     else:
#         return ["or", phi2(), phi1(), phi0()]


def box_normalized_ineq(x_c: Dict[str, int], b: int, m: int, stat: statistics.Statistics) -> Any:
    stat.num_boxing += 1
    assert b >= 0
    if len(x_c) == 1:
        k = list(x_c)[0]
        r = b // x_c[k]
        if r < m - 1:
            return ["<=", k, r]
        else:
            return "true"
    elif len(x_c) == 0:
        return "true"
    stat.num_boxing_multi_variable += 1
    assert len(x_c) >= 2
    assert any([v > 0 for k, v in x_c.items()])
    s = b // (m // 2)

    def phi0() -> Any:
        main = ["<=",  # type: ignore
                ["+"] + [["*", convert_int2list(c), x] for x, c in x_c.items()] + [convert_int2list(-(s - 2) * m // 2)],
                # type: ignore
                convert_int2list(m // 2 - 1)]  # type: ignore
        box = box_d_prime(x_c, convert_int2list((s - 1) * m // 2 - 1), m)
        return ["and", main, box]

    def phi1() -> Any:
        main = ["<=",  # type: ignore
                ["+"] + [["*", convert_int2list(c), x] for x, c in x_c.items()] + [convert_int2list(-(s - 1) * m // 2)],
                # type: ignore
                convert_int2list(m // 2 - 1)]  # type: ignore
        box = box_d_prime(x_c, convert_int2list(s * m // 2 - 1), m)
        return ["and", main, box]

    def phi2() -> Any:
        main = ["<=",  # type: ignore
                ["+"] + [["*", convert_int2list(c), x] for x, c in x_c.items()] + [convert_int2list(-s * m // 2)],
                # type: ignore
                convert_int2list(b % (m // 2))]  # type: ignore
        box = box_d_prime(x_c, b, m)
        return ["and", main, box]

    if s == 0:
        return phi2()
    elif (b % m == m // 2 - 1 and s >= 1) or s == 1:
        return ["or", phi2(), phi1()]
    else:
        return ["or", phi2(), phi1(), phi0()]


# def box_normalized_ineq_new(x_c: Dict[str, int], b: int, m: int, stat: statistics.Statistics) -> Any:
#     assert b >= 0
#     if len(x_c) == 1:
#         k = list(x_c)[0]
#         r = b // x_c[k]
#         if r <= m - 1:
#             return ["<=", k, r]
#         else:
#             return "true"
#     elif len(x_c) == 0:
#         return "true"
#     assert len(x_c) >= 2
#     assert any([v > 0 for k, v in x_c.items()])
#     s = b // (m // 2)

#     def phi0() -> Any:
#         main = ["<=",  # type: ignore
#                 ["+"] + [["*", convert_int2list(c), x] for x, c in x_c.items()],
#                 # type: ignore
#                 convert_int2list(b)]  # type: ignore
#         box = box_d_prime(x_c, b, m)
#         return ["and", main, box]

#     def phi1() -> Any:
#         main = ["<=",  # type: ignore
#                 ["+"] + [["*", convert_int2list(c), x] for x, c in x_c.items()] + [convert_int2list(-b + (m//2 - 1))],
#                 # type: ignore
#                 convert_int2list(m//2 - 1)]  # type: ignore
#         box = box_d_prime(x_c, b, m)
#         return ["and", main, box]

#     def phi2() -> Any:
#         main = ["<=",  # type: ignore
#                 ["+"] + [["*", convert_int2list(c), x] for x, c in x_c.items()] + [convert_int2list(-b + m - 1)],
#                 # type: ignore
#                 convert_int2list(m//2 - 1)]  # type: ignore
#         box = box_d_prime(x_c, b - m//2, m)
#         return ["and", main, box]

    # if s == 0:
    #     return phi0()
    # else:
    #     return ["or", phi2(), phi1()]


def box_le_ineq(x_c: Dict[str, int], b: int, m: int, stat: statistics.Statistics) -> Any:
    return bv2lia.process_le_ineq(x_c, b, m, lambda x_c1, b1, m1: box_normalized_ineq(x_c1, b1, m1, stat))


def box_ineq(t: Tree, m: int, stat: statistics.Statistics) -> Any:
    return bv2lia.process_ineq(t, m, lambda x_c1, b1, m1: box_le_ineq(x_c1, b1, m1, stat))


def box_all_ineqs_in_tree(t: Tree, m: int, stat: statistics.Statistics) -> Any:
    return bv2lia.process_ineqs_in_tree(t, m, lambda t1, m1: box_ineq(t1, m1, stat))
