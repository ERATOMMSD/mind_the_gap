import smtutil #check_sat_lia
import naive_lia2bv #naive_lia2bv_normalized_ineq
import boxing #box_normalized_ineq(x_c, b, m)
import treeutil # get_range_constraints(ss, m), get_variables(t)
import itertools
import statistics
import random

chars = "abcdefghijklmnopqrstuvwxyz"


def check(x_c, b, m):
    print(x_c, b, m)
    naive = naive_lia2bv.naive_lia2bv_normalized_ineq(x_c, b, m)
    boxed = boxing.box_normalized_ineq(x_c, b, m)
    n2b = ["=>", naive, boxed]
    b2n = ["=>", boxed, naive]
    iff = ["and", n2b, b2n]
    vs = treeutil.get_variables(iff)
    cs = treeutil.get_range_constraints(vs, m)
    with_c = ["and"] + [iff] + [cs]
    # print(cs)
    # print(naive)
    # print(boxed)
    return smtutil.get_model_lia([["and", ["not", iff], cs]], statistics.Statistics())


def make_random_setting(max_d, max_b, widths):
    d = random.choice(range(1, max_d + 1))
    width = random.choice(widths)
    m = 2**width
    coef_max = m
    x_c = {treeutil.decorate(chars[i], 0): random.randint(1, coef_max) for i in range(d)}
    b = random.randint(0, max_b)
    return x_c, b, m


for i in range(100):
    x_c, b, m = make_random_setting(3, 300, [8])
    res = check(x_c, b, m)
    print(res)
    if len(res) > 0:
        break

# print(check({"var_0_a": 3, "var_0_b": 1}, 7, 8))