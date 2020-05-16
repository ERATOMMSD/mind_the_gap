import copy
import functools
import logging
import math
from typing import Tuple, Dict, Callable, List, Any, Union

from boxing import box_all_ineqs_in_tree
from bv2lia import reduce_float_from_tree
from mathsat import *
import naive_lia2bv
from util import Tree, SMTUtilOption, get_tag, re_hex
import itertools
import re
import functools
import statistics


def is_var(x: str) -> bool:
    if type(x) is not str:
        return False
    return x.startswith("var_")


def decorate(x: str, n: int) -> str:
    return f"var_{n}_{x}"


def undecorate(x: str) -> Tuple[str, int]:
    if is_var(x):
        y = x.split("_")
        return ("_".join(y[2:]), int(y[1]))
    return x, -1


def shift_index(x: str, n: int) -> str:
    if is_var(x):
        body, index = undecorate(x)
        return decorate(body, index + n)
    return x


def set_index(x: str, n: int) -> str:
    if is_var(x):
        body, index = undecorate(x)
        return decorate(body, n)
    return x


def shift_index_tree_dict_immutable(x: Tree, d: Dict[str, int]) -> Tree:
    return walk_replace_immutable(x, lambda x: shift_index(x, d.get(undecorate(x)[0], 0)))


def walk_replace_immutable(tree: Tree, f: Callable[[str], str]) -> Tree:
    if type(tree) is list:
        return [walk_replace_immutable(i, f) for i in tree]
    else:
        if type(tree) is str:
            return f(tree)
        else:
            return tree


def get_variables(t: Tree) -> List[str]:
    def help(tree: Any) -> List[str]:
        res: List[Any] = []
        if type(tree) is list:
            for i in range(len(tree)):
                if type(tree[i]) is list:
                    res += help(tree[i])
                elif type(tree[i]) is str:
                    if is_var(tree[i]):
                        res.append(tree[i])
        return res

    return help(t)


def get_range_constraints(ss: List[str], m: int) -> Any:
    bits = int(math.log2(m))
    if len(ss) == 0:
        return "true"
    consts: List[Any] = []
    for s in set(ss):
        consts.append(["<=", 0, s])
        consts.append(["<=", s, m - 1])
    return ["and"] + consts


def set_index_tree_immutable(t: Tree, n: int) -> Any:
    return walk_replace_immutable(t, lambda x: set_index(x, n))


def is_def_in_let(x: str) -> bool:
    return x.startswith(".def_")



def replace_symbols_for_bv(t: Tree, m: int, signed: str, var2type: Dict[str, str]) -> Any:
    bitsize = int(math.log2(m))
    dic = {"+": "bvadd", "*": "bvmul", "div": "bvdiv", "mod": "bvurem",
           "and": "and", "or": "or", "not": "not", "=>": "=>", "=": "=", "true": "true", "false": "false"}
    ineq_signed = {"<=": "bvsle", ">=": "bvsge", "<": "bvslt", ">": "bvsgt"}
    ineq_unsigned = {"<=": "bvule", ">=": "bvuge", "<": "bvult", ">": "bvugt"}

    def align_int(typeleft: str, typeright: str) -> str:
        integers = ["int", "unsigned"]
        if typeleft == "dummy" or typeright == "dummy":
            return "dummy"
        elif typeleft == "int" and typeright == "int":
            return "int"
        elif typeleft in integers and typeright in integers:
            return "unsigned"
        else:
            assert False
    def help(tree: Any, def2types: Dict[str, str]) -> Tuple[Any, str]:
        if type(tree) is int:
            if tree < 0:
                sub, subtype = help(-tree, def2types)
                res_term, res_type = ["bvneg", sub], "int"
            else:
                res_term, res_type = ["_", f"bv{tree % m}", bitsize], "int"
        elif type(tree) is str:
            tag = get_tag(tree)
            assert tag != "dummy"
            if is_var(tag):
                undectag = undecorate(tag)[0]
                typ = var2type[undectag] if signed == "depends" else "dummy"
                res_term, res_type =  tag, typ
            elif is_def_in_let(tag):
                res_term, res_type =  tag, def2types[tag]
            elif tag in ["true", "false"]:
                res_term, res_type =  tag, "bool"
            elif tag in ["*nondetint"]:
                res_term, res_type =  tag, "int"
            elif tag in ["*nondetunsigned"]:
                res_term, res_type =  tag, "unsigned"
            elif tag.lstrip("+-").isnumeric():
                res_term, res_type =  help(int(tag), def2types)[0], "int"
            elif re_hex.match(tag.lstrip("+-")) is not None:
                res_term, res_type =  help(int(tag, 16), def2types)[0], "int"
            else:
                assert False
        elif type(tree) is bool:
            res_term, res_type =  tree, "bool"
        elif type(tree) is list:
            if type(tree[0]) is str:
                tag = get_tag(tree[0])
                if tag == "-":
                    if len(tree) == 2:
                        sub, subtype = help(tree[1], def2types)
                        res_term, res_type =  ["bvneg", sub], subtype
                    elif len(tree) == 3:
                        sub1, subtype1 = help(tree[1], def2types)
                        sub2, subtype2 = help(tree[2], def2types)
                        res_term, res_type =  ["bvadd", sub1, ["bvneg", sub2]], align_int(subtype1, subtype2)
                    else:
                        assert False
                elif tag in ["%", "mod", "div"]:
                    assert len(tree) == 3
                    left, lefttype = help(tree[1], def2types)
                    right, righttype = help(tree[2], def2types)
                    if right[1] == "bv0":
                        if dic[tag] == "bvdiv":
                            res_term, res_type = help(0, def2types)[0], lefttype
                        elif dic[tag] == "bvurem":
                            res_term, res_type = left, lefttype
                        else:
                            assert False
                    else:
                        res_term, res_type =  [dic[tag], left, right], align_int(lefttype, righttype)
                    # if int(right[1][2:]) >= m:
                    #     res_term, res_type =  left
                    # else:
                    #     res_term, res_type =  ["bvurem", left, right]
                elif tag in ["*", "+"]:
                    subtrees_and_types = [help(i, def2types) for i in tree[1:]]
                    subtrees = [i[0] for i in subtrees_and_types]
                    subtypes = [i[1] for i in subtrees_and_types]
                    subtype = functools.reduce(align_int, subtypes)
                    res_term, res_type =  [dic[tag]] + subtrees, subtype
                elif tag in ["and", "or"]:
                    subtrees_and_types = [help(i, def2types) for i in tree[1:]]
                    subtrees = [i[0] for i in subtrees_and_types]
                    subtypes = [i[1] for i in subtrees_and_types]
                    assert all(i == "bool" for i in subtypes)
                    res_term, res_type =  [tag] + subtrees, "bool"
                elif tag in ["ite"]:
                    assert len(tree) == 4
                    subtrees_and_types = [help(i, def2types) for i in tree[1:]]
                    subtrees = [i[0] for i in subtrees_and_types]
                    subtypes = [i[1] for i in subtrees_and_types]
                    assert subtypes[0] == "bool"
                    if subtypes[1] == "bool" and subtypes[2] == "bool":
                        subtype = "bool"
                    else:
                        subtype = align_int(subtypes[1], subtypes[2])
                    res_term, res_type = [tag] + subtrees, subtype
                elif tag in ["not"]:
                    assert len(tree) == 2
                    left, lefttype = help(tree[1], def2types)
                    assert lefttype == "bool"
                    res_term, res_type =  ["not", left], "bool"
                elif tag in ["=>"]:
                    assert len(tree) == 3
                    left, lefttype = help(tree[1], def2types)
                    right, righttype = help(tree[2], def2types)
                    assert lefttype == "bool" and righttype == "bool"
                    res_term, res_type =  ["=>", left, right], "bool"
                elif tag in ineq_signed:
                    # equivalent to tag in ineq_unsigned
                    sub1, subtype1 = help(tree[1], def2types)
                    sub2, subtype2 = help(tree[2], def2types)
                    subtype = align_int(subtype1, subtype2)
                    if signed == "signed":
                        applytype = "int"
                    elif signed == "unsigned":
                        applytype = "unsigned"
                    elif signed == "depends":
                        applytype = subtype
                    else:
                        assert False
                    if applytype == "int":
                        res_term, res_type =  [ineq_signed[tag], sub1, sub2], "bool"
                    elif applytype == "unsigned":
                        res_term, res_type =  [ineq_unsigned[tag], sub1, sub2], "bool"
                    else:
                        assert False
                elif tag == "=":
                    subtrees_and_types = [help(i, def2types) for i in tree[1:]]
                    subtrees = [i[0] for i in subtrees_and_types]
                    subtypes = [i[1] for i in subtrees_and_types]
                    subtypes = ["integer" for i in subtypes if i in ["int", "unsigned"]]
                    aligned = len(set(subtypes)) <= 1
                    assert aligned
                    res_term, res_type =  ["="] + subtrees, "bool"
                elif tag == "let":
                    assert len(tree) == 3
                    defs = tree[1]
                    content = tree[2]
                    defs_replaced_both = [[v, help(w, def2types)] for v, w in defs]
                    defs_replaced = [[v, w[0]] for v, w in defs_replaced_both]
                    defs_replaced_type = [[v, w[1]] for v, w in defs_replaced_both]
                    def2types.update(dict(defs_replaced_type))
                    content_replaced, subtype = help(content, def2types)
                    res_term, res_type =  ["let", defs_replaced, content_replaced], subtype
                else:
                    raise Exception(f"Unexpected symbol {tag}")
            elif type(tree[0][0]) == str:
                tag = get_tag(tree[0][0])
                if tag == "_":
                    tag = get_tag(tree[0][1])
                    if tag == "divisible":
                        divisor = int(tree[0][2])
                        assert divisor > 0
                        left, lefttype = help(tree[1], def2types)
                        assert lefttype in ["int", "unsigned"]
                        if divisor >= m:
                            right, righttype = help(0, def2types)
                            res_term, res_type =  ["=", left, right], "bool"
                        else:
                            right, righttype = help(tree[2], def2types)
                            assert righttype in ["int", "unsigned"]
                            res_term, res_type =  [["_", "divisible", right], left], "bool"
                    else:
                        assert False
                else:
                    assert False
        else:
            assert False

        return res_term, res_type
    res = help(t, {})
    return res[0]


def substitute_tree(t: Tree, d: Dict[str, Tree]) -> Tree:
    def f(x):
        if x.startswith("var_strategy2_"):
            return copy.deepcopy(d[x])
        else:
            return x

    return walk_replace_immutable(t, f)


def reduce_red_mod(t, m):
    def help(t):
        if type(t) is list:
            tag = get_tag(t[0])
            if tag == "mod":
                assert len(t) == 3
                if int(t[2]) == m:
                    return t[1]
            else:
                return [help(x) for x in t]
        else:
            return t
    return help(t)


def process_overflow(t: Tree, m: int, opt: SMTUtilOption, stat: statistics.Statistics) -> Any:
    vs = get_variables(t)
    d = {k: (0, m - 1) for k in vs}
    reduced_red_mod = tree_mod_to_floor(t) # can be dangerous
    reduced, replacing_floor = reduce_float_from_tree(reduced_red_mod, d, m, opt)
    if opt.lia2bv == "boxing":
        boxed = box_all_ineqs_in_tree(reduced, m, stat)
    elif opt.lia2bv == "naive":
        boxed = naive_lia2bv.naive_lia2bv_all_ineqs_in_tree(reduced, m)
    else:
        assert False
    processed_replacing_floor = substitute_tree(boxed, replacing_floor)
    return processed_replacing_floor

def convert_lia2bv(t: Tree, m: int, opt: SMTUtilOption, stat: statistics.Statistics) -> Any:
    # vs = get_variables(t)
    # d = {k: (0, m - 1) for k in vs}
    # reduced, replacing_floor = reduce_float_from_tree(t, d, m, opt)
    # if opt.lia2bv == "boxing":
    #     boxed = box_all_ineqs_in_tree(reduced, m)
    # elif opt.lia2bv == "naive":
    #     boxed = naive_lia2bv.naive_lia2bv_all_ineqs_in_tree(reduced, m)
    # else:
    #     assert False
    # processed_replacing_floor = substitute_tree(boxed, replacing_floor)
    processed_replacing_floor = process_overflow(t, m, opt, stat)
    replaced = replace_symbols_for_bv(processed_replacing_floor, m, "unsigned", {})
    return replaced


def convert_bv2lia(t: Tree, m: int) -> Any:
    """embedding part"""

    def cover_mod(x) -> Any:
        return ["mod", x, m]

    def help(tree: Any) -> Any:
        if type(tree) is list:
            tag = tree[0]
            if tag in ["bvadd", "bvdiv", "bvmul"]:
                d = {"bvadd": "+", "bvdiv": "div", "bvmul": "*"}
                return cover_mod([d[tag]] + [help(i) for i in tree[1:]])
            elif tag in ["bvsle", "bvslt", "bvsgt", "bvsge"]:
                tree1 = help(tree[1])
                tree2 = help(tree[2])

                def case_sle(left, right):
                    cond = ["or", ["and", ["<=", left, m // 2 - 1], ["<=", right, m // 2 - 1]],
                            ["and", [">=", left, m // 2], [">=", right, m // 2]]]
                    t = ["<=", left, right]
                    f = [">=", left, m // 2]
                    return ["ite", cond, t, f]

                if tag == "bvsle":
                    return case_sle(tree1, tree2)
                elif tag == "bvslt":
                    return ["not", case_sle(tree2, tree1)]
                elif tag == "bvsgt":
                    return ["not", case_sle(tree1, tree2)]
                elif tag == "bvsge":
                    return case_sle(tree2, tree1)
                else:
                    assert False
            elif tag in ["bvule", "bvult", "bvuge", "bvugt"]:
                left = help(tree[1])
                right = help(tree[2])

                def case_ule(left, right):
                    return ["<=", left, right]

                if tag == "bvule":
                    return case_ule(left, right)
                elif tag == "bvult":
                    return ["not", case_ule(right, left)]
                elif tag == "bvuge":
                    return case_ule(right, left)
                elif tag == "bvugt":
                    return ["not", case_ule(left, right)]
                else:
                    assert False
            elif tag in ["=", "=>", "and", "or", "not"]:
                return [tag] + [help(i) for i in tree[1:]]
            elif tag in ["bvneg"]:
                return cover_mod(["-", help(tree[1])])
            elif tag == "_":
                return int(get_tag(tree[1])[2:])
            elif tag == "let":
                assert len(tree) == 3
                defs = tree[1]
                content = tree[2]
                defs_replaced = [[v, help(w)] for v, w in defs]
                content_replaced = help(content)
                return ["let", defs_replaced, content_replaced]
            elif tag == "bvurem":
                return ["mod", help(tree[1]), help(tree[2])]
            else:
                assert False
        elif type(tree) is str:
            tag = get_tag(tree)
            if tag in ["true", "false"]:
                return tag
            elif is_var(tag) or is_def_in_let(tag):
                return tag
            elif tag.isnumeric():
                return help(int(tag))
            assert False
        elif type(tree) is int:
            assert 0 <= tree and tree < m
            return tree
        elif type(tree) is bool:
            return "true" if tree else "false"
        else:
            assert False
    res = help(t)
    
    return help(t)


def mod_to_floor(fml: Tree) -> Tree:
    assert len(fml) == 3
    n = int(fml[2])
    assert n > 0
    return ["-", fml[1], ["*", n, ["floor", ["/", fml[1], n]]]]

def tree_mod_to_floor(fml: Tree) -> Tree:
    def help(t):
        if type(t) is list:
            tag = t[0]
            if tag == "mod":
                return [help(i) for i in mod_to_floor(t)]
            else:
                return [help(i) for i in t]
        else:
            return t
    return help(fml)



def convert_tree_to_capi_lia_formula(t: Tree, env) -> Tuple[Any, List[str]]:
    logging.debug(f"given: {t}")
    var_dict: Dict[str, Any] = {}
    int_tp = msat_get_integer_type(env)

    def help(fml):
        # print(fml)
        if type(fml) is list:
            tag = fml[0]
            if tag in ["<=", ">=", "<", ">"]:
                assert len(fml) == 3
                left = help(fml[1])
                assert not MSAT_ERROR_TERM(left)
                right = help(fml[2])
                assert not MSAT_ERROR_TERM(right)
                if tag == "<=":
                    return msat_make_leq(env, left, right)
                elif tag == ">=":
                    return msat_make_leq(env, right, left)
                elif tag == "<":
                    x = ["not", ["<=", fml[2], fml[1]]]
                    return help(x)
                    # right_minus_one = help(["-", fml[2], 1])
                    # return msat_make_leq(env, left, right_minus_one)
                elif tag == ">":
                    x = ["not", ["<=", fml[1], fml[2]]]
                    return help(x)
                    # left_minus_one = help(["-", fml[1], 1])
                    # return msat_make_leq(env, right, left_minus_one)
                else:
                    assert False
            elif tag == "=":
                assert len(fml) == 3
                left = help(fml[1])
                assert not MSAT_ERROR_TERM(left)
                right = help(fml[2])
                assert not MSAT_ERROR_TERM(right)
                return msat_make_equal(env, left, right)
            elif tag == "!=":
                assert len(fml) == 3
                return help(["not", ["="] + fml[1:]])
            elif tag == "+":
                assert len(fml) >= 2
                if len(fml) == 2:
                    it = help(fml[1])
                    assert not MSAT_ERROR_TERM(it)
                    return it
                else:
                    left = help(fml[1])
                    assert not MSAT_ERROR_TERM(left)
                    right = help(["+"] + fml[2:])
                    assert not MSAT_ERROR_TERM(right)
                    return msat_make_plus(env, left, right)
            elif tag == "-":
                if len(fml) == 2:
                    return help(["*", -1, fml[1]])
                elif len(fml) == 3:
                    return help(["+", fml[1], ["-", fml[2]]])  # TODO
                else:
                    assert False
            elif tag == "*":
                assert len(fml) >= 2
                if len(fml) == 2:
                    it = help(fml[1])
                    assert not MSAT_ERROR_TERM(it)
                    return it
                else:
                    left = help(fml[1])
                    assert not MSAT_ERROR_TERM(left)
                    right = help(["*"] + fml[2:])
                    assert not MSAT_ERROR_TERM(right)
                    return msat_make_times(env, left, right)
            elif tag in ["/", "div"]:
                assert len(fml) == 3
                left = help(fml[1])
                assert not MSAT_ERROR_TERM(left)
                right = help(fml[2])
                assert not MSAT_ERROR_TERM(right)
                # WARNING: THIS IS JUST A RATIONAL
                divided = msat_make_divide(env, left, right) 
                return msat_make_floor(env, divided)
            elif tag in ["mod", "%"]:
                return help(mod_to_floor(fml))
            elif tag == "and":
                assert len(fml) >= 2
                if len(fml) == 1:
                    return msat_make_true(env)
                elif len(fml) == 2:
                    it = help(fml[1])
                    assert not MSAT_ERROR_TERM(it)
                    return it
                else:
                    left = help(fml[1])
                    assert not MSAT_ERROR_TERM(left)
                    right = help(["and"] + fml[2:])
                    assert not MSAT_ERROR_TERM(right)
                    return msat_make_and(env, left, right)
            elif tag == "or":
                assert len(fml) >= 2
                if len(fml) == 1:
                    return msat_make_false(env)
                elif len(fml) == 2:
                    it = help(fml[1])
                    assert not MSAT_ERROR_TERM(it)
                    return it
                else:
                    left = help(fml[1])
                    assert not MSAT_ERROR_TERM(left)
                    right = help(["or"] + fml[2:])
                    assert not MSAT_ERROR_TERM(right)
                    return msat_make_or(env, left, right)
            elif tag == "not":
                assert len(fml) == 2
                it = help(fml[1])
                assert not MSAT_ERROR_TERM(it)
                return msat_make_not(env, it)
            elif tag == "ite":
                assert len(fml) == 4
                return help(["and", ["=>", fml[1], fml[2]], ["=>", ["not", fml[1]], fml[3]]])
                # cond = help(fml[1])
                # assert not MSAT_ERROR_TERM(cond)
                # print(f"cond type is {msat_type_repr(msat_term_get_type(cond))}")
                # print(f"cond is {msat_term_repr(cond)}")
                # tt = help(fml[2])
                # print(f"tt type is {msat_type_repr(msat_term_get_type(tt))}")
                # print(f"tt is {msat_term_repr(tt)}")
                # assert not MSAT_ERROR_TERM(tt)
                # ff = help(fml[3])
                # print(f"ff type is {msat_type_repr(msat_term_get_type(ff))}")
                # print(f"ff is {msat_term_repr(ff)}")
                # assert not MSAT_ERROR_TERM(ff)
                # res = msat_make_term_ite(env, cond, tt, ff)
                # assert not MSAT_ERROR_TERM(res)
                # return
            elif tag == "=>":
                assert len(fml) == 3
                return help(["or", ["not", fml[1]], fml[2]])
            elif tag == "floor":
                assert len(fml) == 2
                arg = help(fml[1])
                assert not MSAT_ERROR_TERM(arg)
                return msat_make_floor(env, arg)
            else:
                assert False
        elif type(fml) is str:
            if fml == "true":
                return msat_make_true(env)
            elif fml == "false":
                return msat_make_false(env)
            elif is_var(fml):
                if fml in var_dict:
                    return msat_make_constant(env, var_dict[fml])
                else:
                    var_dict[fml] = msat_declare_function(env, fml, int_tp)
                    return msat_make_constant(env, var_dict[fml])
            elif fml.isnumeric() or (fml[0] == "-" and fml[1:].isnumeric()):
                return help(int(fml))
            else:
                assert False
        elif type(fml) is int:
            return msat_make_number(env, str(fml))
        elif type(fml) is bool:
            if fml:
                return msat_make_true(env)
            else:
                return msat_make_false(env)
        else:
            assert False

    res = help(t)
    assert res is not None
    assert not MSAT_ERROR_TERM(res)
    return help(t), list(var_dict.keys()), var_dict


def evaluate_bv(t: Tree, m: int, asg: Dict[str, int]) -> Union[bool, int]:
    def to_signed(n):
        if n < m / 2:
            return n
        else:
            return n - m

    def to_unsigned(n):
        if n >= 0:
            return n
        else:
            return n + m

    def help(t: Tree) -> Union[bool, int]:
        if type(t) is int:
            return t
        elif type(t) is bool:
            return t
        elif type(t) is list:
            tag = t[0]
            if tag == "bvadd":
                return sum([help(x) for x in t[1:]]) % m
            elif tag == "bvdiv":
                assert len(t) == 3
                return help(t[1]) // help(t[2])
            elif tag == "bvmul":
                return functools.reduce(int.__mul__, [help(x) for x in t[1:]], 1) % m
            elif tag == "bvsle":
                assert len(t) == 3
                return to_signed(help(t[1])) <= to_signed(help(t[2]))
            elif tag == "bvslt":
                assert len(t) == 3
                return to_signed(help(t[1])) < to_signed(help(t[2]))
            elif tag == "bvsgt":
                assert len(t) == 3
                return to_signed(help(t[1])) > to_signed(help(t[2]))
            elif tag == "bvsge":
                assert len(t) == 3
                return to_signed(help(t[1])) >= to_signed(help(t[2]))
            elif tag == "bvule":
                assert len(t) == 3
                return help(t[1]) <= help(t[2])
            elif tag == "bvult":
                assert len(t) == 3
                return help(t[1]) < help(t[2])
            elif tag == "bvugt":
                assert len(t) == 3
                return help(t[1]) > help(t[2])
            elif tag == "bvuge":
                assert len(t) == 3
                return help(t[1]) >= help(t[2])
            elif tag == "=":
                assert len(t) == 3
                return help(t[1]) == help(t[2])
            elif tag == "=>":
                assert len(t) == 3
                return (not help(t[1])) or help(t[2])
            elif tag == "and":
                return functools.reduce(bool.__and__, [help(x) for x in t[1:]], True)
            elif tag == "or":
                return functools.reduce(bool.__or__, [help(x) for x in t[1:]], False)
            elif tag == "not":
                assert len(t) == 2
                return not help(t[1])
            elif tag == "_":
                assert len(t) == 3
                return int(t[1][2:])
            elif tag == "bvneg":
                assert len(t) == 2
                return to_unsigned(-to_signed(help(t[1])))
            elif tag == "bvurem":
                assert len(t) == 3
                left = help(t[1])
                right = help(t[2])
                if right % m == 0:
                    return left
                else:
                    return left % right
            else:
                raise Exception(f"Unexpected tag {tag}")
        elif type(t) is str:
            if is_var(t):
                return asg[t]
            elif t.lstrip("+-").isnumeric():
                return int(t)
            elif t in ["true", "True"]:
                return True
            elif t in ["false", "False"]:
                return False
            else:
                assert False
        else:
            assert False
        raise Exception(f"Unexpected type: {type(t)}")

    return help(t)


def convert_tree_to_capi_bv_formula(t: Tree, m: int, env) -> Tuple[Any, List[str]]:
    """take tree consists of <= or something and return BV formula in CAPI.  Just convert symbols"""
    var_dict: Dict[str, Any] = {}
    bitwidth: int = int(math.log2(m))
    bv_tp = msat_get_bv_type(env, bitwidth)

    def help(fml):
        # print(fml)
        if type(fml) is list:
            tag = fml[0]
            if tag in ["bvule", "bvult", "bvuge", "bvugt"]:
                assert len(fml) == 3
                left = help(fml[1])
                right = help(fml[2])
                if tag == "bvule":
                    term = msat_make_bv_uleq(env, left, right)
                    assert not MSAT_ERROR_TERM(term)
                    return term
                elif tag == "bvult":
                    term = msat_make_bv_ult(env, left, right)
                    assert not MSAT_ERROR_TERM(term)
                    return term
                elif tag == "bvuge":
                    term = msat_make_bv_uleq(env, right, left)
                    assert not MSAT_ERROR_TERM(term)
                    return term
                elif tag == "bvugt":
                    term = msat_make_bv_ult(env, right, left)
                    assert not MSAT_ERROR_TERM(term)
                    return term
                else:
                    assert False
            elif tag in ["bvsle", "bvslt", "bvsge", "bvsgt"]:
                assert len(fml) == 3
                left = help(fml[1])
                right = help(fml[2])
                if tag == "bvsle":
                    term = msat_make_bv_sleq(env, left, right)
                    assert not MSAT_ERROR_TERM(term)
                    return term
                elif tag == "bvslt":
                    term = msat_make_bv_slt(env, left, right)
                    assert not MSAT_ERROR_TERM(term)
                    return term
                elif tag == "bvsge":
                    term = msat_make_bv_sleq(env, right, left)
                    assert not MSAT_ERROR_TERM(term)
                    return term
                elif tag == "bvsgt":
                    term = msat_make_bv_slt(env, right, left)
                    assert not MSAT_ERROR_TERM(term)
                    return term
                else:
                    assert False
            elif tag == "=":
                assert len(fml) == 3
                left = help(fml[1])
                right = help(fml[2])
                term = msat_make_equal(env, left, right)
                assert not MSAT_ERROR_TERM(term)
                return term
            elif tag == "bvadd":
                assert len(fml) >= 2
                if len(fml) == 2:
                    it = help(fml[1])
                    return it
                else:
                    left = help(fml[1])
                    right = help(["bvadd"] + fml[2:])
                    term = msat_make_bv_plus(env, left, right)
                    assert not MSAT_ERROR_TERM(term)
                    return term
            elif tag == "bvneg":
                assert len(fml) == 2
                term = msat_make_bv_neg(env, help(fml[1]))
                assert not MSAT_ERROR_TERM(term)
                return term
            elif tag == "bvmul":
                assert len(fml) >= 2
                if len(fml) == 2:
                    it = help(fml[1])
                    return it
                else:
                    left = help(fml[1])
                    right = help(["bvmul"] + fml[2:])
                    term = msat_make_bv_times(env, left, right)
                    assert not MSAT_ERROR_TERM(term)
                    return term
            # elif tag == "mod":
            #     assert len(fml) == 3
            #     assert fml[2] > 0
            #     left = help(fml[1])
            #     if fml[2] > m:
            #         return left
            #     right = help(fml[2])
            #     return msat_make_bv_urem(env, left, right)
            elif tag == "and":
                assert len(fml) >= 2
                if len(fml) == 1:
                    term = msat_make_true(env)
                    assert not MSAT_ERROR_TERM(term)
                    return term
                elif len(fml) == 2:
                    it = help(fml[1])
                    return it
                else:
                    left = help(fml[1])
                    right = help(["and"] + fml[2:])
                    term = msat_make_and(env, left, right)
                    assert not MSAT_ERROR_TERM(term)
                    return term
            elif tag == "or":
                assert len(fml) >= 2
                if len(fml) == 1:
                    term = msat_make_false(env)
                    assert not MSAT_ERROR_TERM(term)
                    return term
                elif len(fml) == 2:
                    it = help(fml[1])
                    return it
                else:
                    left = help(fml[1])
                    right = help(["or"] + fml[2:])
                    term = msat_make_or(env, left, right)
                    assert not MSAT_ERROR_TERM(term)
                    return term
            elif tag == "not":
                assert len(fml) == 2
                it = help(fml[1])
                term = msat_make_not(env, it)
                assert not MSAT_ERROR_TERM(term)
                return term
            elif tag == "ite":
                assert len(fml) == 4
                return help(["and", ["=>", fml[1], fml[2]], ["=>", ["not", fml[1]], fml[3]]])
            elif tag == "=>":
                assert len(fml) == 3
                return help(["or", ["not", fml[1]], fml[2]])
            elif tag == "_":
                assert len(fml) == 3
                return help(int(fml[1][2:]))
            elif tag == "bvurem":
                assert len(fml) == 3
                left = help(fml[1])
                right = help(fml[2])
                # assert int(fml[2][1][2:]) % m != 0
                term = msat_make_bv_urem(env, left, right)
                assert not MSAT_ERROR_TERM(term)
                return term
            elif tag == "bvdiv":
                assert len(fml) == 3
                left = help(fml[1])
                right = help(fml[2])
                # assert int(fml[2][1][2:]) % m != 0
                term = msat_make_bv_udiv(env, left, right)
                assert not MSAT_ERROR_TERM(term)
                return term
            else:
                assert False
        elif type(fml) is str:
            if fml == "true":
                term = msat_make_true(env)
                assert not MSAT_ERROR_TERM(term)
                return term
            elif fml == "false":
                term = msat_make_false(env)
                assert not MSAT_ERROR_TERM(term)
                return term
            elif is_var(fml):
                if fml in var_dict:
                    term = msat_make_constant(env, var_dict[fml])
                    assert not MSAT_ERROR_TERM(term)
                    return term
                else:
                    var_dict[fml] = msat_declare_function(env, fml, bv_tp)
                    term = msat_make_constant(env, var_dict[fml])
                    assert not MSAT_ERROR_TERM(term)
                    return term
        elif type(fml) is int:
            if fml < 0:
                fml += ((-fml) / m + 1) * m
            fml = fml % m
            term = msat_make_bv_number(env, str(fml), bitwidth, 10)
            assert not MSAT_ERROR_TERM(term)
            return term
        elif type(fml) is bool:
            if fml:
                term = msat_make_true(env)
                assert not MSAT_ERROR_TERM(term)
                return term
            else:
                term = msat_make_false(env)
                assert not MSAT_ERROR_TERM(term)
                return term
        else:
            assert False

    res = help(t)
    assert res is not None
    return res, list(var_dict.keys())


def convert_capi_formula_to_tree(fml, env) -> Any:
    def help(fml):
        ts = str(msat_term_repr(fml))
        # print(ts)
        if msat_term_is_number(env, fml) != 0:
            x = msat_term_to_number(env, fml)
            return ts
        elif msat_term_is_constant(env, fml) != 0:
            return ts
        else:
            d = msat_term_get_decl(fml)
            # if not MSAT_ERROR_DECL(d):
            #     raise Exception(f"Unexpected decl {msat_term_repr(fml)}")
            tag = msat_decl_get_tag(env, d)
            if tag == MSAT_TAG_TRUE:
                return True
            elif tag == MSAT_TAG_FALSE:
                return False
            elif tag == MSAT_TAG_AND:
                t1 = help(msat_term_get_arg(fml, 0))
                t2 = help(msat_term_get_arg(fml, 1))
                return ["and", t1, t2]
            elif tag == MSAT_TAG_OR:
                t1 = help(msat_term_get_arg(fml, 0))
                t2 = help(msat_term_get_arg(fml, 1))
                return ["or", t1, t2]
            elif tag == MSAT_TAG_NOT:
                t1 = help(msat_term_get_arg(fml, 0))
                return ["not", t1]
            elif tag == MSAT_TAG_PLUS:
                t1 = help(msat_term_get_arg(fml, 0))
                t2 = help(msat_term_get_arg(fml, 1))
                return ["+", t1, t2]
            elif tag == MSAT_TAG_TIMES:
                t1 = help(msat_term_get_arg(fml, 0))
                t2 = help(msat_term_get_arg(fml, 1))
                return ["*", t1, t2]
            elif tag == MSAT_TAG_DIVIDE:
                t1 = help(msat_term_get_arg(fml, 0))
                t2 = help(msat_term_get_arg(fml, 1))
                return ["/", t1, t2]
            elif tag == MSAT_TAG_FLOOR:
                t1 = help(msat_term_get_arg(fml, 0))
                return ["floor", t1]
            elif tag == MSAT_TAG_LEQ:
                t1 = help(msat_term_get_arg(fml, 0))
                t2 = help(msat_term_get_arg(fml, 1))
                return ["<=", t1, t2]
            elif tag == MSAT_TAG_EQ:
                t1 = help(msat_term_get_arg(fml, 0))
                t2 = help(msat_term_get_arg(fml, 1))
                return ["=", t1, t2]
            elif tag == MSAT_TAG_INT_MOD_CONGR:
                m = msat_term_is_int_modular_congruence(env, fml)[1]
                t1 = help(msat_term_get_arg(fml, 0))
                t2 = help(msat_term_get_arg(fml, 1))
                return ["=", ["mod", ["-", t1, t2], m], 0]
            else:
                raise Exception(f"Unexpected tag {tag}/{msat_term_repr(fml)}")

    return help(fml)

def get_atomic_formulae(t: Tree) -> List[Tree]:
    def help(t: Tree, stock: List[Tree]):
        if type(t) is list:
            tag = get_tag(t[0])
            if tag in ["and", "or", "not"]:
                for st in t[1:]:
                    help(st, stock)
            else:
                stock.append(t)
        else:
            stock.append(t)
    stock = []
    help(t, stock)
    return stock


if __name__ == "__main__":
    x = ['not', ['ite', ['or', ['and', ['<=', 1, 127], ['<=', ['mod', ['+', 'var_2_j', 1], 256], 127]], ['and', ['>=', 1, 128], ['>=', ['mod', ['+', 'var_2_j', 1], 256], 128]]], ['<=', 1, ['mod', ['+', 'var_2_j', 1], 256]], ['>=', 1, 128]]]
    print(reduce_red_mod(x, 256))