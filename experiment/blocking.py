import smtutil
import sexpdata
from typing import *
import math

import treeutil

lias = [
    # "true",
"(and (<= var_1_x 1) (<= 1 var_1_x))",
"(and (<= (+ var_1_x var_1_y) 2) (<= 2 (+ var_1_x var_1_y)))",
"(and (<= (+ var_1_x var_1_y) 2) (<= 2 (+ var_1_x var_1_y)))",
"(and (<= (+ var_1_y var_1_t1) 2) (<= 2 (+ var_1_y var_1_t1)))",
"(and (<= (+ var_1_t1 var_1_t2) 2) (<= 2 (+ var_1_t1 var_1_t2)))",
"(and (<= (+ var_1_t1 var_1_t2) 2) (<= 2 (+ var_1_t1 var_1_t2)))",
"(or (<= var_2_y (- 254)) (and (<= var_2_y 2) (<= 2 var_2_y)))",
"(or (<= var_2_y (- 254)) (and (<= var_2_y 2) (<= 2 var_2_y)))",
"(or (<= var_2_y (- 254)) (and (<= var_2_y 2) (<= 2 var_2_y)))",
"false",
"false"
]

bvs = [
    # "(or (and true))",
"(or (and (and (bvule var_1_x (_ bv1 8)) (bvule (bvadd (_ bv255 8) (bvneg var_1_x)) (_ bv254 8)))))",
"(or (and (and (and (bvule (bvadd (bvmul (_ bv1 8) var_1_x) (bvmul (_ bv1 8) var_1_y) (_ bv0 8)) (_ bv2 8)) (or (and (bvule var_1_x (_ bv127 8)) (bvule var_1_y (_ bv127 8))))) (or (and (bvule (bvadd (bvmul (_ bv1 8) (bvadd (_ bv255 8) (bvneg var_1_x))) (bvmul (_ bv1 8) (bvadd (_ bv255 8) (bvneg var_1_y))) (bvneg (_ bv128 8))) (_ bv124 8)) (or (and (bvule (bvadd (_ bv255 8) (bvneg var_1_x)) (_ bv255 8))) (and (bvule (bvadd (_ bv255 8) (bvneg var_1_y)) (_ bv255 8))))) (and (bvule (bvadd (bvmul (_ bv1 8) (bvadd (_ bv255 8) (bvneg var_1_x))) (bvmul (_ bv1 8) (bvadd (_ bv255 8) (bvneg var_1_y))) (bvneg (_ bv0 8))) (_ bv127 8)) (or (and (bvule (bvadd (_ bv255 8) (bvneg var_1_x)) (_ bv127 8))) (and (bvule (bvadd (_ bv255 8) (bvneg var_1_x)) (_ bv255 8)) (bvule (bvadd (_ bv255 8) (bvneg var_1_y)) (_ bv255 8))) (and (bvule (bvadd (_ bv255 8) (bvneg var_1_y)) (_ bv127 8))))) (and (bvule (bvadd (bvmul (_ bv1 8) (bvadd (_ bv255 8) (bvneg var_1_x))) (bvmul (_ bv1 8) (bvadd (_ bv255 8) (bvneg var_1_y))) (bvneg (_ bv128 8))) (_ bv127 8)) (or (and (bvule (bvadd (_ bv255 8) (bvneg var_1_x)) (_ bv127 8)) (bvule (bvadd (_ bv255 8) (bvneg var_1_y)) (_ bv255 8))) (and (bvule (bvadd (_ bv255 8) (bvneg var_1_x)) (_ bv255 8)) (bvule (bvadd (_ bv255 8) (bvneg var_1_y)) (_ bv127 8)))))))))",
"(or (and (and (and (bvule (bvadd (bvmul (_ bv1 8) var_1_x) (bvmul (_ bv1 8) var_1_y) (_ bv0 8)) (_ bv2 8)) (or (and (bvule var_1_x (_ bv127 8)) (bvule var_1_y (_ bv127 8))))) (or (and (bvule (bvadd (bvmul (_ bv1 8) (bvadd (_ bv255 8) (bvneg var_1_x))) (bvmul (_ bv1 8) (bvadd (_ bv255 8) (bvneg var_1_y))) (bvneg (_ bv128 8))) (_ bv124 8)) (or (and (bvule (bvadd (_ bv255 8) (bvneg var_1_x)) (_ bv255 8))) (and (bvule (bvadd (_ bv255 8) (bvneg var_1_y)) (_ bv255 8))))) (and (bvule (bvadd (bvmul (_ bv1 8) (bvadd (_ bv255 8) (bvneg var_1_x))) (bvmul (_ bv1 8) (bvadd (_ bv255 8) (bvneg var_1_y))) (bvneg (_ bv0 8))) (_ bv127 8)) (or (and (bvule (bvadd (_ bv255 8) (bvneg var_1_x)) (_ bv127 8))) (and (bvule (bvadd (_ bv255 8) (bvneg var_1_x)) (_ bv255 8)) (bvule (bvadd (_ bv255 8) (bvneg var_1_y)) (_ bv255 8))) (and (bvule (bvadd (_ bv255 8) (bvneg var_1_y)) (_ bv127 8))))) (and (bvule (bvadd (bvmul (_ bv1 8) (bvadd (_ bv255 8) (bvneg var_1_x))) (bvmul (_ bv1 8) (bvadd (_ bv255 8) (bvneg var_1_y))) (bvneg (_ bv128 8))) (_ bv127 8)) (or (and (bvule (bvadd (_ bv255 8) (bvneg var_1_x)) (_ bv127 8)) (bvule (bvadd (_ bv255 8) (bvneg var_1_y)) (_ bv255 8))) (and (bvule (bvadd (_ bv255 8) (bvneg var_1_x)) (_ bv255 8)) (bvule (bvadd (_ bv255 8) (bvneg var_1_y)) (_ bv127 8)))))))))",
"(or (and (and (and (bvule (bvadd (bvmul (_ bv1 8) var_1_y) (bvmul (_ bv1 8) var_1_t1) (_ bv0 8)) (_ bv2 8)) (or (and (bvule var_1_y (_ bv127 8)) (bvule var_1_t1 (_ bv127 8))))) (or (and (bvule (bvadd (bvmul (_ bv1 8) (bvadd (_ bv255 8) (bvneg var_1_y))) (bvmul (_ bv1 8) (bvadd (_ bv255 8) (bvneg var_1_t1))) (bvneg (_ bv128 8))) (_ bv124 8)) (or (and (bvule (bvadd (_ bv255 8) (bvneg var_1_y)) (_ bv255 8))) (and (bvule (bvadd (_ bv255 8) (bvneg var_1_t1)) (_ bv255 8))))) (and (bvule (bvadd (bvmul (_ bv1 8) (bvadd (_ bv255 8) (bvneg var_1_y))) (bvmul (_ bv1 8) (bvadd (_ bv255 8) (bvneg var_1_t1))) (bvneg (_ bv0 8))) (_ bv127 8)) (or (and (bvule (bvadd (_ bv255 8) (bvneg var_1_y)) (_ bv127 8))) (and (bvule (bvadd (_ bv255 8) (bvneg var_1_y)) (_ bv255 8)) (bvule (bvadd (_ bv255 8) (bvneg var_1_t1)) (_ bv255 8))) (and (bvule (bvadd (_ bv255 8) (bvneg var_1_t1)) (_ bv127 8))))) (and (bvule (bvadd (bvmul (_ bv1 8) (bvadd (_ bv255 8) (bvneg var_1_y))) (bvmul (_ bv1 8) (bvadd (_ bv255 8) (bvneg var_1_t1))) (bvneg (_ bv128 8))) (_ bv127 8)) (or (and (bvule (bvadd (_ bv255 8) (bvneg var_1_y)) (_ bv127 8)) (bvule (bvadd (_ bv255 8) (bvneg var_1_t1)) (_ bv255 8))) (and (bvule (bvadd (_ bv255 8) (bvneg var_1_y)) (_ bv255 8)) (bvule (bvadd (_ bv255 8) (bvneg var_1_t1)) (_ bv127 8)))))))))",
"(or (and (and (and (bvule (bvadd (bvmul (_ bv1 8) var_1_t2) (bvmul (_ bv1 8) var_1_t1) (_ bv0 8)) (_ bv2 8)) (or (and (bvule var_1_t2 (_ bv127 8)) (bvule var_1_t1 (_ bv127 8))))) (or (and (bvule (bvadd (bvmul (_ bv1 8) (bvadd (_ bv255 8) (bvneg var_1_t2))) (bvmul (_ bv1 8) (bvadd (_ bv255 8) (bvneg var_1_t1))) (bvneg (_ bv128 8))) (_ bv124 8)) (or (and (bvule (bvadd (_ bv255 8) (bvneg var_1_t2)) (_ bv255 8))) (and (bvule (bvadd (_ bv255 8) (bvneg var_1_t1)) (_ bv255 8))))) (and (bvule (bvadd (bvmul (_ bv1 8) (bvadd (_ bv255 8) (bvneg var_1_t2))) (bvmul (_ bv1 8) (bvadd (_ bv255 8) (bvneg var_1_t1))) (bvneg (_ bv0 8))) (_ bv127 8)) (or (and (bvule (bvadd (_ bv255 8) (bvneg var_1_t2)) (_ bv127 8))) (and (bvule (bvadd (_ bv255 8) (bvneg var_1_t2)) (_ bv255 8)) (bvule (bvadd (_ bv255 8) (bvneg var_1_t1)) (_ bv255 8))) (and (bvule (bvadd (_ bv255 8) (bvneg var_1_t1)) (_ bv127 8))))) (and (bvule (bvadd (bvmul (_ bv1 8) (bvadd (_ bv255 8) (bvneg var_1_t2))) (bvmul (_ bv1 8) (bvadd (_ bv255 8) (bvneg var_1_t1))) (bvneg (_ bv128 8))) (_ bv127 8)) (or (and (bvule (bvadd (_ bv255 8) (bvneg var_1_t2)) (_ bv127 8)) (bvule (bvadd (_ bv255 8) (bvneg var_1_t1)) (_ bv255 8))) (and (bvule (bvadd (_ bv255 8) (bvneg var_1_t2)) (_ bv255 8)) (bvule (bvadd (_ bv255 8) (bvneg var_1_t1)) (_ bv127 8)))))))))",
"(or (and (and (and (bvule (bvadd (bvmul (_ bv1 8) var_1_t2) (bvmul (_ bv1 8) var_1_t1) (_ bv0 8)) (_ bv2 8)) (or (and (bvule var_1_t2 (_ bv127 8)) (bvule var_1_t1 (_ bv127 8))))) (or (and (bvule (bvadd (bvmul (_ bv1 8) (bvadd (_ bv255 8) (bvneg var_1_t2))) (bvmul (_ bv1 8) (bvadd (_ bv255 8) (bvneg var_1_t1))) (bvneg (_ bv128 8))) (_ bv124 8)) (or (and (bvule (bvadd (_ bv255 8) (bvneg var_1_t2)) (_ bv255 8))) (and (bvule (bvadd (_ bv255 8) (bvneg var_1_t1)) (_ bv255 8))))) (and (bvule (bvadd (bvmul (_ bv1 8) (bvadd (_ bv255 8) (bvneg var_1_t2))) (bvmul (_ bv1 8) (bvadd (_ bv255 8) (bvneg var_1_t1))) (bvneg (_ bv0 8))) (_ bv127 8)) (or (and (bvule (bvadd (_ bv255 8) (bvneg var_1_t2)) (_ bv127 8))) (and (bvule (bvadd (_ bv255 8) (bvneg var_1_t2)) (_ bv255 8)) (bvule (bvadd (_ bv255 8) (bvneg var_1_t1)) (_ bv255 8))) (and (bvule (bvadd (_ bv255 8) (bvneg var_1_t1)) (_ bv127 8))))) (and (bvule (bvadd (bvmul (_ bv1 8) (bvadd (_ bv255 8) (bvneg var_1_t2))) (bvmul (_ bv1 8) (bvadd (_ bv255 8) (bvneg var_1_t1))) (bvneg (_ bv128 8))) (_ bv127 8)) (or (and (bvule (bvadd (_ bv255 8) (bvneg var_1_t2)) (_ bv127 8)) (bvule (bvadd (_ bv255 8) (bvneg var_1_t1)) (_ bv255 8))) (and (bvule (bvadd (_ bv255 8) (bvneg var_1_t2)) (_ bv255 8)) (bvule (bvadd (_ bv255 8) (bvneg var_1_t1)) (_ bv127 8)))))))))",
"(or (and (and (bvule var_2_y (_ bv2 8)) (or (bvule (bvadd (_ bv255 8) (bvneg var_2_y)) (_ bv253 8)) false))))",
"(or (and (and (bvule var_2_y (_ bv2 8)) (or (bvule (bvadd (_ bv255 8) (bvneg var_2_y)) (_ bv253 8)) false))))",
"(or (and (and (bvule var_2_y (_ bv2 8)) (or (bvule (bvadd (_ bv255 8) (bvneg var_2_y)) (_ bv253 8)) false))))",
"(or (and false))",
"(or (and false))"]

def get_model_lia(s: str, vars: List[str]) -> Optional[Dict[str, int]]:
    head = [["set-logic", "QF_LIA"], ["set-option", ":produce-models", "true"]]
    vardecl = [["declare-fun", n, [], "Int"] for n in vars]
    assertion = [["assert", f"{s}"]]
    getmodel =[["check-sat"], ["get-model"]]
    total = head + vardecl + assertion + getmodel # type: ignore
    s = smtutil.list2sexplines(total)
    x = smtutil.call_mathsat(s)
    lines = [i.strip() for i in x.split("\n")]
    if lines[0] == "unsat":
        return None
    else:
        modelsexp = "".join(lines[1:])
        # print("!", modelsexp)
        modellist = sexpdata.loads(modelsexp)
        d = {}
        for s, v in modellist:
            d[s._val] = smtutil.list2sexp(v)
        return d
    return None


def get_model_bv(s: str, vars: List[str], bitsize: int) -> Optional[Dict[str, int]]:
    head = [["set-logic", "QF_BV"], ["set-option", ":produce-models", "true"]]
    vardecl = [["declare-const", n, ["_", "BitVec", bitsize]] for n in vars]
    assertion = [["assert", f"{s}"]]
    getmodel =[["check-sat"], ["get-model"]]
    total = head + vardecl + assertion + getmodel # type: ignore
    s = smtutil.list2sexplines(total)
    x = smtutil.call_mathsat(s)
    lines = [i.strip() for i in x.split("\n")]
    if lines[0] == "unsat":
        return None
    else:
        modelsexp = "".join(lines[1:])
        # print("!", modelsexp)
        modellist = sexpdata.loads(modelsexp)
        d = {}
        for s, v in modellist:
            d[s._val] = smtutil.list2sexp(v)
        return d
    return None

def enumerate_model(s: str, vars: List[str], logic: str, bitwidth: int):
    t = sexpdata.loads(s)
    res = set()
    if logic == "lia":
        rc = treeutil.get_range_constraints(vars, 2 ** bitwidth)
        t = ["and", t, rc]
    while True:
        s_giving = smtutil.list2sexp(t)
        if logic == "lia":
            res_sat = get_model_lia(s_giving, vars)
        elif logic == "bv":
            res_sat = get_model_bv(s_giving, vars, bitwidth)
        else:
            assert False
        if res_sat == None:
            break
        else:
            # print(res_sat)
            if logic == "bv":
                res_sat1 = {k: int(sexpdata.loads(v)[1]._val[2:]) for k, v in res_sat.items()}
            elif logic == "lia":
                res_sat1 = {k: int(v) for k, v in res_sat.items()}
            else:
                assert False
            res.add(str(res_sat1))
            blocks = []
            for k, v in res_sat.items():
                blocks += [["distinct", k, v]]
            t = ["and"] + [t]  + blocks
    return res

def compare(fmlbv, fmllia, bitwidth: int) -> bool:
    vars = smtutil.get_variables_from_sexp(fmlbv)
    modelbv = enumerate_model(fmlbv, vars, "bv", bitwidth)
    modellia = enumerate_model(fmllia, vars, "lia", bitwidth)
    print("---")
    print(modellia)
    print("---")
    print(modelbv)
    print("---")
    return modellia == modelbv

# print(get_model_lia(y, ["var_1_x"]))
# enumerate_model_lia(y, ["var_1_x"])
# print(enumerate_model(y3, ["var_1_x", "var_1_y"], "bv", 8))
# enumerate_model(x3, ["var_1_x", "var_1_y"], "lia", 8)
# print(compare(y3, x3, 8))

for lia, bv in zip(lias, bvs):
    print(compare(bv, lia, 8))
