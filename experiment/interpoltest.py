# import sexpdata

# # http://smtlib.cs.uiowa.edu/theories-Ints.shtml
# head = [["set-option", ":produce-interpolants", "true"],
#     ["set-logic", "QF_LIA"]]
# vars  = ["x", "y", "z"]
# decl = [["declare-fun", s, [], "Int"] for s in vars]
# fmls = [["=", "y", ["*", "2", "x"]],
#     ["=", "y", ["+", ["*", "2", "z"], "1"]]]
# def_fmls = [["define-fun", f"A{i}", [], "Bool", v] for i, v in enumerate(fmls)]
# assertions = [["assert", ["!", f"A{i}", ":interpolation-group", f"g{i}"]] for i in range(len(fmls))]
# checksat =[["check-sat"]]
# gets = [["get-interpolant", [f"g{i}" for i in range(n)]] for n in range(len(fmls) + 1)]

# all = head + decl + def_fmls + assertions + checksat + gets #type: ignore
# for line in all:
#     x = sexpdata.dumps(line)
#     x = x.replace("\"", "")
#     print(x)


from pysmt.shortcuts import Solver, sequence_interpolant, Equals, \
Times, Plus, Symbol, Int, get_env, Interpolator, is_sat, And, Exists, qelim, LE, LT, GT, GE, QuantifierEliminator
import pysmt

# print(get_env().factory.all_quantifier_eliminators(logic=pysmt.logics.LIA))
# with QuantifierEliminator(name="msat_lw", logic=pysmt.logics.LIA) as solver:
#     x = Symbol("x", pysmt.shortcuts.types.INT)
#     y = Symbol("y", pysmt.shortcuts.types.INT)
#     f = Exists([x], Equals(y, Times(x, Int(2))))
#     qelim(f, solver)

# x = Symbol("x", pysmt.shortcuts.types.INT)
# y = Symbol("y", pysmt.shortcuts.types.INT)
# f = Exists([x], Equals(y, Times(x, Int(2))))
# qelim(f)

# with Solver(name="msat", logic=pysmt.logics.QF_LIA) as solver:
#     x = Symbol("x", pysmt.shortcuts.types.INT)
#     y = Symbol("y", pysmt.shortcuts.types.INT)
#     f = Exists([x], Equals(y, Times(x, Int(2))))
#     qelim(f, solver)

#     y = Symbol("y", pysmt.shortcuts.types.INT)
#     x = Symbol("x", pysmt.shortcuts.types.INT)
#     z = Symbol("z", pysmt.shortcuts.types.INT)
#     f1 = Equals(y, Times(Int(2), x))
#     f2 = Equals(y, Plus(Times(Int(2), z), Int(1)))
#     s = is_sat(And(f1, f2))
#     print(s)

i = Interpolator("msat")

x = Symbol("x", pysmt.shortcuts.types.INT)
y = Symbol("y", pysmt.shortcuts.types.INT)
z = Symbol("z", pysmt.shortcuts.types.INT)

a = And(LE(x, Int(1)), LT(y, x))
b = And(GE(y, z), GT(z, Int(0)))
s = i.sequence_interpolant([a, b])
print(s)


f1 = Equals(y, Plus(x, Int(2)))
f2 = Equals(y, x)
s = i.sequence_interpolant([f1, f2])
print(s)


f1 = Equals(y, Plus(Int(2), Times(Int(2), x)))
f2 = Equals(y, Plus(Times(Int(2), x), Int(1)))
s = i.sequence_interpolant([f1, f2])
print(s)

f1 = Equals(y, Plus(Int(2), Times(Int(2), x)))
f2 = Equals(y, Plus(Int(1), Times(Int(2), z)))
print(f2)
s = i.binary_interpolant(f1, f2)
# Does not work!  The implementation in msat.py try to convert the sequences back and fails (why?).
# s = i.sequence_interpolant([f1, f2])
print(s)

# with Solver(name="msat", logic=pysmt.logics.QF_LIA) as solver:
#     y = Symbol("y", pysmt.shortcuts.types.INT)
#     x = Symbol("x", pysmt.shortcuts.types.INT)
#     z = Symbol("z", pysmt.shortcuts.types.INT)
#     f1 = Equals(y, Times(Int(2), x))
#     f2 = Equals(y, Plus(Times(Int(2), z), Int(1)))
#     s = sequence_interpolant([f1, f2], solver, logic=pysmt.logics.QF_LIA)
#     print(s)
# (compute-interpolant
#    (= y (* 2 x))
#    (= y (+ (* 2 z) 1)))
