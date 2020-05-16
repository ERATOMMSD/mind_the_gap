from typing import *
import math
# import simplifier
from mathsat import *
from treeutil import get_variables, get_range_constraints, replace_symbols_for_bv, \
    substitute_tree, convert_lia2bv, convert_bv2lia, mod_to_floor, \
    convert_tree_to_capi_lia_formula, convert_tree_to_capi_bv_formula, convert_capi_formula_to_tree
from bv2lia import reduce_float_from_tree
from util import debug_print_list, debug_print_smtform, Tree, SMTUtilOption, find_subtrees_by_tags
import logging
logging.basicConfig(level=logging.DEBUG)
import util
import functools
import statistics
import time

fp = open("out/memory.log", "w+")

ENABLE_ASSERTION = False

def debug_msat_get_str(formula):
    s = msat_term_repr(formula)
    assert s
    return s



def debug_out_interpolation_problem_lia(xs):
    lines = []
    lines.append("(set-option :produce-interpolants true)")
    lines.append("(set-logic QF_LIA)")
    vsl = []
    for x in xs:
        vsl += get_variables(x)
    vs = set(vsl)
    for v in vs:
        lines.append(f"(declare-fun {v} () Int)")
    n = len(xs)
    for i, x in enumerate(xs):
        lines.append(f"(define-fun A{i} () Bool {debug_print_smtform(x)})")
    for i in range(n):
        lines.append(f"(assert (! A{i} :interpolation-group g{i}))")
    lines.append("(check-sat)")
    for i in range(n + 1):
        gs = " ".join([f"g{j}" for j in range(i)])
        lines.append(f"(get-interpolant ({gs}))")
    with open("iproblem.smt", "w") as f:
        f.write("\n".join(lines))


def make_cfg(opt: SMTUtilOption):
    """Please make sure that generated cfg is destroyed."""
    cfg = msat_create_config()
    if opt.disable_euf:
        msat_set_option(cfg, "theory.euf.enabled", "false")
    return cfg

# @lru_cache(maxsize=1024)
# @profile(stream=fp)
def check_sat_lia(fml: Tree, stat: statistics.Statistics) -> bool:
    st = time.time()
    #TODO
    cfg = msat_create_config()
    env = msat_create_env(cfg)
    assert(not MSAT_ERROR_ENV(env))

    msat_destroy_config(cfg)

    liafml_and_vars = convert_tree_to_capi_lia_formula(fml, env)
    formula = liafml_and_vars[0]
    vars = set(liafml_and_vars[1])

    int_tp = msat_get_integer_type(env)

    # declare variables/functions
    for i in vars:
        d = msat_declare_function(env, i, int_tp)
        assert (not MSAT_ERROR_DECL(d))

    assert not MSAT_ERROR_TERM(formula)

    res = msat_assert_formula(env, formula)
    assert res == 0

    s = msat_term_repr(formula)
    assert s
    logging.debug(f"Asserted formula : {s}")

    st2 = time.time()
    res = msat_solve(env)
    stat.time_smt_pure += time.time() - st2

    if res == MSAT_UNSAT:
        res_bool = False
    elif res == MSAT_SAT:
        res_bool = True
    else:
        assert False
    msat_destroy_env(env)
    stat.time_smt += time.time() - st
    stat.num_smt += 1
    return res_bool



def get_model_lia(fmls: List[Tree], stat: statistics.Statistics) -> str:
    cfg = msat_create_config()
    msat_set_option(cfg, "model_generation", "true")
    env = msat_create_env(cfg)
    assert(not MSAT_ERROR_ENV(env))

    msat_destroy_config(cfg)
    fml = ["and"] + fmls
    liafml_and_vars = convert_tree_to_capi_lia_formula(fml, env)
    formula = liafml_and_vars[0]
    vars = set(liafml_and_vars[1])
    v2decls = liafml_and_vars[2]

    int_tp = msat_get_integer_type(env)

    # declare variables/functions
    varterms = []
    for i in vars:
        d = msat_declare_function(env, i, int_tp)
        # varterms.append(msat_make_constant(env, d))
        assert (not MSAT_ERROR_DECL(d))

    assert not MSAT_ERROR_TERM(formula)

    res = msat_assert_formula(env, formula)
    assert res == 0
    

    s = msat_term_repr(formula)
    assert s
    logging.debug(f"Asserted formula : {s}")

    # st2 = time.time()
    res = msat_solve(env)
    # stat.time_smt_pure += time.time() - st2

    res_model = {}
    if res == MSAT_UNSAT:
        res_bool = False
    elif res == MSAT_SAT:
        res_bool = True

        model = msat_get_model(env)
        assert not(MSAT_ERROR_MODEL(model))
        i = msat_model_create_iterator(model)
        while(msat_model_iterator_has_next(i)):
            t, vv = msat_model_iterator_next(i)
            st = msat_term_repr(t)
            svv = msat_term_repr(vv)
            res_model[st] = svv

    else:
        assert False
    msat_destroy_env(env)
    return res_model


# @profile(stream=fp)
def check_sat_bv(fml: Tree, m: int, stat: statistics.Statistics) -> bool:
    st = time.time()
    bitwidth = int(math.log2(m))
    cfg = msat_create_config()
    env = msat_create_env(cfg)
    assert(not MSAT_ERROR_ENV(env))

    msat_destroy_config(cfg)

    bvfml_and_vars = convert_tree_to_capi_bv_formula(fml, m, env)
    formula = bvfml_and_vars[0]
    vars = set(bvfml_and_vars[1])

    bv_tp = msat_get_bv_type(env, bitwidth)

    # declare variables/functions
    for i in vars:
        d = msat_declare_function(env, i, bv_tp)
        assert (not MSAT_ERROR_DECL(d))

    assert not MSAT_ERROR_TERM(formula)

    res = msat_assert_formula(env, formula)
    assert res == 0

    s = msat_term_repr(formula)
    assert s
    logging.info(f"Asserted formula : {s}")

    st2 = time.time()
    res = msat_solve(env)
    stat.time_smt_pure += time.time() - st2

    if res == MSAT_UNSAT:
        res_bool = False
    elif res == MSAT_SAT:
        res_bool = True
    else:
        assert False
    msat_destroy_env(env)
    stat.time_smt += time.time() - st
    stat.num_smt += 1
    return res_bool


def get_model_bv(fmls: List[Tree], m: int, stat: statistics.Statistics) -> str:
    st = time.time()
    bitwidth = int(math.log2(m))
    cfg = msat_create_config()
    msat_set_option(cfg, "model_generation", "true")
    env = msat_create_env(cfg)
    assert(not MSAT_ERROR_ENV(env))

    msat_destroy_config(cfg)
    fml = ["and"] + fmls
    bvfml_and_vars = convert_tree_to_capi_bv_formula(fml, m, env)
    formula = bvfml_and_vars[0]
    vars = set(bvfml_and_vars[1])

    bv_tp = msat_get_bv_type(env, bitwidth)

    int_tp = msat_get_integer_type(env)

    # declare variables/functions
    for i in vars:
        d = msat_declare_function(env, i, bv_tp)
        # varterms.append(msat_make_constant(env, d))
        assert (not MSAT_ERROR_DECL(d))

    assert not MSAT_ERROR_TERM(formula)

    res = msat_assert_formula(env, formula)
    assert res == 0
    

    s = msat_term_repr(formula)
    assert s
    logging.debug(f"Asserted formula : {s}")

    res = msat_solve(env)

    res_model = {}
    if res == MSAT_UNSAT:
        res_bool = False
    elif res == MSAT_SAT:
        res_bool = True

        model = msat_get_model(env)
        assert not(MSAT_ERROR_MODEL(model))
        i = msat_model_create_iterator(model)
        while(msat_model_iterator_has_next(i)):
            t, vv = msat_model_iterator_next(i)
            st = msat_term_repr(t)
            svv = msat_term_repr(vv)
            res_model[st] = svv

    else:
        assert False
    msat_destroy_env(env)
    return res_model



def check_valid_lia(fml: Tree, stat: statistics.Statistics) -> bool:
    fml = ["not", fml]
    return not check_sat_lia(fml, stat)


def check_valid_bv(fml: Tree, m: int, stat: statistics.Statistics) -> bool:
    fml = ["not", fml]
    return not check_sat_bv(fml, m, stat)

class UnexpectedInterpolationError(Exception):
    pass

# @profile(stream=fp)
def get_interpolants_lia(fmls: List[Tree], opt: SMTUtilOption, stat:statistics.Statistics) -> List[Any]:
    st = time.time()
    cfg = make_cfg(opt)
    msat_set_option(cfg, "interpolation", "true")

    env = msat_create_env(cfg)
    assert(not MSAT_ERROR_ENV(env))
    try:

        msat_destroy_config(cfg)

        liafmls_and_vars = [convert_tree_to_capi_lia_formula(fml, env) for fml in fmls]
        liafmls = [i[0] for i in liafmls_and_vars]
        vars = set()
        for _, i, _ in liafmls_and_vars:
            vars = vars.union(set(i))

        int_tp = msat_get_integer_type(env)

        # declare variables/functions
        for i in vars:
            d = msat_declare_function(env, i, int_tp)
            assert (not MSAT_ERROR_DECL(d))

        groups: List[Any] = []
        for formula in liafmls:
            # now create the interpolation groups representing the two formulas A and B
            group = msat_create_itp_group(env)
            assert group != -1

            assert not MSAT_ERROR_TERM(formula)

            # tell MathSAT that all subsequent formulas belonging to group A
            res = msat_set_itp_group(env, group)
            assert res == 0
            res = msat_assert_formula(env, formula)
            assert res == 0

            s = msat_term_repr(formula)
            assert s
            logging.info(f"Asserted formula (in group {group}: {s}")
            groups.append(group)

        if msat_solve(env) == MSAT_UNSAT:
            interpolants = []
            for fmlid in range(len(fmls) + 1):
                st2 = time.time()
                interpolant = msat_get_interpolant(env, groups[:fmlid])
                stat.time_interp_pure += time.time() - st2
                stat.time_smt_pure += time.time() - st2
                if MSAT_ERROR_TERM(interpolant):
                    raise UnexpectedInterpolationError()
                s = msat_term_repr(interpolant)
                assert s
                logging.info(f"\nOK, the interpolant is {s}")
                conv = convert_capi_formula_to_tree(interpolant, env)
                interpolants.append(conv)
            # msat_destroy_env(env)
            res_interp = interpolants
        else:
            # msat_destroy_env(env)
            res_interp = []
    finally:
        msat_destroy_env(env)
    # reduce floor
    untranslateds = res_interp
    translateds = []
    st2 = time.time()
    for i, t in enumerate(res_interp):
        if opt.theory == "lia":
            translateds.append(rewrite_float_to_mod(t))
            assert_no_float(translateds[-1])
        elif opt.theory == "liabv":
            # res = rewrite_float_to_mod(t)
                
            # assert_no_float(res)
            # translateds.append(res)
            try:
                res = rewrite_float_to_mod(t)
                
                assert_no_float(res)
                translateds.append(res)
            except:
                print("Fallen to strategical")
                m = 2**opt.bitwidth
                d = {v: (0, m - 1) for v in get_variables(t)}
                t1, rep = reduce_float_from_tree(t, d, m, opt)
                t2 = substitute_tree(t1, rep)
                translateds.append(t2)
                assert_no_float(translateds[-1])
            pass
        else:
            assert False
    print("converted")
    stat.time_process_lia += time.time() - st2
    res_interp = translateds
    # calc time
    res_time = time.time() - st
    stat.time_smt += res_time
    stat.time_interp += res_time
    stat.num_smt += 1
    stat.num_interp += 1
    assert_valid_interpolant_lia(fmls, res_interp, untranslateds, opt)
    return res_interp

class NaiveRewritingFailedException(Exception):
    pass

def rewrite_float_to_mod(t: Tree) -> Tree:
    import copy
    def is_floor(t):
        if type(t) is list and t[0] == "floor":
            return True
        return False
    def help(t):
        if type(t) is list:
            tag = t[0]
            if tag == "<=":
                left = t[1]
                right = t[2]
                if is_floor(right):
                    inside_floor = right[1]
                    if "/" in inside_floor[1]:
                        num, den = inside_floor[1].split("/")
                        num = int(num)
                        den = int(den)
                        return ["<=", help(["*", den, left]), help(["*", num, inside_floor[2]])]
                    else:
                        return ["<=", help(left), help(right[1])]
            elif tag == "*" and is_floor(t[2]):
                left = t[1]
                left = int(left)
                inside_floor = t[2][1]
                assert inside_floor[0] == "*"
                assert "/" in inside_floor[1]
                num, den = inside_floor[1].split("/")
                num = int(num)
                den = int(den)
                divided = help(inside_floor[2])
                dividing = den
                if left % dividing != 0:
                    raise NaiveRewritingFailedException()
                res = ["*", left//dividing, ["-", copy.deepcopy(divided), ["mod", copy.deepcopy(divided), dividing]]]
                return res
            return [tag] + [help(i) for i in t[1:]]
        elif type(t) is str and "/" in t:
            # print("?")
            raise NaiveRewritingFailedException()
            assert False
        else:
            return t
    try:
        return help(t)
    except:
        raise NaiveRewritingFailedException()


class FloorException(Exception):
    pass

def assert_no_float(t: Tree) -> None:
    if ENABLE_ASSERTION == False:
        return
    def help(t):
        if type(t) is list:
            for i in t:
                help(i)
        else:
            if t == "floor":
                raise FloorException()
            if type(t) is str and "/" in t:
                raise FloorException()
    try:
        help(t)
    except FloorException:
        print(debug_print_list(t))
        print(t)
        assert False


def get_interpolants_bv(bvfmls: List[Tree], m: int, opt: SMTUtilOption, stat: statistics.Statistics) -> Tuple[List[str], List[str], List[Any]]:
    st = time.time()
    liafmls = [convert_bv2lia(fml, m) for fml in bvfmls]
    # print("checking")
    # for b, l in zip(bvfmls, liafmls):
    #     assert_equal_bv(b, m, opt)
    # print("checked")
    stat.time_to_lia += time.time() - st
    withconstraints = [["and", p, get_range_constraints(get_variables(p), m)] for p in liafmls]
    # withconstraints = liafmls
    # lia_interpols = get_interpolants_lia(withconstraints)
    while True:
        try:
            lia_interpols = get_interpolants_lia(withconstraints, opt, stat)
            break
        except UnexpectedInterpolationError:
            logging.warning("Failed at interpolation")
            if not opt.expand_floor_in_inteprolation_error:
                raise UnexpectedInterpolationError()
            modsmods = [find_subtrees_by_tags(["mod"], liafml) for liafml in liafmls]
            mods = functools.reduce(list.__add__, modsmods)
            if not mods:
                raise UnexpectedInterpolationError()

            def get_divisor(x):
                assert x[0][0] == "mod"
                return int(x[0][2])

            maxmod, _ = util.argmax(mods, get_divisor)
            logging.warning(f"{debug_print_list(maxmod[0])} is being replaced.")
            maxmod[1][maxmod[2]] = mod_to_floor(maxmod[0])

            def fv(t):
                return {v: (0, m - 1) for v in get_variables(t)}

            def ft(t):
                reduced_t, replacing_floor = reduce_float_from_tree(t, fv(t), m, opt, "strategy1")
                processed_replacing_floor = substitute_tree(reduced_t, replacing_floor)
                return processed_replacing_floor

            withconstraints = [ft(t) for t in withconstraints]
            print("hoge")
            stat.num_interp_failure += 1
    st = time.time()
    converteds = [convert_lia2bv(fml, m, opt, stat) for fml in lia_interpols]
    stat.time_from_lia += time.time() - st
    assert_valid_interpolant_bv(bvfmls, converteds, lia_interpols, liafmls, m, opt)
    return converteds, lia_interpols, withconstraints


def assert_valid_interpolant_lia(liafmls: List[Tree], interpolants: List[Tree], u, opt: SMTUtilOption):
    if ENABLE_ASSERTION == False:
        return
    assert len(liafmls) + 1 == len(interpolants)
    for i in range(1, len(liafmls)):
        a = ["and"] + liafmls[:i]
        b = ["and"] + liafmls[i:]
        interpolant = interpolants[i]
        valid_a = ["=>", a, interpolant]
        res_a = check_valid_lia(valid_a, statistics.Statistics())
        if res_a == False:
            print("[Invalid interpolant]")
            print("[A part]", util.debug_print_preds(liafmls[:i]))
            print("[interpolant]", util.debug_print_list(interpolant))
            print("[model]", get_model_lia([["not", valid_a]], statistics.Statistics()))
            print("[corresponding u]", util.debug_print_list(u[i]))
            print(u[i])
            
            assert False
        valid_b = ["=>", ["and", b, interpolant], False]
        res_b = check_valid_lia(valid_b, statistics.Statistics())
        if res_b == False:
            print("[Invalid interpolant]")
            print("[B part]", util.debug_print_preds(liafmls[:i]))
            print("[interpolant]", util.debug_print_list(interpolant))
            # print("corresponding LIA", util.debug_print_list(lia_interpolant[i+1]))
            assert False


def assert_valid_interpolant_bv(bvfmls: List[Tree], interpolants: List[Tree], lia_interpolant: List[Tree], liafmls: List[Tree], m: int, opt: SMTUtilOption):
    if ENABLE_ASSERTION == False:
        return
    assert len(bvfmls) + 1 == len(interpolants)
    for i in range(1, len(bvfmls)):
        a = ["and"] + bvfmls[:i]
        b = ["and"] + bvfmls[i:]
        interpolant = interpolants[i]
        valid_a = ["=>", a, interpolant]
        res_a = check_valid_bv(valid_a, m, statistics.Statistics())
        if res_a == False:
            print("[Invalid interpolant]")
            print("[A part]", util.debug_print_preds(bvfmls[:i]))
            print("[corresponding A part]", util.debug_print_preds(liafmls[:i]))
            print("[interpolant]", util.debug_print_list(interpolant))
            print("[corresponding LIA interp]", util.debug_print_list(lia_interpolant[i]))
            print("[model]", get_model_bv([["not", valid_a]], m, statistics.Statistics()))
            assert_equal_lia_and_bv(interpolant, lia_interpolant[i], m, opt)
            assert False
        valid_b = ["=>", ["and", b, interpolant], "false"]
        res_b = check_valid_bv(valid_b, m, statistics.Statistics())
        if res_b == False:
            print("[Invalid interpolant]")
            print("[B part]", util.debug_print_preds(bvfmls[:i]))
            print("[corresponding B part]", util.debug_print_preds(liafmls[:i]))
            print("[interpolant]", util.debug_print_list(interpolant))
            print("[corresponding LIA]", util.debug_print_list(lia_interpolant[i]))
            print("[model]", get_model_bv([["not", valid_b]], m, statistics.Statistics()))
            assert_equal_lia_and_bv(interpolant, lia_interpolant[i], m, opt)
            assert False



def assert_lia_processing_ok(liafml, m: int, opt: SMTUtilOption) -> None:
    if ENABLE_ASSERTION == False:
        return
    import treeutil
    import smtutil
    import copy
    liafml1 = treeutil.process_overflow(copy.deepcopy(liafml), m, opt)
    for st in treeutil.get_atomic_formulae(liafml1):
        bv = treeutil.replace_symbols_for_bv(st, m, "unsigned", {})
        assert_equal_lia_and_bv(bv, st, m, opt, False)


def assert_equal_bv(bvfml, m: int, opt: SMTUtilOption):
    if ENABLE_ASSERTION == False:
        return
    import treeutil
    import smtutil
    liafml = treeutil.convert_bv2lia(bvfml, m)
    bvfml1 = convert_lia2bv(liafml, m, opt)
    iff = ["and", ["=>", bvfml, bvfml1], ["=>", bvfml1, bvfml]]
    m = smtutil.get_model_bv([["not", iff]], m, statistics.Statistics())
    if m:
        print("[bvfml]", util.debug_print_list(bvfml))
        print("[bvfml1]", util.debug_print_list(bvfml1))
        print("got model in bvprocessing", m)
        assert False

def assert_equal_lia_and_processed_lia(liafml, m: int, opt: SMTUtilOption) -> None:
    if ENABLE_ASSERTION == False:
        return
    import treeutil
    import smtutil
    import copy
    print("hoge")
    liafml1 = treeutil.process_overflow(copy.deepcopy(liafml), m, opt)
    iff = ["and", ["=>", liafml, liafml1], ["=>", liafml1, liafml]]
    vs = treeutil.get_variables(iff)
    cs = treeutil.get_range_constraints(vs, m)
    with_c = ["and"] + [iff] + [cs]
    m = smtutil.get_model_lia([["and", ["not", iff], cs]], statistics.Statistics())
    if m:
        print("[liafml]", util.debug_print_list(liafml))
        print("[liafml1]", util.debug_print_list(liafml1))
        print("got model in liaprocessing", m)
        assert False
    return

def evaluate_subformulae_lia(fml, subst, m, opt, check_bv):
    import treeutil
    for st in treeutil.get_atomic_formulae(fml):
        # print("----Checking subformula")
        ev = check_sat_lia(["and", st, subst], statistics.Statistics())
        print(util.debug_print_list(st), "::", ev)
        # hoge
        if check_bv:
            bv = convert_lia2bv(st, m, opt)
            assert_equal_lia_and_bv(bv, st, m, opt, False)


def evaluate_subformulae_bv(fml, subst, m):
    import treeutil
    for st in treeutil.get_atomic_formulae(fml):
        f = ["and", st, subst]
        ev = check_sat_bv(f, m, statistics.Statistics())
        print(util.debug_print_list(st), "::", ev)


def assert_equal_lia_and_bv(bvfml, liafml, m: int, opt, check_sub=True) -> None:
    import treeutil
    import smtutil
    import copy
    liafml_p = treeutil.process_overflow(copy.deepcopy(liafml), m, opt)
    liafml1 = treeutil.convert_bv2lia(bvfml, m)
    iff = ["and", ["=>", liafml, liafml1], ["=>", liafml1, liafml]]
    vs = treeutil.get_variables(iff)
    cs = treeutil.get_range_constraints(vs, m)
    with_c = ["and"] + [iff] + [cs]
    mo = smtutil.get_model_lia([["and", ["not", iff], cs]], statistics.Statistics())
    if mo:
        liafml_subst = ["and"] + [["=", k, v] for k, v in mo.items()]
        bv_subst = ["and"] + [["=", k, ["_", f"bv{v}", m]] for k, v in mo.items()]
        liafml_with_c = ["and"] + [liafml] + [treeutil.get_range_constraints(treeutil.get_variables(liafml), m)]
        liafml_p_with_c = ["and"] + [liafml_p] + [treeutil.get_range_constraints(treeutil.get_variables(liafml_p), m)]
        print("----")
        print("[liafml]", util.debug_print_list(liafml))
        print("[liafml_p]", util.debug_print_list(liafml_p))
        print("[bvfml]", util.debug_print_list(bvfml))
        print("[liafml2]", util.debug_print_list(liafml1))
        print("got model in liaandbv", mo)
        print("[res_liafml]", check_sat_lia(["and", liafml_with_c, liafml_subst], statistics.Statistics()))
        print("[res_liafml_p]", check_sat_lia(["and", liafml_p_with_c, liafml_subst], statistics.Statistics()))
        print("[res_bvfml]", check_sat_bv(["and", bv_subst, bvfml], m, statistics.Statistics()))
        print("----")
        if check_sub:
            evaluate_subformulae_lia(liafml_with_c, liafml_subst, m, opt, True)
        print("----")
        evaluate_subformulae_lia(liafml_p_with_c, liafml_subst, m, opt, False)
        print("----")
        evaluate_subformulae_bv(bvfml, bv_subst, m)
        assert False
    return


# can be more efficient

# def box_all_ineqs_in_tree(t: Any, m: int) -> Any:
#     def help(tree: Any, parent: Any=None, id: int=-1) -> Any:
#         if type(tree) is list:
#             if tree[0]._val in ["<=", ">=", "<", ">"]:
#                 if parent is None:
#                     tree = box_ineq(tree, m) # type: ignore
#                 else:
#                     parent[id] = box_ineq(tree, m) # type: ignore
#             else:
#                 for i in range(len(tree)):
#                     help(tree[i], tree, i)
#         return tree
#     t = help(t)
#     return t


# def find_mod(t: Tree) -> Tuple[int, ]


if __name__ == "__main__":
    def test1():
        cfg = msat_create_config()
        env = msat_create_env(cfg)
        type_int = msat_get_integer_type(env)
        msat_declare_function(env, "x", type_int)
        # automatically normalized inside
        formula = msat_from_string(env, "(<= (+ 1 x) 2)")
        print(convert_capi_formula_to_tree(formula, env))
    # test1()

    def test2():
        cfg = msat_create_config()
        env = msat_create_env(cfg)
        type_int = msat_get_integer_type(env)
        msat_declare_function(env, "x", type_int)
        # automatically normalized inside
        formula = msat_from_string(env, "(<= (mod x 3) 1)")
        print(convert_capi_formula_to_tree(formula, env))
    test2()

    def test3():
        cfg = msat_create_config()
        env = msat_create_env(cfg)
        type_int = msat_get_integer_type(env)
        msat_declare_function(env, "x", type_int)
        # automatically normalized inside
        formula = msat_from_string(env, "(= (mod x 3) 0)")
        print(convert_capi_formula_to_tree(formula, env))
    test3()

    def test4():
        cfg = msat_create_config()
        env = msat_create_env(cfg)
        type_int = msat_get_integer_type(env)
        msat_declare_function(env, "x", type_int)
        # automatically normalized inside
        formula = ["<=", ["+", 2, "var_0_x", "var_0_z"], "var_0_y"]
        f = convert_tree_to_capi_bv_formula(formula, 2 ** 8, env)
        print(msat_term_repr(f))
    # test4()

    def test5():
        cfg = msat_create_config()
        env = msat_create_env(cfg)
        type_int = msat_get_integer_type(env)
        msat_declare_function(env, "x", type_int)
        # automatically normalized inside
        formula = ["<=", ["+", 2, "var_0_x", "var_0_z"], "var_0_y"]
        f, vars = convert_tree_to_capi_lia_formula(formula, env)
        print(msat_term_repr(f))
    # test5()

    def test6():
        cfg = msat_create_config()
        env = msat_create_env(cfg)
        type_int = msat_get_integer_type(env)
        msat_declare_function(env, "x", type_int)
        # automatically normalized inside
        formula = ['<=', 0, 'var_1_x']
        f, vars = convert_tree_to_capi_lia_formula(formula, env)
        print(msat_term_repr(f))
    # test6()

    def test7():
        cfg = msat_create_config()
        env = msat_create_env(cfg)
        type_int = msat_get_integer_type(env)
        msat_declare_function(env, "x", type_int)
        # automatically normalized inside
        formula = ['and', ['not', ['ite', ['or', ['and', ['<=', 1, 127], ['<=', 'var_0_y', 127]], ['and', ['>=', 1, 128], ['>=', 'var_0_y', 128]]], ['<=', 1, 'var_0_y'], ['>=', 1, 128]]], ['and', ['<=', 0, 'var_0_y'], ['<=', 'var_0_y', 255]]]
        formula = ['not', ['ite', ['or', ['and', ['<=', 1, 127], ['<=', 'var_0_y', 127]], ['and', ['>=', 1, 128], ['>=', 'var_0_y', 128]]], ['<=', 1, 'var_0_y'], ['>=', 1, 128]]]
        formula = ['ite', ['or', ['and', ['<=', 1, 127], ['<=', 'var_0_y', 127]], ['and', ['>=', 1, 128], ['>=', 'var_0_y', 128]]], ['<=', 1, 'var_0_y'], ['>=', 1, 128]]
        f, vars = convert_tree_to_capi_lia_formula(formula, env)
        print(msat_term_repr(f))
    # test7()

    def test8():
        fml = ['not', ['bvsge', 'var_1_y', ['_', 'bv1', 3]]]
        res = convert_bv2lia(fml, 8)
        print(res)
    # test8()

    def test9():
        fml = ['and', ['<=', 'var_3_x', '7'], ['<=', ['+', 'var_3_x', ['*', '-8', ['floor', ['*', '1/8', ['*', '1', 'var_3_x']]]]], '0']]
        res = reduce_float_from_tree(fml, {"var_3_x": (0, 7)})
        print(res)
    # test9()

    def test10():
        # It has to be var_1_x=0
        fml = ["and", ["<=", "0", "var_1_x"], ["<=", 0, ["floor", ["*", "1/8", ["*", "-1", "var_1_x"]]]]]
        res = convert_lia2bv(fml, 2 ** 3)
        print(res)
    # test10()


    def test11():
        fml = ["<=", "var_1_x", 0]
        res = replace_symbols_for_bv(fml, 2 ** 3, False)
        print(res)
    # test11()

    def test12():
        fml = ["<=", "var_1_x", ["-", "1"]]
        res = convert_lia2bv(fml, 2 ** 3)
        print(res)
    # get_model_lia([["=", "var_0_x", "3"]], None)
        # (<=, var_1_x, 0)
    def test13():
        x = ["and", ['and', ['=', 'var_1_n', 'var_0_dummy_nondetint_0'], 'true'], ['and', ['=', 'var_1_l', 'var_0_dummy_nondetint_1'], 'true'], ['and', ['=', 'var_1_r', 'var_0_dummy_nondetint_2'], 'true'], ['and', ['=', 'var_1_i', 'var_0_dummy_nondetint_3'], 'true'], ['and', ['=', 'var_1_j', 'var_0_dummy_nondetint_4'], 'true'], ['and', ['=', 'var_2_n', 'var_0_dummy_nondetint_5'], 'true'], ['and', ['ite', ['or', ['and', ['<=', 1, 127], ['<=', 'var_2_n', 127]], ['and', ['>=', 1, 128], ['>=', 'var_2_n', 128]]], ['<=', 1, 'var_2_n'], ['>=', 1, 128]], ['ite', ['or', ['and', ['<=', 'var_2_n', 127], ['<=', 64, 127]], ['and', ['>=', 'var_2_n', 128], ['>=', 64, 128]]], ['<=', 'var_2_n', 64], ['>=', 'var_2_n', 128]]], 
        ['and', ['=', 'var_2_l', ['mod', ['+', ['mod', ['div', 'var_2_n', 2], 256], 1], 256]], 'true']]
        # (and (<= var_2_n 127) (<= (* 2 1) (* 1 (* 1 var_2_n))))
        y = ["and", ["<=", "var_2_n", 127], ["<=", ["*", 2, 1], ["*", ["*", 1, "var_2_n"]]]]
        a = ["=>", x, y]
        res = check_valid_lia(a, statistics.Statistics())
        print(res)

    test13()

    def test14():
        x = ['=', 'var_2_l', ['bvadd', ['bvdiv', 'var_2_n', ['_', 'bv2', 8]], ['_', 'bv1', 8]]]
        res = convert_bv2lia(x, 256)
        print(res)
    # test14()

    def test15():
        x = ["=", "var_0_x", ["div", 1, 2]]
        res = check_sat_lia(x, statistics.Statistics())
        print(res)
    test15()
