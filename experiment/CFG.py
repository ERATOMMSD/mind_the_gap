import sys
from pycparser import *
import pycparser
from typing import *

import treeutil
import util
from graphviz import Digraph
import logging
logging.basicConfig(level=logging.WARNING)
import smtutil

FORCE_NOMERGE = False

def is_error_function(x: str) -> bool:
    if x == "errorFn":
        return True
    if x.startswith("__VERIFIER_error"):
        return True
    return False

def is_assumption(x: str) -> bool:
    return x in ["assume", "__VERIFIER_assume"]


def is_assertion(x: str) -> bool:
    return x in ["assert", "__VERIFIER_assert"]


def is_nondet(x: str) -> bool:
    if x.startswith("unknown"):
        return True
    return False


def is_nondet_int(x: str) -> bool:
    if x.startswith("__VERIFIER_nondet_int") or x.startswith("__VERIFIER_nondet_bool") or x == "rand" or x == "nondet":
        return True
    return False


def is_nondet_unsigned(x: str) -> bool:
    if x.startswith("__VERIFIER_nondet_uint"):
        return True
    return False

def is_ignoring_function_name(x: str) -> bool:
    return is_error_function(x) or is_nondet(x) or is_nondet_int(x) or is_assumption(x) or is_assertion(x) or is_nondet_unsigned(x) or x.startswith("__VERIFIER_nondet") or x == "__mark"

class CFGVar():
    def __init__(self,
        name: str,
        typ: str) -> None:
            self.name: str = name
            self.typ: str = typ
    def __eq__(self, opponent):
        return self.name == opponent.name and self.typ == opponent.typ
    def __hash__(self):
        return hash((self.name, self.typ))

# nesting of (type: str, value of str or CFGValue, ...)
class CFGValue():
    def __init__(self, x: Any = None):
        if x is None:
            x = ("Constant", "true", "bool")
        self.val: Any = x
    def __str__(self):
        if self.val[0] == "Nondet":
            return "ND"
        elif self.val[0] == "Constant":
            return self.val[1]
        elif self.val[0] == "NondetInt":
            return "*nondetint"
        elif self.val[0] == "NondetUnsigned":
            return "*nondetunsigned"
        elif self.val[0] == "Var":
            return self.val[1]
        elif self.val[0] == "Unary":
            return f"{self.val[1]} {self.val[2]}"
        elif self.val[0] == "Binary":
            return f"{self.val[2]} {self.val[1]} {self.val[3]}"
        else:
            raise Exception("Unexpected formula")

class Operation:
    pass

class OperationBin(Operation):
    def __init__(self, lhs_: str, rhs_: CFGValue, op_: str):
        self.lhs: str = lhs_
        self.rhs: CFGValue = rhs_
        self.op: str = op_
    def __str__(self) -> str:
        return f"{self.lhs} {self.op} {self.rhs}"

class OperationUni(Operation):
    def __init__(self, v_: str, op: str):
        self.v = v_
        self.op = op
    def __str__(self) -> str:
        if self.op == "p++":
            return f"{self.v}++"
        elif self.op == "p--":
            return f"{self.v}--"
        elif self.op == "++":
            return f"++{self.v}"
        elif self.op == "--":
            return f"--{self.v}"
        else:
            return f"????"

class CFGNode():
    def __init__(self, id_: int):
        self.id: int = id_
        self.coming: Set[CFGEdge] = set()
        self.going: Set[CFGEdge] = set()
    def __str__(self):
        return f"N{self.id}"

class CFGEdge():
    def __init__(self,
                 id_: int,
                 source_: CFGNode,
                 target_: CFGNode,
                 pred_: CFGValue,
                 op_: Optional[OperationBin]):
        self.id = id_
        self.source: CFGNode = source_
        self.target: CFGNode = target_
        self.condition: CFGValue = pred_
        self.operation: Optional[OperationBin] = op_
        self.pred: util.Tree = None
        self.vars: List[str] = []
    def __str__(self):
        s = ""
        if self.condition.val == ("Constant", "true", "bool"):
            pass
        else:
            s = f"[{str(self.condition)}]"
        t = ""
        if self.operation is not None:
            t = str(self.operation)
        return f"{self.id})" + s + t
    def get_predicate_as_str(self):
        return util.debug_print_list(self.pred)


class GotoManager:
    def __init__(self):
        self.sources: Set[CFGNode] = set()
        self.target: Optional[CFGNode] = None
    def swap(self, a: CFGNode, b: CFGNode):
        if a in self.sources:
            self.sources.remove(a)
            self.sources.add(b)
        if self.target == a:
            self.target = b

class ControlFlowGraph():
    def __init__(self):
        self.initial: CFGNode = CFGNode(0)
        self.final: CFGNode = CFGNode(1)
        self.exit: CFGNode = self.initial
        self.nodes: List[CFGNode] = [self.initial, self.final]
        self.edges: List[CFGEdge] = []
        self.vars: Set[CFGVar] = set()
    def get_vars_as_dict(self) -> Dict[str, str]:
        return dict([(x.name, x.typ) for x in self.vars])
    def add_edge(self,
                 src: CFGNode,
                 tgt: Optional[CFGNode],
                 cond: CFGValue,
                 op: Optional[Operation]) -> Tuple[CFGEdge, CFGNode]:
        assert op is None or isinstance(op, Operation)
        assert self.nodes[src.id] == src
        if tgt is not None:
            assert self.nodes[tgt.id] == tgt
        if tgt is None:
            tgt = CFGNode(len(self.nodes))
            self.nodes.append(tgt)
            assert self.nodes[tgt.id] == tgt
        e = CFGEdge(len(self.edges), src, tgt, cond, op)
        e.source = src
        e.target = tgt
        e.condition = cond
        e.operation = op
        src.going.add(e)
        tgt.coming.add(e)
        self.edges.append(e)
        return e, tgt
    def remove_edge(self, e: CFGEdge) -> None:
        assert self.edges[e.id] == e
        e.source.going.remove(e)
        e.target.coming.remove(e)

        # for n in self.nodes[e.id:]:
        #     n.id -= 1
        self.edges.remove(e)
        for i, n in enumerate(self.nodes):
            n.id = i
        for i, e in enumerate(self.edges):
            e.id = i
    def remove_node(self, n: CFGNode) -> None:
        assert self.nodes[n.id] == n
        assert n != self.initial
        assert False

    def print(self, name="print", style="c") -> None:
        G = Digraph(format="png")

        for n in self.nodes:
            # logging.debug(f"node from {n.id}")
            if n == self.initial:
                G.node(str(n.id), f"ENTRY")
            elif n == self.final:
                G.node(str(n.id), f"ERROR")
            else:
                G.node(str(n.id), f"N{n.id}")
        for e in self.edges:
            # logging.debug(f"edge from {e.source.id} to {e.target.id}")
            if style == "c":
                s = str(e)
            elif style == "pred":
                s = e.get_predicate_as_str()
            G.edge(str(e.source.id), str(e.target.id), s, fontsize="8")
        G.render(name)
    def _check_consistency(self):
        # coming
        comings: Dict[CFGNode, Set[CFGEdge]] = {x: set() for x in self.nodes}
        for e in self.edges:
            comings[e.target].add(e)
        for n in self.nodes:
            assert comings[n] == n.coming
        # going
        goings: Dict[CFGNode, Set[CFGEdge]] = {x: set() for x in self.nodes}
        for e in self.edges:
            goings[e.source].add(e)
        for n in self.nodes:
            assert goings[n] == n.going
        # edge -> node -> edge consitency
        for e in self.edges:
            assert e in e.source.going
            assert e in e.target.coming
        # node -> edge -> node consistency
        for n in self.nodes:
            for e in n.going:
                assert n == e.source
            for e in n.coming:
                assert n == e.target
    def _make_pred_edge(self, cfg_edge: CFGEdge, theory: str, bitwidth: int) -> Tuple[util.Tree, List[str]]:
        """Converts the operations in C to LIA formulae."""
        cfg_ = self
        vars = []
        unary_trans = {"!": "not"}
        binary_trans = {"==": "=", "||": "or", "&&": "and", "/": "div"}
        def trans(x: str, trans: Dict[str, str]):
            if x in trans:
                return trans[x]
            return x
        integers = ["int", "unsigned"]
        type2bits = {"int": bitwidth, "unsigned": bitwidth}
        var2ctype = dict({x.name: x.typ for x in cfg_.vars})
        def get_type(x: Any) -> str:
            if x.val[0] == "Constant":
                if x.val[2] in type2bits:
                    return "int"
                    # return x.val[2]
                elif x.val[2] == "bool":
                    return "bool"
                else:
                    raise Exception(f"Unsupported type {x.val[2]}")
            elif x.val[0] == "Var":
                varname = x.val[1]
                return var2ctype[varname]
            elif x.val[0] == "Nondet":
                return "bool"
            elif x.val[0] == "NondetInt":
                return "int"
            elif x.val[0] == "NondetUnsigned":
                return "unsigned"
            elif x.val[0] == "Unary":
                ty = get_type(x.val[2])
                if x.val[1] == "!":
                    return "bool"
                elif x.val[1] == "-":
                    return "int"
                else:
                    assert False
            elif x.val[0] == "Binary":
                # temporarily
                lt = get_type(x.val[2])
                rt = get_type(x.val[3])
                if lt == "Var":
                    lt = var2ctype[x.val[2][1]]
                if rt == "Var":
                    rt = var2ctype[x.val[3][1]]
                if x.val[1] in ["==", "!="]:
                    if lt == rt or (lt in integers and rt in integers):
                        pass
                    else:
                        raise Exception("Type mismatch")
                    return "bool"
                elif x.val[1] in ["<=", "<", ">", ">="]:
                    if lt in integers and rt in integers:
                        pass
                    else:
                        raise Exception("Values have to be int in inequalities")
                    return "bool"
                elif x.val[1] in ["||", "&&"]:
                    return "bool"
                elif x.val[1] in ["+", "*", "-", "/", "%"]:
                    if lt in integers and rt in integers:
                        pass
                    else:
                        raise Exception("Values have to be int")
                    return "int"
                else:
                    raise f"Unexpected operator {x.val[1]}"
            else:
                assert False
        def int_to_bool(t) -> Any:
            return ["not", ["=", t, 0]]
        def bool_to_int(t) -> Any:
            tag = t[0]
            if type(t) is str and treeutil.undecorate(t)[0] in var2ctype:
                return t
            elif type(t) is list:
                if tag in ["true", True]:
                    assert len(t) == 1
                    return 1 
                elif tag in ["false", False]:
                    assert len(t) == 1
                    return 0
                elif tag == "not":
                    assert len(t) == 2
                    return ["-", 1, bool_to_int(t[1])]
                elif tag == "=":
                    assert len(t) == 3
                    if t[2] in ["0", 0]:
                        return ["-", 1, bool_to_int(t[1])]
                    elif t[1] in ["0", 0]:
                        return ["-", 1, bool_to_int(t[2])]
                    else:
                        assert False
                else:
                    raise Exception(f"Unsupported boolean assignment: {t}")
            else:
                assert False
        def ite(cond: Any, t: Any, f: Any) -> Any:
            return ["ite", cond, t, f]
            # return ["and", ["=>", cond, t], ["=>", ["not", cond], f]]
        def helper(x: Any) -> Any:
            if x.val[0] == "Constant":
                if x.val[2] in ["bool"] + integers:
                    if util.re_hex.match(x.val[1]) is not None:
                        return int(x.val[1], 16)
                    elif x.val[1].lstrip("+-").isnumeric():
                        return int(x.val[1])
                    elif x.val[1] in ["true", "false"]:
                        return x.val[1]
                    else:
                        assert False
                else:
                    assert False
            elif x.val[0] == "Var":
                v = treeutil.decorate(x.val[1], 0)
                vars.append(v)
                return v
            elif x.val[0] == "NondetInt":
                return "*nondetint"
            elif x.val[0] == "NondetUnsigned":
                return "*nondetunsigned"
            elif x.val[0] == "Nondet":
                return "true"
            elif x.val[0] == "Unary":
                op = trans(x.val[1], unary_trans)
                if op in ["-", "not"]:
                    ty = get_type(x.val[2])
                    if op == "not":
                        if ty in integers:
                            return ["not", int_to_bool(helper(x.val[2]))]
                        elif ty == "bool":
                            return ["not", helper(x.val[2])]
                        else:
                            assert False
                    elif op == "-":
                        return [op, helper(x.val[2])]
                    else:
                        assert False
                else:
                    assert False
                    logging.warning("Unexpected unary symbol " + str(op))
                    return [op, helper(x.val[2])]
            elif x.val[0] == "Binary":
                if x.val[1] in ["+", "-", "*", "div", "<=", ">=", "<", ">", "=="] or x.val[1] in binary_trans:
                    left_sub = helper(x.val[2])
                    right_sub = helper(x.val[3])
                    op = binary_trans.get(x.val[1], x.val[1])
                    if op in ["and", "or"]:
                        left_sub = int_to_bool(left_sub) if get_type(x.val[2]) in integers else left_sub
                        right_sub = int_to_bool(right_sub) if get_type(x.val[3]) in integers else right_sub
                    return [op, left_sub, right_sub]
                elif x.val[1] == "!=":
                    return ["not", ["=", helper(x.val[2]), helper(x.val[3])]]
                elif x.val[1] == "%":
                    return ["mod", helper(x.val[2]), helper(x.val[3])]
                else:
                    assert False
                    logging.warning("Unexpected symbol " + str(x.val[1]))
                    return [x.val[1], helper(x.val[2]), helper(x.val[3])]
            else:
                raise Exception("Unexpected predicate " + str(x))
        slist_cond = helper(cfg_edge.condition)

        if get_type(cfg_edge.condition) in integers:
            slist_cond = int_to_bool(slist_cond)
        if cfg_edge.operation is None:
            res = slist_cond
        else:
            if isinstance(cfg_edge.operation, OperationBin):
                slist_upd_rhs = helper(cfg_edge.operation.rhs)
                var_upd_lhs = treeutil.decorate(cfg_edge.operation.lhs, 1)
                type_left = var2ctype[cfg_edge.operation.lhs] 
                type_right = get_type(cfg_edge.operation.rhs)
                if type_left in integers and type_right in integers:
                    pass
                elif type_left in integers and type_right == "bool":
                    slist_upd_rhs = bool_to_int(slist_upd_rhs)
                else:
                    raise Exception("Mismatched type in assignment")
                vars.append(var_upd_lhs)
                if cfg_edge.operation.op == "=":
                    slist_upd = ["=", var_upd_lhs, slist_upd_rhs]
                elif cfg_edge.operation.op in ["+=", "-=", "*="]:
                    var_cur_lhs = treeutil.decorate(cfg_edge.operation.lhs, 0)
                    slist_upd = ["=", var_upd_lhs, [cfg_edge.operation.op[0], var_cur_lhs, slist_upd_rhs]]
                else:
                    raise Exception(f"Unexpected type of assignment {cfg_edge.operation.op}")
            elif isinstance(cfg_edge.operation, OperationUni):
                slist_upd_rhs = treeutil.decorate(cfg_edge.operation.v, 0)
                var_upd_lhs = treeutil.decorate(cfg_edge.operation.v, 1)
                if cfg_edge.operation.op == "p++":
                    slist_upd = ["=", var_upd_lhs, ["+", slist_upd_rhs, 1]]
                elif cfg_edge.operation.op == "p--":
                    slist_upd = ["=", var_upd_lhs, ["-", slist_upd_rhs, 1]]
                elif cfg_edge.operation.op == "++":
                    slist_upd = ["=", var_upd_lhs, ["+", slist_upd_rhs, 1]]
                elif cfg_edge.operation.op == "--":
                    slist_upd = ["=", var_upd_lhs, ["-", slist_upd_rhs, 1]]
                else:
                    raise Exception(f"Non supported op {cfg_edge.operation.op}")
            else:
                raise Exception(f"Unspported Operation {cfg_edge.operation}")
            res = ["and", slist_upd, slist_cond]
        assert res != []
        if theory == "liabv":
            replaced = treeutil.replace_symbols_for_bv(res, 2 ** bitwidth, "depends", var2ctype)
        elif theory == "lia":
            replaced = res
        else:
            assert False
        return replaced, vars
    def make_predicates(self, theory: str, bitwidth: int):
        for e in self.edges:
            p, v = self._make_pred_edge(e, theory, bitwidth)
            e.pred = p
            e.vars = v




def join_CFG(
    cfg1: ControlFlowGraph,
    cfg2: ControlFlowGraph,
    source_: Optional[CFGNode] = None,
    target_: Optional[CFGNode] = None,
    cond: CFGValue = None,
    goto_dict: Dict[str, GotoManager] = None,
    returns: Set[CFGNode] = None) -> ControlFlowGraph:
    # process source and target
    source = source_ if source_ is not None else cfg1.exit
    target = target_ if target_ is not None else cfg2.initial
    # stop if cfg1 is dying
    # if cfg1.exit == cfg1.final:
    #     return cfg1
    # is merge mode on?
    if cond is None:
        merge_mode = True
        cond = CFGValue()
    else:
        merge_mode = False
    # disable merging if it is related to jump
    for k, v in goto_dict.items():
        if target in v.sources:
            merge_mode = False
            break
        if target == v.target:
            merge_mode = False
            break
    if target in returns:
        merge_mode = False
    merge_mode = merge_mode and (not FORCE_NOMERGE)
    # move nodes from cfg2 to cfg1 (rename ids)
    # cfg1.nodes.append(cfg2.nodes[0])
    # cfg1.nodes[-1].id = len(cfg1.nodes) - 1
    # assert cfg1.nodes[cfg1.nodes[-1].id] == cfg1.nodes[-1]
    for n in cfg2.nodes: # exclude cfg2's final
        if merge_mode and n == target:
            logging.debug(f"skipping {n.id}")
            continue
        if n == cfg2.final:
            continue
        if n in cfg1.nodes:
            continue
        cfg1.nodes.append(n)
        cfg1.nodes[-1].id = len(cfg1.nodes) - 1
        assert cfg1.nodes[cfg1.nodes[-1].id] == cfg1.nodes[-1]
    # move edges from cfg2 to cfg1
    for e in cfg2.edges:
        if e in cfg1.edges:
            continue
        e.id = len(cfg1.edges)
        cfg1.edges.append(e)
    # convert the edges of (cfg2's init)->* to (cfg1's exit)->* preserving conditions
    # for e in cfg2.edges:
    #     if e.source == cfg2.initial:
    #         e.source = cfg1.exit
    #         cfg1.exit.coming.add(e)
    if merge_mode:
        for e in cfg2.edges:
            if e.source == target:
                e.source = source
                source.going.add(e)
            if e.target == target:
                e.target = source
                source.coming.add(e)
    else:
        # connect from source to target
        cfg1.add_edge(source, target, cond, None)
    # merge cfg2's final into cfg1's final
    for e in cfg2.edges:
        if e.target == cfg2.final:
            e.target = cfg1.final
            cfg1.final.coming.add(e)
    # change exit
    if merge_mode and target == cfg2.exit:
        pass
    else:
        if cfg2.exit == cfg2.final:
            cfg1.exit = cfg1.final
        else:
            cfg1.exit = cfg2.exit
    # merge variables
    cfg1.vars.update(cfg2.vars)
    # assertion
    if cfg2.final == cfg2.exit:
        assert cfg1.final == cfg1.exit
    # change gotos
    if merge_mode:
        for k, v in goto_dict.items():
            v.swap(target, source)
    # change returns
    if merge_mode and target in returns:
        returns.remove(target)
        returns.add(source)
    return cfg1

def join_CFGs(cfgs: List[Optional[ControlFlowGraph]], goto_dict: Dict[str, GotoManager], returns: Set[CFGNode]) -> ControlFlowGraph:
    cfgs1: List[ControlFlowGraph] = [c for c in cfgs if c is not None]
    cfg = cfgs1[0]
    for c in cfgs1[1:]:
        cfg = join_CFG(cfg, c, None, None, None, goto_dict, returns)
    return cfg

def parse_exp(p: pycparser.c_ast.Node, pre: str = "") -> CFGValue:
    if type(p) is pycparser.c_ast.Constant:
        return CFGValue(("Constant", p.value, p.type))
    elif type(p) is pycparser.c_ast.ID:
        return CFGValue(("Var", pre + p.name))
    elif type(p) is pycparser.c_ast.UnaryOp:
        u: pycparser.c_ast.UnaryOp = p
        ex = parse_exp(u.expr, pre)
        if ex.val[0] == "Nondet":
            return CFGValue(("Nondet", ))
        return CFGValue(("Unary", u.op, ex))
    elif type(p) is pycparser.c_ast.BinaryOp:
        b: pycparser.c_ast.BinaryOp = p
        left = parse_exp(b.left, pre)
        right = parse_exp(b.right, pre)
        if left.val[0] == "Nondet" or left.val[0] == "Nondet":
            return CFGValue(("Nondet",))
        return CFGValue(("Binary", b.op, left, right))
    elif type(p) is pycparser.c_ast.FuncCall:
        fc: pycparser.c_ast.FuncCall = p
        if is_nondet(fc.name.name):
            return CFGValue(("NondetInt", ))
        elif is_nondet_int(fc.name.name):
            return CFGValue(("NondetInt", ))
        elif is_nondet_unsigned(fc.name.name):
            return CFGValue(("NondetUnsigned", ))
        else:
            raise Exception(f"Unexpected function invoke: {fc}")
    elif p is None:
        return CFGValue(("Constant", "true", "bool"))
    else:
        raise Exception(f"Unexpected Operation: {p}")

class BreakManager:
    def __init__(self):
        self.cnt = 0
        self.stack = []
    def push(self):
        s = f"*dummy_break_{self.cnt}"
        self.cnt += 1
        self.stack.append(s)
        return s
    def pop(self):
        return self.stack.pop()
    def get_top(self):
        return self.stack[-1]

def construct_CFG_help(p: pycparser.c_ast.Node, pre: str, goto_dict: Dict[str, GotoManager], returns: Set[CFGNode], stack_break: BreakManager) -> Optional[ControlFlowGraph]:
    def process_function(p: pycparser.c_ast.Node, pre: str) -> Optional[ControlFlowGraph]:
        if is_ignoring_function_name(p.name):
            return None
        elif p.name == "main":
            if type(p) is pycparser.c_ast.FuncDef:
                fd: pycparser.c_ast.FuncDef = cast(pycparser.c_ast.FuncDef, p)
                if type(fd.body) is pycparser.c_ast.Compound:
                    comp: pycparser.c_ast.Compound = fd.body
                    cfgs = [construct_CFG_help(c, pre, goto_dict, returns, stack_break) for c in comp.block_items]
                    return join_CFGs(cfgs, goto_dict, returns)
                else:
                    assert False
            else:
                raise Exception("Unexcepted type for main")
            return None
        else:
            raise Exception(f"Unexpected function definition {p.name}")
    # cfg = ControlFlowGraph()
    if type(p) is pycparser.c_ast.FileAST:
        cs = [c[1] for c in p.children()]
        cfgs = [construct_CFG_help(c, pre, goto_dict, returns, stack_break) for c in cs]
        return join_CFGs(cfgs, goto_dict, returns)
    elif type(p) is pycparser.c_ast.FuncDef:
        fd: pycparser.c_ast.FuncDef = cast(pycparser.c_ast.FuncDef, p)
        logging.debug(f"FuncDef {fd.decl.name}")
        if fd.decl.name == "main":
            if type(fd.body) is pycparser.c_ast.Compound:
                if hasattr(fd.decl.type.args, "params"):
                    decls = [construct_CFG_help(d, pre, goto_dict, returns, stack_break) for d in fd.decl.type.args.params]
                else:
                    decls = []
                comp: pycparser.c_ast.Compound = fd.body
                cfgs = [construct_CFG_help(c, pre, goto_dict, returns, stack_break) for c in comp.block_items]
                return join_CFGs(decls + cfgs, goto_dict, returns)
        elif is_ignoring_function_name(fd.decl.name):
            return None
        raise Exception("Unexpected Function Definition/" + str(p))
        # return process_function(fd, pre)
    elif type(p) is pycparser.c_ast.Decl:
        d: pycparser.c_ast.Decl = p
        if type(d.type) is pycparser.c_ast.FuncDecl:
            logging.debug(f"Decl/FuncDecl {d.name}")
            return process_function(d, pre)
        elif type(d.type) is pycparser.c_ast.TypeDecl:
            logging.debug(f"Decl/TypeDecl {d.name}")
            cfg = ControlFlowGraph()
            if d.init is None:
                typename = d.type.type.names[0]
                if typename == "int":
                    # var = CFGVar(pre + "*nondetint", "int")  #  TODO: Is it needed?
                    # cfg.vars.add(var)
                    val = CFGValue(("NondetInt", ))
                    # val = CFGValue(("Constant", "0", "int"))
                elif typename == "unsigned":
                    val = CFGValue(("NondetUnsigned", ))
                else:
                    raise Exception("Unexpected type name")
            else:
                val = parse_exp(d.init, pre)
            var = CFGVar(pre + d.name, d.type.type.names[0])
            cfg.vars.add(var)
            ex = cfg.add_edge(cfg.initial, None, CFGValue(), OperationBin(var.name, val, "="))
            cfg.exit = ex[1]
            return cfg
        else:
            raise Exception("Unexpected type of declaration/" + str(d))
    elif type(p) is pycparser.c_ast.While:
        logging.debug("While")
        w: pycparser.c_ast.While = p
        label_break = stack_break.push()
        gm = GotoManager()
        goto_dict[label_break] = gm
        cond = parse_exp(w.cond, pre) # Here assignment cannot come here
        body_cfg = construct_CFG_help(w.stmt, pre, goto_dict, returns, stack_break)
        stack_break.pop()
        body_cfg = body_cfg if body_cfg is not None else ControlFlowGraph()
        if cond.val[0] == "Nondet":
            cond_in = CFGValue()
            cond_out = CFGValue()
        else:
            cond_in = cond
            cond_out = CFGValue(("Unary", "!", cond))
        cfg = ControlFlowGraph()
        exit_node = cfg.add_edge(cfg.initial, None, cond_out, None)[1] # from branching to exit
        gm.target = exit_node
        branch_node = cfg.initial
        cfg = join_CFG(cfg, body_cfg, branch_node, body_cfg.initial, cond_in, goto_dict, returns) #from branching to main
        cfg.add_edge(cfg.exit, branch_node, CFGValue(), None) # from the bottom to top
        cfg.exit = exit_node
        return cfg
    elif type(p) is pycparser.c_ast.If:
        logging.debug("If")
        i: pycparser.c_ast.If = p
        cond = parse_exp(i.cond, pre)
        if cond.val[0] == "Nondet":
            cond_if = CFGValue()
            cond_else = CFGValue()
        else:
            cond_if = cond
            cond_else = CFGValue(("Unary", "!", cond))
        if_cfg = construct_CFG_help(i.iftrue, pre, goto_dict, returns, stack_break)
        if_cfg = if_cfg if if_cfg is not None else ControlFlowGraph()
        cfg = ControlFlowGraph()
        branch_node = cfg.initial
        cfg = join_CFG(cfg, if_cfg, branch_node, if_cfg.initial, cond_if, goto_dict, returns)
        if cfg.final == cfg.exit:
            if_dying = True
        else:
            if_dying = False
            exit_node = cfg.add_edge(cfg.exit, None, CFGValue(), None)[1]
            cfg.exit = exit_node
        if i.iffalse is None:
            if if_dying:
                exit_node = cfg.add_edge(branch_node, None, cond_else, None)[1]
                cfg.exit = exit_node
            else:
                cfg.add_edge(branch_node, exit_node, cond_else, None)
        else:
            else_cfg = construct_CFG_help(i.iffalse, pre, goto_dict, returns, stack_break)
            else_cfg = else_cfg if else_cfg is not None else ControlFlowGraph()
            cfg = join_CFG(cfg, else_cfg, branch_node, else_cfg.initial, cond_else, goto_dict, returns)
            if if_dying:
                if else_cfg.final == else_cfg.exit:
                    cfg.exit = cfg.final
                else:
                    exit_node = cfg.add_edge(cfg.exit, None, CFGValue(), None)[1]
                    cfg.exit = exit_node
            else:
                cfg.add_edge(cfg.exit, exit_node, CFGValue(), None)
                cfg.exit = exit_node
        return cfg
    elif type(p) is pycparser.c_ast.Assignment:
        a: pycparser.c_ast.Assignment = p
        logging.debug(f"Assigning {a.lvalue.name}")
        varname = pre + a.lvalue.name
        val = parse_exp(a.rvalue)
        cfg = ControlFlowGraph()
        ex = cfg.add_edge(cfg.initial, None, CFGValue(), OperationBin(varname, val, a.op))
        cfg.exit = ex[1]
        return cfg
    elif type(p) is pycparser.c_ast.FuncCall:
        fc: pycparser.c_ast.FuncCall = p
        logging.debug(f"FuncCall {fc.name.name}")
        if is_error_function(fc.name.name):
            cfg = ControlFlowGraph()
            cfg.add_edge(cfg.initial, cfg.final, CFGValue(), None)
            cfg.exit = cfg.final
            logging.log(logging.DEBUG, "added edge")
            return cfg
        elif is_assumption(fc.name.name):
            cond = parse_exp(fc.args.exprs[0], pre)
            cfg = ControlFlowGraph()
            ex = cfg.add_edge(cfg.initial, None, cond, None)
            cfg.exit = ex[1]
            return cfg
        elif is_assertion(fc.name.name):
            cond = parse_exp(fc.args.exprs[0], pre)
            cond_neg = CFGValue(("Unary", "!", cond))
            cfg = ControlFlowGraph()
            ex = cfg.add_edge(cfg.initial, None, cond, None)
            cfg.add_edge(cfg.initial, cfg.final, cond_neg, None)
            cfg.exit = ex[1]
            return cfg
        else:
            raise Exception(f"Unexpected function call {fc.name}")
    elif type(p) is pycparser.c_ast.Compound:
        comp = p
        if comp.block_items is None:
            return ControlFlowGraph()
        else:
            cfgs = [construct_CFG_help(b, pre, goto_dict, returns, stack_break) for b in comp.block_items]
            return join_CFGs(cfgs, goto_dict, returns)
    elif type(p) is pycparser.c_ast.Goto:
        cfg = ControlFlowGraph()
        if p.name not in goto_dict:
            goto_dict[p.name] = GotoManager()
        goto_dict[p.name].sources.add(cfg.initial)
        print("added source", goto_dict[p.name], goto_dict[p.name].sources)
        return cfg
    elif type(p) is pycparser.c_ast.Label:
        name = p.name
        stmt = p.stmt

        cfg_main = construct_CFG_help(stmt, pre, goto_dict, returns, stack_break)
        if name not in goto_dict:
            goto_dict[name] = GotoManager()
        goto_dict[name].target = cfg_main.initial

        return cfg_main
    elif type(p) is pycparser.c_ast.Return:
        c = ControlFlowGraph()
        returns.add(c.initial)
        return c  # TODO: it can be fixed when it needs subproceedure
    elif type(p) is pycparser.c_ast.UnaryOp:
        cfg = ControlFlowGraph()
        varname = p.expr.name
        ex = cfg.add_edge(cfg.initial, None, CFGValue(), OperationUni(varname, p.op))
        cfg.exit = ex[1]
        return cfg
    elif type(p) is pycparser.c_ast.For:
        init = construct_CFG_help(p.init, pre, goto_dict, returns, stack_break)
        cond = parse_exp(p.cond, pre)
        cfg_next = construct_CFG_help(p.next, pre, goto_dict, returns, stack_break)

        if cond.val[0] == "Nondet":
            cond_in = CFGValue()
            cond_out = CFGValue()
        else:
            cond_in = cond
            cond_out = CFGValue(("Unary", "!", cond))
        
        exit_node = init.add_edge(init.exit, None, cond_out, None)[1] # from branching to exit
        branch_node = init.exit
        label_break = stack_break.push()
        gm = GotoManager()
        gm.target = exit_node
        goto_dict[label_break] = gm
        body_cfg = construct_CFG_help(p.stmt, pre, goto_dict, returns, stack_break)
        stack_break.pop()
        body_cfg = join_CFG(body_cfg, cfg_next, body_cfg.exit, cfg_next.initial, CFGValue(), goto_dict, returns)
        # body_cfg.add_edge(body_cfg.exit, body_cfg.initial, CFGValue(), None)
        cfg = join_CFG(init, body_cfg, branch_node, body_cfg.initial, cond_in, goto_dict, returns) #from branching to main
        cfg.add_edge(cfg.exit, branch_node, CFGValue(), None) # from the bottom to top
        cfg.exit = exit_node
        return cfg
    elif type(p) is pycparser.c_ast.EmptyStatement \
        or p is None \
        or (type(p) is pycparser.c_ast.Typename and p.name is None):
        return ControlFlowGraph()
    elif type(p) is pycparser.c_ast.Break:
        cfg = ControlFlowGraph()
        goto_dict[stack_break.get_top()].sources.add(cfg.initial)
        print("added source (break)", goto_dict[stack_break.get_top()], goto_dict[stack_break.get_top()].sources)
        return cfg
    else:
        raise Exception("Unexpected type " + str(p))
    raise Exception("Forgot returning somewhere")

def connect_goto(res, goto_dict):
    for k, v in goto_dict.items():
        print(k, v.sources, v.target)
        assert v.target in res.nodes
        for s in v.sources:
            assert s in res.nodes
            edges = s.going
            while len(edges) > 0:
                e = next(edges.__iter__())
                res.remove_edge(e)
            if k == "ERROR":
                tgt = res.final
            else:
                tgt = v.target
            res.add_edge(s, tgt, CFGValue(), None)
    return res

def connect_returns(res, returns):
    for r in returns:
        assert r in res.nodes
        edges = r.going
        while len(edges) > 0:
            e = next(edges.__iter__())
            res.remove_edge(e)
        if r != res.exit and r != res.final:
            res.add_edge(r, res.exit, CFGValue(), None)
    return res

def construct_CFG(p: pycparser.c_ast.Node, mode: str, bitwidth: int, pre: str = "") -> ControlFlowGraph:
    goto_dict = {}
    returns = set()
    stack_break = BreakManager()
    res = construct_CFG_help(p, pre, goto_dict, returns, stack_break)
    returns.add(res.final)
    print("returns", returns)
    res = connect_returns(res, returns)
    res = connect_goto(res, goto_dict)
    print("goto_dict:", goto_dict)
    if res is None:
        assert False
    else:
        res.make_predicates(mode, bitwidth)
        return res


def main():
    # x = pycparser.parse_file(sys.argv[1])
    if len(sys.argv) > 1:
        x = pycparser.parse_file(sys.argv[1])
    else:
        x = pycparser.parse_file("01xx.c")

    

    # with open(sys.argv[1]) as f:
    #     text = f.read()
    # print(text)
    
    # parser = c_parser.CParser()
    # ast = parser.parse(text)

    # print(dir(ast))
    # print(ast)
    x.show()
    cfg = construct_CFG(x)
    cfg.print()

    cfg._check_consistency()

if __name__ == "__main__":
    main()