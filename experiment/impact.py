import CFG
from typing import *
import smtutil
import logging

import treeutil
import util

logging.basicConfig(level=logging.DEBUG)
import pycparser
import sys
from graphviz import Digraph
import simplifier
import argparse
import copy
import gc
import shutil
import os
from util import debug_print_list, debug_print_preds
# import tracemalloc
# import objgraph
from memory_profiler import profile
import time
import depict
import subprocess
import statistics

fp = open("out/memory.log", "w+")
filename_interpols = "interpols.txt"

def debug_write_interpolants(predsbv, predslia, prob, comment):
    with open("out/" + filename_interpols, "a") as f:
        f.write(comment + "\n")
        f.write("[Interpolation problem]\n")
        for p in prob:
            f.write(debug_print_list(simplifier.erase_obvious_connectives(p)) + "\n")
        f.write("[Interpolants in LIA]\n")
        for p in predslia:
            f.write(debug_print_list(simplifier.erase_obvious_connectives(p)) + "\n")
        f.write("[Interpolants in BV]\n")
        for p in predsbv:
            f.write(debug_print_list(simplifier.erase_obvious_connectives(p)) + "\n")
        f.write("-"*80 + "\n")


class Configuration():
    def __init__(self, theory: str, bitwidth: int, simplification: bool=False, 
        floor_reduce="strategy1",
        expand_floor_in_inteprolation_error = False,
        disable_euf = True,
                 lia2bv = "boxing"):
        self.theory: str = theory
        self.bitwidth: int = bitwidth
        self.depth_forcecover = 0 # 0 means disabled
        self.simplification = simplification
        self.smtutilopt = util.SMTUtilOption(floor_reduce=floor_reduce,
                                             expand_floor_in_inteprolation_error = expand_floor_in_inteprolation_error,
                                             disable_euf = disable_euf,
                                             lia2bv=lia2bv)
    def to_dict(self):
        return {"theory": self.theory,
        "bitwidth": self.bitwidth,
        "simplification": self.simplification,
        "smtutilopt": self.smtutilopt.to_dict()}
                                             


class UNode():
    def __init__(self,
    id_: int,
    cfg_node_: CFG.CFGNode,
    coming_: Optional["UEdge"]):
        self.id: int = id_
        self.cfg_node: CFG.CFGNode = cfg_node_
        self.label: util.Tree = "true"
        self.coming: Optional["UEdge"] = coming_
        self.going: Set[UEdge] = set()
        self.priority = 0
    def get_info(self) -> str:
        if self.cfg_node.id == 0:
            name = "ENTRY"
        elif self.cfg_node.id == 1:
            name = "ERROR"
        else:
            name = f"N{self.cfg_node.id}"
        return f"{self.id}/{name}"

class UEdge():
    def __init__(self, 
    id_: int,
    src_: UNode,
    tgt_: UNode,
    cfg_edge_: CFG.CFGEdge,
    cfg_: CFG.ControlFlowGraph,
    config_: Configuration):
        self.id: int = id_
        self.src: UNode = src_
        self.tgt: UNode = tgt_
        self.cfg_edge: CFG.CFGEdge = cfg_edge_
        assert src_.cfg_node == cfg_edge_.source
        assert tgt_.cfg_node == cfg_edge_.target
        self.config: Configuration = config_
        self.pred, self.vars = self.make_pred(cfg_)
        assert True

    def make_pred(self, cfg_: CFG.ControlFlowGraph) -> Tuple[util.Tree, List[str]]:
        return self.cfg_edge.pred, self.cfg_edge.vars

ctype2smttype: Dict[str, str] = {"int": "Int", "unsigned": "Int"}


class Unwinding():
    def __init__(self, cfg_: CFG.ControlFlowGraph, config_: Configuration, stat: statistics.Statistics):
        self.cfg: CFG.ControlFlowGraph = cfg_
        self.vars_dict: Dict[str, str] = {treeutil.decorate(k, 0): ctype2smttype[v] for k, v in
                                          self.cfg.get_vars_as_dict().items()}  # have to be decolrated
        self.edges: List[UEdge] = []
        self.root: UNode = UNode(0, self.cfg.nodes[0], None)
        self.nodes: List[UNode] = [self.root]
        self.coverings: Set[Tuple[UNode, UNode]] = set()
        self.covereds: Set[UNode] = set() # has to be propagated
        self.unexpandeds: Set[UNode] = set([self.root]) # has to be maintained
        self.updated_covering: bool = False
        self.config: Configuration = config_
        self.cfgnode2node: Dict[CFG.CFGNode, List[UNode]] = {n: [] for n in self.cfg.nodes}
        self.error_trace: Optional[List[UEdge]] = None
        self.stat = stat
        self.strengthened: Set[UNode] = set()
    def get_parents(self, n: UNode) -> List[UNode]:
        res = []
        while True:
            res.append(n)
            if n.coming is None:
                break
            else:
                n = n.coming.src
        res.reverse()
        return res
    def get_children(self, n: UNode) -> Set[UNode]:
        res = set([n])
        for e in n.going:
            res = res.union(self.get_children(e.tgt))
        return res
    def _maintain_covereds(self):
        # self._assert_valid_covering()
        self.updated_covering = True
        res = set()
        for s, _ in self.coverings:
            res = res.union(self.get_children(s))
        self.covereds = res
        # self._assert_no_double_covering()
    def get_path(self, v: UNode) -> List[UEdge]:
        res: List[UEdge] = []
        while True:
            e = v.coming
            if e is None:
                break
            else:
                res.append(e)
                v = e.src
        res.reverse()
        return res
    def add_edge(self, src_: UNode, tgt_: Optional[UNode], cfg_edge_: CFG.CFGEdge) -> Tuple[UNode, UEdge]:
        src = src_
        if tgt_ is None:
            tgt = UNode(len(self.nodes), cfg_edge_.target, None)
            self.nodes.append(tgt)
        else:
            tgt = tgt_
        e = UEdge(len(self.edges), src, tgt, cfg_edge_, self.cfg, self.config)
        self.edges.append(e)
        src.going.add(e)
        tgt.coming = e
        return tgt, e
    def expand(self, v: UNode):
        logging.debug(f"Entering expanding at {v.cfg_node.id} at CFG")
        if not(v in self.covereds) and v in self.unexpandeds:
            self.unexpandeds.remove(v)
            logging.debug(f"Expanding at {v.cfg_node.id} at CFG")
            for a in v.cfg_node.going:
                v_new, e_new = self.add_edge(v, None, a)
                self.unexpandeds.add(v_new)
                self.cfgnode2node[v_new.cfg_node].append(v_new)
            self.print(debug=True)

    def reindex_preds(self, preds: List[util.Tree]) -> List[util.Tree]:
        if len(preds) == 0:
            return []
        res = []
        current_indices: Dict[str, int] = {}
        for pred in preds:
            res.append(treeutil.shift_index_tree_dict_immutable(pred, current_indices))
            vars = treeutil.get_variables(res[-1])
            for var in vars:
                undec, ind = treeutil.undecorate(var)
                if undec.startswith("*"):
                    current_indices[undec] = current_indices.get(undec, 0) + 1
                else:
                    current_indices[undec] = max(current_indices.get(undec, 0), ind)
        return res
    def replace_nondet(self, preds):
        class ReplaceInt:
            def __init__(self):
                self.ind = 0
            def __call__(self, s):
                res = s
                if s in ["*nondetint", "*nondetunsigned"]:
                    res = f"var_0_dummy_nondetint_{self.ind}"
                    self.ind += 1
                return res
        r = ReplaceInt()
        preds = [treeutil.walk_replace_immutable(p, r) for p in preds]
        return preds
    # @profile(stream=fp)
    def refine(self, v: UNode) -> bool: # False means unsafe
        logging.debug("Refining")
        if self.config.theory == "liabv":
            sat_result = smtutil.check_sat_bv(v.label, 2 ** self.config.bitwidth, self.stat)
        elif self.config.theory == "lia":
            sat_result = smtutil.check_sat_lia(v.label, self.stat)
        else:
            assert False
        if v.cfg_node == self.cfg.final and sat_result: # here vars can be undecorated
            pi = self.get_path(v)
            parents = self.get_parents(v)
            # pi_pred = [smtutil.sexp_and(pi1.pred, pi1.src.label) for i, pi1 in enumerate(pi)]  # TODO: add variable constraint here!
            pi_pred = [pi1.pred for i, pi1 in enumerate(pi)]
            pi_pred = self.reindex_preds(pi_pred)
            # TODO: koko ni nondetint no are wo irru
            pi_pred = self.replace_nondet(pi_pred)
            # shifted_vars: Dict[str, str] = {}
            # for pi_pred_i in pi_pred:
            #     for v1 in treeutil.get_variables(pi_pred_i):
            #         shifted_vars[v1] = self.vars_dict[treeutil.set_index(v1, 0)]
            if self.config.theory == "liabv":
                interp, interp_lia, prob_lia = smtutil.get_interpolants_bv(pi_pred, 2 ** self.config.bitwidth, self.config.smtutilopt, self.stat)
                debug_write_interpolants(interp, interp_lia, prob_lia, f"Path {[x.id for x in pi]}")
                for b, l in zip(interp, interp_lia):
                    pass
                    # smtutil.assert_lia_processing_ok(l, 2**self.config.bitwidth, self.config.smtutilopt)
                    # smtutil.assert_equal_lia_and_processed_lia(l, 2**self.config.bitwidth, self.config.smtutilopt)
                    # smtutil.assert_equal_lia_and_bv(b, l, 2**self.config.bitwidth, self.config.smtutilopt)
            elif self.config.theory == "lia":
                interp = smtutil.get_interpolants_lia(pi_pred, self.config.smtutilopt, self.stat)
                debug_write_interpolants(interp, pi_pred, [], f"Path {[x.id for x in pi]}")
            else:
                assert False
            self.stat.sizes_interpol.append([util.count_terms(i) for i in interp])
            # logging.debug("Raw Interpolants:" + str("\n".join(interp)))
            # interp = [smtutil.list2sexp(smtutil.convert_lia2_liabv(sexpdata.loads(i), 2**self.bitwidth)) for i in interp]
            # logging.debug("Interpolants:" + str("\n".join(interp)))
            logging.debug(f"Path {[x.id for x in pi]}")
            self.print(debug=True)
            pass
            if interp == []:
                # Feasible
                self.stat.counter_path_id = [x.cfg_edge.id for x in pi]
                self.stat.counter_path_pred = pi_pred
                if self.config.theory == "liabv":
                    # print("original LIA interp", interp_lia)
                    print("problem LIA interp\n", util.debug_print_preds(prob_lia))
                    self.stat.counter_model = smtutil.get_model_bv(pi_pred, 2**self.config.bitwidth, self.stat)
                    # DEBUG
                    print("model in LIA", smtutil.get_model_lia(prob_lia, self.stat))
                    # print("model", smtutil.get_model_bv(pi_pred, list(shifted_vars.items()), 2**self.config.bitwidth))
                elif self.config.theory == "lia":
                    
                    self.stat.counter_model = smtutil.get_model_lia(pi_pred, self.stat)
                    # print(util.debug_print_preds  (pi_pred))
                    # print("model", )
                else:
                    assert False
                # self.stat.error_trace = pi
                print("Feasible!", self.stat.counter_model)
                return False
            else:
                for i, ahat in enumerate(interp):
                    phi = treeutil.set_index_tree_immutable(ahat, 0)
                    assert len(treeutil.get_variables(phi)) == 0 or any(
                        [treeutil.undecorate(i)[1] == 0 for i in treeutil.get_variables(phi)])
                    vi = parents[i]
                    s = ["=>", vi.label, phi]
                    vars_with_types = {var: self.vars_dict[treeutil.set_index(var, 0)] for var in
                                       treeutil.get_variables(s)}
                    logging.debug(f"{i}th phi: {phi}, ahat: {ahat}, vilabel: {vi.label}")
                    if self.config.theory == "liabv":
                        validity_res = smtutil.check_valid_bv(s, 2 ** self.config.bitwidth, self.stat)
                    elif self.config.theory == "lia":
                        validity_res = smtutil.check_valid_lia(s, self.stat)
                    else:
                        assert False
                    if not validity_res:
                        self.coverings = {(x, y) for x, y in self.coverings if not (y == vi)}
                        self._maintain_covereds()
                        s = ["=>", phi, vi.label]
                        if self.config.theory == "liabv":
                            validity_res = smtutil.check_valid_bv(s, 2 ** self.config.bitwidth, self.stat)
                        elif self.config.theory == "lia":
                            validity_res = smtutil.check_valid_lia(s, self.stat)
                        else:
                            assert False
                        if validity_res:
                            vi.label = phi
                        else:
                            vi.label = ["and", vi.label, phi]
                            self.strengthened.add(vi)
                        nonsimplified = vi.label
                        if self.config.simplification:
                            vi.label = simplifier.simplify_tree(vi.label, 2**self.config.bitwidth, self.config.theory)
                        # vi.label = simplifier.simplify_sexp(vi.label, 2**self.config.bitwidth, "liabv")
                        logging.debug(f"Refined, {vi.label}")

        return True
    def cover(self, v: UNode, w: UNode):
        logging.debug("Covering")
        # if len(w.cfg_node.going) == 0:  # no expansible nodes does not need cover
        #     return
        if not(w in self.covereds) and v.cfg_node.id == w.cfg_node.id and not(v in self.get_parents(w)):
            if self.config.theory == "liabv":
                validity_res = smtutil.check_valid_bv(["=>", v.label, w.label], 2 ** self.config.bitwidth, self.stat)
            elif self.config.theory == "lia":
                validity_res = smtutil.check_valid_lia(["=>", v.label, w.label], self.stat)
            else:
                assert False
            if validity_res:
                logging.debug(f"Established Covering from {v.label} to {w.label}")
                self.coverings.add((v, w))
                self.coverings = {(x, y) for x, y in self.coverings if not (v in self.get_parents(y))}
                self._maintain_covereds()
            self.print(debug=True)
    def _assert_no_double_covering(self):
        heads = {h for _, h in self.coverings}
        for c, _ in self.coverings:
            s = set(self.get_children(c)).intersection(heads)
            if set(self.get_children(c)).intersection(heads):
                logging.error(f"N{s.pop().id} is covering, but covered by N{c.id}")
                assert False
    def _assert_proper_covering(self):
        heads = {h for _, h in self.coverings}
        for f, t in self.coverings:
            if self.config.theory == "liabv":
                validity_res = smtutil.check_valid_bv(["=>", f.label, t.label], 2 ** self.config.bitwidth, self.stat)
                assert validity_res
            else:
                validity_res = smtutil.check_valid_lia(["=>", f.label, t.label], self.stat)
                assert validity_res
    def _assert_valid_covering(self):
        for f, _ in self.coverings:
            pi = self.get_path(f)
            pi_pred = [pi1.pred for i, pi1 in enumerate(pi)]
            pi_pred.append(f.label)
            pi_pred = self.reindex_preds(pi_pred)
            pi_pred = self.replace_nondet(pi_pred)
            p = ["=>", ["and"] + pi_pred[:-1], pi_pred[-1]]
            if self.config.theory == "liabv":
                res = smtutil.check_valid_bv(p, 2 ** self.config.bitwidth, self.stat)
                if res == False:
                    print("path", debug_print_preds(pi_pred[:-1]))
                    print("goal", debug_print_list(pi_pred[-1]))
                    assert False

   

    def close(self, v: UNode):
        logging.debug("Closing")
        for w in self.nodes:
            if w.id < v.id and w.cfg_node == v.cfg_node: # can be efficient by making a dictionary
                self.cover(v, w)
    # @profile(stream=fp)
    def dfs(self, v: UNode) -> bool:
        gc.collect()
        # objgraph.show_growth()
        logging.debug(f"DFS {v.id}@tree, {v.cfg_node.id}@cfg")
        self.close(v)
        if not(v in self.covereds):
            if v.cfg_node == self.cfg.final:
                res_refine = self.refine(v)
                if res_refine == False:
                    return False
                for w in self.get_parents(v):
                    self.close(w)
            if self.config.depth_forcecover > 0:
                res_forced = self.forced_cover_from(v)
                if res_forced:
                    return True
            self.expand(v)
            for e in v.going:
                res_dfs = self.dfs(e.tgt)
                if res_dfs == False:
                    return False
        return True
    # @profile(stream=fp)
    def unwind(self) -> Tuple[bool, statistics.Statistics]:
        while True:
            logging.debug("Unwinding")
            # if self.updated_covering:
            #     logging.debug("updated_covering is ON")
            #     self.updated_covering = False
            #     remaining = set(self.nodes) - self.covereds
            # else:
            #     logging.debug("updated_covering is OFF")
            #     remaining = self.unexpandeds - self.covereds
            remaining = self.unexpandeds.union(self.strengthened) - self.covereds
            logging.debug(f"{[x.cfg_node.id for x in remaining]}")
            if len(remaining) == 0:
                #DEBUG
                logging.debug("unexpandeds: " + str([x.id for x in self.unexpandeds]))
                logging.debug("covereds: " + str([x.id for x in self.covereds]))
                #DEBUG
                break
            # if len(uncovereds) == 0:
            #     break
            # v = remaining.pop() #Experimental
            v = util.pick_minimum_priority(remaining, lambda x: x.priority)
            v.priority += 1
            self.strengthened -= {v}
            # print("popped", v)
            for w in self.get_parents(v):
                if v == w:
                    continue
                self.close(w)
            res_dfs = self.dfs(v)
            if res_dfs == False:
                return False, self.stat
            self.print(debug=True)
            gc.collect()
            
            pass
        print("remaining:", self.unexpandeds - self.covereds)
        # self._assert_no_double_covering()
        # self._assert_proper_covering()
        # self._assert_valid_covering()
        assert not (self.unexpandeds - self.covereds)
        return True, self.stat

    def print(self, name="unwind", debug=False, draw=True, format="png"):
        if debug:
            logging.debug("The debug print of graph is omitted")
            return
        if draw:
            # usual
            G = Digraph(format=format)
            for n in self.nodes:
                # logging.debug(f"node from {n.id}")
                c = "(C)" if n in self.covereds else ""
                G.node(str(n.id), f"{n.get_info()}{c}")
            for e in self.edges:
                # logging.debug(f"edge from {e.src.id} to {e.tgt.id}")
                if self.error_trace is not None and e in self.error_trace:
                    color = "red"
                else:
                    color = "black"
                G.edge(str(e.src.id), str(e.tgt.id), str(e.id), color=color)
            for covered, covering in self.coverings:
                G.edge(str(covered.id), str(covering.id), style="dotted")
            G.render("out/" + name)

            # covereds are dropped
            G = Digraph(format=format)
            droppeds = set()
            for n in self.nodes:
                if n in self.covereds and n.coming.src in self.covereds:
                    droppeds.add(n)
                    continue
                c = "(C)" if n in self.covereds else ""
                G.node(str(n.id), f"{n.get_info()}{c}")
            for e in self.edges:
                if e.src in droppeds or e.tgt in droppeds:
                    continue
                # logging.debug(f"edge from {e.src.id} to {e.tgt.id}")
                if self.error_trace is not None and e in self.error_trace:
                    color = "red"
                else:
                    color = "black"
                G.edge(str(e.src.id), str(e.tgt.id), str(e.id), color=color)
            for covered, covering in self.coverings:
                if covered in droppeds or covering in droppeds:
                    continue
                G.edge(str(covered.id), str(covering.id), style="dotted")
            G.render("out/" + name + "_dropped")

        # text
        with open("out/unwind.txt", "w") as f:
            for n in self.nodes:
                f.writelines([str(n.id), "\n", debug_print_list(n.label), "\n"])
                if self.config.theory == "liabv":
                    m = 2**self.config.bitwidth
                    if draw:
                        f.write(depict.depict_bv(n.label, m)[0])
                f.write("-"*40 + "\n")
        # covering
        with open("out/covering.txt", "w") as f:
            for cfrom, cto in self.coverings:
                if cto in self.covereds:
                    print("Wow!")
                f.write(f"[{cfrom.get_info()}->{cto.get_info()}]\n")
                f.write(debug_print_list(cfrom.label) + "\n")
                f.write(debug_print_list(cto.label) + "\n")
                if self.config.theory == "liabv":
                    m = 2**self.config.bitwidth
                    f.write(debug_print_list(simplifier.erase_obvious_connectives(
                        treeutil.convert_bv2lia(cfrom.label, m))) + "\n")
                    f.write(debug_print_list(simplifier.erase_obvious_connectives(
                        treeutil.convert_bv2lia(cto.label, m))) + "\n")
                    f.write(depict.depict_bv(cfrom.label, m)[0])
                    f.write(depict.depict_bv(cto.label, m)[0])
                f.write("-"*80 + "\n")
        print("hoge")

class MyException(Exception):
    pass

def run_impact_config(filename: str, config: Configuration, omit_print: bool, timeout_unwind: Optional[int]=None, print_format="png") -> Tuple[bool, statistics.Statistics]:
    config.smtutilopt.bitwidth = config.bitwidth
    config.smtutilopt.theory = config.theory
    if os.path.exists(filename_interpols):
        os.remove(filename_interpols)
    stat = statistics.Statistics()
    def temp_construct():
        return construct_cfg(config.bitwidth, filename, config.theory)
    st = time.time()
    cfg = temp_construct()
    t = time.time() - st
    stat.time_cfg_constr = t
    st = time.time()
    uw = Unwinding(cfg, config, stat)
    def temp_unwind():
        return uw.unwind()
    res_unwind = "init"
    res_unwind, stat = temp_unwind()
    # except Exception as e:
    #     import traceback, io
    #     mes = ""
    #     with io.StringIO() as f:
    #         traceback.print_exc(file=f)
    #         mes = f.getvalue()
    #         mes += "\n"
    #         mes += "-"*10 + "\n"
    #         mes += "\n".join(traceback.format_tb(e.__traceback__))
    #     raise MyException(mes)
    if res_unwind == True:
        res_unwind = "safe"
    elif res_unwind == False:
        res_unwind = "unsafe"
    else:
        res_unwind = str(res_unwind)
    # res_unwind = "safe" if res_unwind  else "unsafe"
    t = time.time() - st
    stat.time_verify = t
    print(f"It took {t} seconds.")
    logging.info(f"It took {t} seconds.")
    logging.info(f"The result is {res_unwind}.")
    if res_unwind == "unsafe":
        print(stat.to_dict())
        print(stat.counter_path_id)
        lst = [util.debug_print_list(x) for x in stat.counter_path_pred]
        print("Counterpath:")
        for n, pred in zip(stat.counter_path_id, lst):
            print(f"E{n}: {pred}")
        # print("Counterpath: \n", "\n".join(lst))
        print("Countermodel: \n", "\n".join([f"{k}: {v}" for k, v in stat.counter_model.items()]))
        # print("Counterpath(Edges):", stat.counter_path_id)
    if not omit_print:
        uw.print(debug=False, format=print_format, draw=False)

    return res_unwind, stat





def run_impact(filename: str, mode: str, bitwidth: int, simplification: bool, 
    floor_reduce: str,
    expand_floor_in_inteprolation_error: bool,
    disable_euf: bool,
               lia2bv: str,
                omit_print: bool,
                timeout: int) -> Tuple[bool, statistics.Statistics]:
    config = Configuration(mode, bitwidth, simplification, floor_reduce,
        expand_floor_in_inteprolation_error, disable_euf, lia2bv)
    return run_impact_config(filename, config, False, 60*10, "dot")


def my_preprocess(s):
    return s.replace(" __attribute__((noreturn));", ";")

def construct_cfg(bitwidth, filename, mode):
    processed = subprocess.check_output(f"gcc -E {filename}", shell=True).decode()
    processed = my_preprocess(processed)
    print(processed)
    parser = pycparser.c_parser.CParser()
    x = parser.parse(processed, filename="<none>")
    cfg = CFG.construct_CFG(x, mode, bitwidth)
    cfg.print("out/impact")
    cfg.print("out/cfg_pred", "pred")
    cfg._check_consistency()
    return cfg


if __name__ == "__main__":
    if len(sys.argv) > 1:
        parser = argparse.ArgumentParser(description="Implementation of 'Mind the Gap'")
        parser.add_argument("filename", type=str)
        parser.add_argument("mode", type=str, help="liabv/lia/cfg")
        parser.add_argument("-w", type=int, help="bitwidth", default=8)
        parser.add_argument("-s", type=bool, help="simplification", default=False)
        parser.add_argument("-timeout", type=int, help="timeout", default=600)
        args = parser.parse_args()
        if args.mode == "cfg":
            construct_cfg(args.w, args.filename, args.mode)
        else:
            res, stat = run_impact(args.filename, args.mode, args.w, False, "strategy2",
            expand_floor_in_inteprolation_error=False,
            disable_euf=True, lia2bv="boxing", omit_print=False, timeout=args.timeout)
    else:
        print("No arguments")

        # construct_cfg(6, "samples/nfm2017/benchmarks/svcomp16/loop-invgen/fragtest_simple_true-unreach-call.i.annot.c", "liabv")

        # tracemalloc.start(25)
        # tgt = "samples/my41.c" 
        tgt = 'samples/nfm2017/benchmarks/svcomp16/loops/trex03_true-unreach-call.i.annot.c'
        # res, stat = run_impact(tgt, "liabv", 3, False, "strategy2",
        # expand_floor_in_inteprolation_error=False,
        # disable_euf=True, lia2bv="boxing", omit_print=False)
        res, stat = run_impact(tgt, "lia", 4, False, "strategy2",
        expand_floor_in_inteprolation_error=False,
        disable_euf=True, lia2bv="boxing", omit_print=False, timeout=600)
        # snapshot = tracemalloc.take_snapshot()
        # top_stats = snapshot.statistics('traceback')

        # # pick the biggest memory block
        # stat = top_stats[0]
        # print("%s memory blocks: %.1f KiB" % (stat.count, stat.size / 1024))
        # for line in stat.traceback.format():
        #     print(line)

    print("result", res)
    if res:
        print("safe")
    else:
        print("unsafe")

