from unittest import TestCase
import naive_lia2bv
import util
import smtutil
import treeutil


class TestNaive_lia2bv_normalized_ineq(TestCase):
    def test_naive_lia2bv_normalized_ineq(self):
        x_c = {"var_0_x": 1, "var_0_y": 2}
        b = 5
        m = 8
        orig = ["<=", ["+", ["*", "var_0_x", "1"], ["*", "var_0_y", "2"]], "5"]
        gen = naive_lia2bv.naive_lia2bv_normalized_ineq(x_c, b, m)
        gen = ["and", gen, smtutil.get_range_constraints(x_c.keys(), m)]
        print(util.debug_print_list(gen))
        orig = ["and", orig, smtutil.get_range_constraints(x_c.keys(), m)]
        print(util.debug_print_list(orig))
        assr = ["and", ["=>", gen, orig], ["=>", orig, gen]]
        res = smtutil.check_sat_lia(assr)
        self.assertTrue(res)

    def test_naive_lia2bv_normalized_ineq2(self):
        x_c = {"var_0_x": 20, "var_0_y": 2}
        b = 5
        m = 8
        orig = ["<=", ["+", ["*", "var_0_x", "20"], ["*", "var_0_y", "2"]], "5"]
        gen = naive_lia2bv.naive_lia2bv_normalized_ineq(x_c, b, m)
        gen = ["and", gen, smtutil.get_range_constraints(x_c.keys(), m)]
        print(util.debug_print_list(gen))
        orig = ["and", orig, smtutil.get_range_constraints(x_c.keys(), m)]
        print(util.debug_print_list(orig))
        assr = ["and", ["=>", gen, orig], ["=>", orig, gen]]
        res = smtutil.check_sat_lia(assr)
        self.assertTrue(res)
