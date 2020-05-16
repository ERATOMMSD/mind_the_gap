import unittest
import smtutil
import treeutil
import util
import depict

class TestSMTUtil(unittest.TestCase):
    def test_evaluate_bv(self):
        x = ["bvult", 2, 7]
        self.assertTrue(treeutil.evaluate_bv(x, 16, {}))
    def test_evaluate_bv_2(self):
        x = ["bvult", "var_x", 7]
        self.assertFalse(treeutil.evaluate_bv(x, 16, {"var_x": 9}))
    def test_evaluate_bv_3(self):
        x = ["bvslt", "var_x", 7]
        self.assertTrue(treeutil.evaluate_bv(x, 16, {"var_x": 9}))
    def test_evaluate_bv_4(self):
        x = ["bvadd", ["_", "bv3", "8"], ["_", "bv7", "8"]]
        self.assertEqual(treeutil.evaluate_bv(x, 8, {"var_x": 9}), 2)
    def test_convert_lia2bv(self):
        m = 8
        s = "(and (<= 0 var_9_i) (<= var_9_i 7) (<= (+ var_9_i (* -8 (floor (* 1/8 (* 1 var_9_i))))) 0) (<= (+ var_5_x (* -4 (floor (* 1/4 (* 1 var_5_x))))) 0))"
        t = util.read_tree(s)
        opt = util.SMTUtilOption("strategy1", False, True)
        res = treeutil.convert_lia2bv(t, m, opt)
        pict = depict.depict_bv(res, 8)[0]
        ress = util.debug_print_list(res)
        expect = """^var_9_i
>var_5_x
........
........
........
........
........
........
........
*...*...
"""
        self.assertEqual(expect, pict)
        
if __name__ == "__main__":
    unittest.main()