import unittest
import smtutil
import depict


class TestDepict(unittest.TestCase):
    def test_depict_bv(self):
        x = ["bvult", "var_x", 3]
        res, vars = depict.depict_bv(x, 8)
        self.assertEqual(res, ">var_x\n***.....\n")
    def test_depict_bv_2(self):
        x = ["bvult", "var_x", ["bvadd", "var_y", ["_", "bv2", "8"]]]
        res, vars = depict.depict_bv(x, 8)
        expect = """^var_y
>var_x
*.......
........
*******.
******..
*****...
****....
***.....
**......
"""
        self.assertEqual(res, expect)
    def test_depict_bv_3(self):
        x = ["and", ["=", "var_x", ["_", "bv1", "8"]], ["=", ["bvurem", "var_y", ["_", "bv2", "8"]], ["_", "bv0", "8"]]]
        res, vars = depict.depict_bv(x, 8)
        expect = """^var_y
>var_x
........
.*......
........
.*......
........
.*......
........
.*......
"""
        self.assertEqual(res, expect)
    
        
if __name__ == "__main__":
    unittest.main()