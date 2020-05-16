import util
import unittest

class TestUtil(unittest.TestCase):
    def test_compare_trees_1(self):
        t1 = ["hoge", ["piyo", 1, True]]
        t2 = ["hoge", [], ["piyo", 1, True]]
        self.assertFalse(util.compare_trees(t1, t2))
    def test_compare_trees_2(self):
        t1 = ["hoge", []]
        t2 = ["hoge", []]
        self.assertTrue(util.compare_trees(t1, t2))
    def test_compare_trees_3(self):
        t1 = ["hoge", ["piyo", 1, True]]
        t2 = ["hoge", ["piyo", 1, True]]
        self.assertTrue(util.compare_trees(t1, t2))
    def test_argmax(self):
        x = ["hoge", "hogetarou", "a", "hogeh"]
        res, resval = util.argmax(x, len)
        self.assertEqual((res, resval), ("hogetarou", 9))
    def test_read_tree(self):
        x = "(<= (+ var_3_x (* -2 (floor (* 1/2 (* 1 var_3_x))))) 0)"
        res = util.read_tree(x)
        res2 = util.debug_print_list(res)
        self.assertEqual(x, res2)
    def test_compare_trees_4(self):
        t1 = util.read_tree('(floor (* 1/8 (* 1 var_9_i)))')
        t2 = util.read_tree('(floor (* 1/4 (* 1 var_5_x)))')
        self.assertFalse(util.compare_trees(t1, t2))
    def test_count_terms(self):
        x = ["and", ["or", "true", ["<", "x", "2"]], ["not", ["=>", ["<", "x", "4"], "false"]]]
        res = util.count_terms(x)
        self.assertEqual(res, 4)


if __name__ == "__main__":
    unittest.main()