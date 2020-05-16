import unittest
import impact
import timeout_decorator

simplification = [False]
st = "strategy2"
TIMEOUT = 60
expand_floor_in_inteprolation_error = False
disable_euf = True
lia2bv = "boxing"

class TestImpact(unittest.TestCase):
    @timeout_decorator.timeout(TIMEOUT)
    def test_test1(self):
        fn = "samples/test1.c"
        for s in simplification:
            print("*" + fn + str(s))
            self.assertEqual("unsafe", impact.run_impact(fn, "lia", 0, s, st, expand_floor_in_inteprolation_error, disable_euf, lia2bv, True)[0])
            self.assertEqual("unsafe", impact.run_impact(fn, "liabv", 3, s, st, expand_floor_in_inteprolation_error, disable_euf, lia2bv, True)[0])
    @timeout_decorator.timeout(TIMEOUT)
    def test_test1_(self):
        fn = "samples/test1_.c"
        for s in simplification:
            print("*" + fn + str(s))
            self.assertEqual("safe", impact.run_impact(fn, "lia", 0, s, st, expand_floor_in_inteprolation_error, disable_euf, lia2bv, True)[0])
            self.assertEqual("safe", impact.run_impact(fn, "liabv", 3, s, st, expand_floor_in_inteprolation_error, disable_euf, lia2bv, True)[0])
    @timeout_decorator.timeout(TIMEOUT)
    def test_test2(self):
        fn = "samples/test2.c"
        for s in simplification:
            print("*" + fn + str(s))
            self.assertEqual("unsafe", impact.run_impact(fn, "lia", 0, s, st, expand_floor_in_inteprolation_error, disable_euf, lia2bv, True)[0])
            self.assertEqual("unsafe", impact.run_impact(fn, "liabv", 3, s, st, expand_floor_in_inteprolation_error, disable_euf, lia2bv, True)[0])
    @timeout_decorator.timeout(TIMEOUT)
    def test_test3(self):
        fn = "samples/test3.c"
        for s in simplification:
            print("*" + fn + str(s))
            self.assertEqual("safe", impact.run_impact(fn, "lia", 0, s, st, expand_floor_in_inteprolation_error, disable_euf, lia2bv, True)[0])
            self.assertEqual("safe", impact.run_impact(fn, "liabv", 3, s, st, expand_floor_in_inteprolation_error, disable_euf, lia2bv, True)[0])
    @timeout_decorator.timeout(TIMEOUT)
    def test_test3_(self):
        fn = "samples/test3_.c"
        for s in simplification:
            print("*" + fn + str(s))
            self.assertEqual("unsafe", impact.run_impact(fn, "lia", 0, s, st, expand_floor_in_inteprolation_error, disable_euf, lia2bv, True)[0])
            self.assertEqual("unsafe", impact.run_impact(fn, "liabv", 3, s, st, expand_floor_in_inteprolation_error, disable_euf, lia2bv, True)[0])
    @timeout_decorator.timeout(TIMEOUT)
    def test_test3__(self):
        fn = "samples/test3__.c"
        for s in simplification:
            print("*" + fn + str(s))
            self.assertEqual("unsafe", impact.run_impact(fn, "lia", 0, s, st, expand_floor_in_inteprolation_error, disable_euf, lia2bv, True)[0])
            self.assertEqual("unsafe", impact.run_impact(fn, "liabv", 3, s, st, expand_floor_in_inteprolation_error, disable_euf, lia2bv, True)[0])
    @timeout_decorator.timeout(TIMEOUT)
    def test_test4(self):
        fn = "samples/test4.c"
        for s in simplification:
            print("*" + fn + str(s))
            self.assertEqual("unsafe", impact.run_impact(fn, "lia", 0, s, st, expand_floor_in_inteprolation_error, disable_euf, lia2bv, True)[0])
            self.assertEqual("unsafe", impact.run_impact(fn, "liabv", 3, s, st, expand_floor_in_inteprolation_error, disable_euf, lia2bv, True)[0])
    @timeout_decorator.timeout(TIMEOUT)
    def test_test5(self):
        fn = "samples/test5.c"
        for s in simplification:
            print("*" + fn + str(s))
            self.assertEqual("unsafe", impact.run_impact(fn, "lia", 0, s, st, expand_floor_in_inteprolation_error, disable_euf, lia2bv, True)[0])
            self.assertEqual("unsafe", impact.run_impact(fn, "liabv", 3, s, st, expand_floor_in_inteprolation_error, disable_euf, lia2bv, True)[0])
    @timeout_decorator.timeout(TIMEOUT)
    def test_test5_(self):
        fn = "samples/test5_.c"
        for s in simplification:
            print("*" + fn + str(s))
            self.assertEqual("unsafe", impact.run_impact(fn, "lia", 0, s, st, expand_floor_in_inteprolation_error, disable_euf, lia2bv, True)[0])
            self.assertEqual("unsafe", impact.run_impact(fn, "liabv", 3, s, st, expand_floor_in_inteprolation_error, disable_euf, lia2bv, True)[0])
    @timeout_decorator.timeout(TIMEOUT)
    def test_test5__(self):
        fn = "samples/test5__.c"
        for s in simplification:
            print("*" + fn + str(s))
            self.assertEqual("safe", impact.run_impact(fn, "lia", 0, s, st, expand_floor_in_inteprolation_error, disable_euf, lia2bv, True)[0])
            self.assertEqual("unsafe", impact.run_impact(fn, "liabv", 3, s, st, expand_floor_in_inteprolation_error, disable_euf, lia2bv, True)[0])
    @timeout_decorator.timeout(TIMEOUT)
    def test_01xx(self):
        fn = "samples/01xx.c"
        for s in simplification:
            print("*" + fn + str(s))
            self.assertEqual("safe", impact.run_impact(fn, "lia", 0, s, st, expand_floor_in_inteprolation_error, disable_euf, lia2bv, True)[0])
            self.assertEqual("safe", impact.run_impact(fn, "liabv", 3, s, st, expand_floor_in_inteprolation_error, disable_euf, lia2bv, True)[0])
    @timeout_decorator.timeout(TIMEOUT)
    def test_mod(self):
        fn = "samples/mod.c"
        for s in simplification:
            print("*" + fn + str(s))
            self.assertEqual("safe", impact.run_impact(fn, "lia", 0, s, st, expand_floor_in_inteprolation_error, disable_euf, lia2bv, True)[0])
            self.assertEqual("safe", impact.run_impact(fn, "liabv", 3, s, st, expand_floor_in_inteprolation_error, disable_euf, lia2bv, True)[0])
    @timeout_decorator.timeout(TIMEOUT)
    def test_mod2(self):
        fn = "samples/mod2.c"
        for s in simplification:
            print("*" + fn + str(s))
            # self.assertEqual("safe", impact.run_impact(fn, "lia", 0, s, st, expand_floor_in_inteprolation_error, disable_euf, lia2bv, True)[0])
            self.assertEqual("safe", impact.run_impact(fn, "liabv", 3, s, st, expand_floor_in_inteprolation_error, disable_euf, lia2bv, True)[0])
    @timeout_decorator.timeout(TIMEOUT)
    def test_mod3(self):
        fn = "samples/mod3.c"
        for s in simplification:
            print("*" + fn + str(s))
            # self.assertEqual("safe", impact.run_impact(fn, "lia", 0, s, st, expand_floor_in_inteprolation_error, disable_euf, lia2bv, True)[0])
            self.assertEqual("unsafe", impact.run_impact(fn, "liabv", 3, s, st, expand_floor_in_inteprolation_error, disable_euf, lia2bv, True)[0])
    @timeout_decorator.timeout(TIMEOUT)
    def test_mod_eldarica(self):
        fn = "samples/mod_eldarica.c"
        for s in simplification:
            print("*" + fn + str(s))
            # self.assertEqual("safe", impact.run_impact(fn, "lia", 0, s, st, expand_floor_in_inteprolation_error, disable_euf, lia2bv, True)[0])
            self.assertEqual("safe", impact.run_impact(fn, "liabv", 3, s, st, expand_floor_in_inteprolation_error, disable_euf, lia2bv, True)[0])
    @timeout_decorator.timeout(TIMEOUT)
    def test_mod_eldarica2(self):
        fn = "samples/mod_eldarica2.c"
        for s in simplification:
            print("*" + fn + str(s))
            # self.assertEqual("safe", impact.run_impact(fn, "lia", 0, s, st, expand_floor_in_inteprolation_error, disable_euf, lia2bv, True)[0])
            self.assertEqual("safe", impact.run_impact(fn, "liabv", 3, s, st, expand_floor_in_inteprolation_error, disable_euf, lia2bv, True)[0])
    @timeout_decorator.timeout(TIMEOUT)
    def test_mod_eldarica3(self):
        fn = "samples/mod_eldarica3.c"
        for s in simplification:
            print("*" + fn + str(s))
            # self.assertEqual("safe", impact.run_impact(fn, "lia", 0, s, st, expand_floor_in_inteprolation_error, disable_euf, lia2bv, True)[0])
            self.assertEqual("unsafe", impact.run_impact(fn, "liabv", 3, s, st, expand_floor_in_inteprolation_error, disable_euf, lia2bv, True)[0])
    @timeout_decorator.timeout(TIMEOUT)
    def test_free(self):
        fn = "samples/free.c"
        for s in simplification:
            print("*" + fn + str(s))
            self.assertEqual("unsafe", impact.run_impact(fn, "lia", 0, s, st, expand_floor_in_inteprolation_error, disable_euf, lia2bv, True)[0])
            self.assertEqual("unsafe", impact.run_impact(fn, "liabv", 3, s, st, expand_floor_in_inteprolation_error, disable_euf, lia2bv, True)[0])

if __name__ == "__main__":
    unittest.main()