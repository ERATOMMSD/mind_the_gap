
class Statistics():
    def __init__(self):
        self.time_cfg_constr = 0 #ok
        self.time_verify = 0 #ok
        self.time_smt = 0 #ok
        self.time_smt_pure = 0
        self.time_interp = 0 #ok
        self.time_interp_pure = 0 #ok
        self.time_to_lia = 0 #ok
        self.time_from_lia = 0 #ok
        self.time_process_lia = 0
        self.sizes_interpol = [] #ok
        self.num_smt = 0 #ok
        self.num_interp = 0 #ok
        self.num_interp_failure = 0 #ok
        self.num_boxing = 0
        self.num_boxing_multi_variable = 0
        self.giveup_print_unwind = False #ok
        self.counter_path_id = None
        self.counter_model = None
        self.counter_path_pred = None
        self.bitwidth = None
        self.theory = None
    def to_dict(self):
        return {"time_cfg_constr": self.time_cfg_constr,
        "time_verify": self.time_verify,
        "time_smt": self.time_smt,
        "time_interp": self.time_interp,
        "time_to_lia": self.time_to_lia,
        "time_from_lia": self.time_from_lia,
        "sizes_interpol": self.sizes_interpol,
        "num_smt": self.num_smt,
        "num_interp": self.num_interp,
        "num_interp_failure": self.num_interp_failure,
        "giveup_print_unwind": self.giveup_print_unwind,
        "counter_path_id": self.counter_path_id,
        "counter_model": self.counter_model,
        "counter_path_pred": self.counter_path_pred,
        "time_smt_pure": self.time_smt_pure,
        "time_interp_pure": self.time_interp_pure,
        "time_process_lia": self.time_process_lia,
        "bitwidth": self.bitwidth,
        "theory": self.theory,
        "num_boxing": self.num_boxing,
        "num_boxing_multi_variable": self.num_boxing_multi_variable
        }