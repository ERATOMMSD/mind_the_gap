import impact
import json
import shutil
import json
import util_boto3
import glob
import hashlib
import time
import os
import pickle


with open("task.pickle", "rb") as f:
    task = pickle.load(f)

if os.path.exists("out"):
    shutil.rmtree("out")
os.makedirs("out")
shutil.copy("task.pickle", "out")

try:
    omit_print = True
    config = task["config"]
    #TEMP
    # config.smtutilopt.bitwidth = config.bitwidth
    # config.smtutilopt.theory = config.theory
    #TEMP
    safety, stat = impact.run_impact_config(task["filename"], config, omit_print, task["timeout"], "dot")
    dumping = {"status": "ok", "safety": safety, "stat": stat.to_dict()}
except Exception as e:
    # raise(e)
    import traceback, io
    with io.StringIO() as f:
        traceback.print_exc(file=f)
        dumping = {"status": "error", "message": f.getvalue()}
        dumping["message"] += "\n"
        dumping["message"] += "-"*10 + "\n"
        dumping["message"] += "\n".join(traceback.format_tb(e.__traceback__))
        dumping["message"] += "-"*10 + "\n"
        dumping["message"] += "\n***\n".join([str(x) for x in e.args])
        dumping["message"] += "-"*10 + "\n"
        dumping["message"] += str(dir(e))
dumping["params"] = {"filename": task["filename"], "config": task["config"].to_dict()}

with open("out/result.json", "w") as f:
    json.dump(dumping, f)
