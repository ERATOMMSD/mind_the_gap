import util_boto3
import os
import shutil
import zipfile
import json
import pathlib
import sys
import re
import pickle
import csv

workingdir = "summary"

tmpl_item = """<h2>#SETTING#</h2>
#RESULT#
<pre>
#CONTENT#
</pre>
<div>#ZIPNAME#</div>
<div>#OUTERTIME# sec </div>
<hr/>
"""

def reset_workingdir():
    if os.path.exists(workingdir):
        shutil.rmtree(workingdir)
    os.mkdir(workingdir)

def make_summary(files, no_download=False):
    reset_workingdir()
    total = ""
    errors = 0
    timeouts = 0
    rows = []
    for i, file in enumerate(files):
        row = {"zipname": file}
        print(i, file)
        if no_download == False:
            util_boto3.download(file, workingdir)
        else:
            shutil.copy(file, workingdir)
        # try:
        #     shutil.move(file, workingdir)
        # except shutil.Error as e:
        #     print("Ignoring existing error")
        with zipfile.ZipFile(os.path.join(workingdir, file)) as myzip:
            with myzip.open("time.txt") as f:
                outertime = f.readline().decode()
                row["time"] = outertime.replace("(timeout)", "")
                if "timeout" in outertime:
                    timeouts += 1
                    row["timeout"] = 1
                else:
                    row["timeout"] = 0
            with myzip.open("task.pickle", "r") as f:
                task = pickle.load(f)
                setting = {"filename": task["filename"], "config": task["config"].to_dict()}
                row["targetname"] = task["filename"]
                row["theory"] = task["config"].theory
                row["bitwidth"] = task["config"].bitwidth
                row["lia2bv"] = task["config"].smtutilopt.lia2bv
                row["id"] = task["id"]
            if "result.json" in myzip.namelist():
                with myzip.open("result.json") as f:
                    resjson = json.load(f)
                    result = resjson["status"]
                    # result = str(resjson)
                    message = resjson.get("message", "no message")
                    if result == "error":
                        result = "errorinverif"
                        errors += 1
                    else:
                        message = resjson.get("safety", "???")
                    row["safety"] = resjson.get("safety", "")
                    if "stat" in resjson:
                        stat = resjson["stat"]
                        row["time_verify"] = stat["time_verify"]
                        row["time_smt"] = stat["time_smt"]
                        row["time_interp"] = stat["time_interp"]
                        row["time_to_lia"] = stat["time_to_lia"]
                        row["time_from_lia"] = stat["time_from_lia"]
                        row["total_sizes_interp"] = sum([sum(i) for i in stat["sizes_interpol"]])
                        row["num_smt"] = stat["num_smt"]
                        row["num_interp"] = stat["num_interp"]
                        row["num_interp_failure"] = stat["num_interp_failure"]
                        row["time_smt_pure"] = stat["time_smt_pure"]
                        row["time_interp_pure"] = stat["time_interp_pure"]
                        row["time_process_lia"] = stat["time_process_lia"]
                        row["num_boxing"] = stat["num_boxing"]
                        row["num_boxing_multi_variable"] = stat["num_boxing_multi_variable"]
                    if "MemoryError" in message:
                        row["error_kind"] = "MemoryError"
                    elif "SystemError" in message:
                        row["error_kind"] = "SystemError"
                    elif "Error" in message:
                        row["error_kind"] = "UnknownError"
            else:
                result = "NONE"
                message = "NONE"
        # outertime = resjson["outertime"]
        item = tmpl_item
        item = item.replace("#SETTING#", json.dumps(setting))
        item = item.replace("#RESULT#", result)
        item = item.replace("#CONTENT#", message)
        item = item.replace("#ZIPNAME#", file)
        item = item.replace("#OUTERTIME#", str(outertime))
        total += item

        rows.append(row)
    total = f"<h2>ERRORS: {errors}/{len(files)}</h2>\n" + total
    total = f"<h2>TIMEOUTS: {timeouts}/{len(files)}</h2>\n" + total
    pathlib.Path(os.path.join(workingdir, "summary_error.html")).write_text(total)

    with open(os.path.join(workingdir, "summary.csv"), "w") as f:
#         targetname/theory/bitwidth/lia2bv/
# safety/time/timeout/error_kind/time_verify/time_smt/time_interp/time_to_lia/
# time_from_lia/total_sizes_interpol/num_smt/num_interp/num_interp_failure/
# zipname/link_to_path(あとで)
        cols = ["targetname", "theory", "bitwidth", "lia2bv", "id",
        "safety", "time", "timeout", "error_kind", "time_verify", "time_smt",
        "time_smt_pure",
        "time_interp", "time_interp_pure", "time_process_lia",
         "time_to_lia", "time_from_lia", "total_sizes_interp",
        "num_smt", "num_interp", "num_boxing", "num_boxing_multi_variable",
         "num_interp_failure", "zipname"]
        writer = csv.DictWriter(f, cols)
        writer.writeheader()
        writer.writerows(rows)


def make_summary_from_s3(txt):
    reset_workingdir()
    util_boto3.download(txt)
    shutil.move(txt, workingdir)
    files = pathlib.Path(os.path.join(workingdir, txt)).read_text()
    files = files.split("\n")
    files = [file for file in files if file != ""]
    print(len(set(files)), len(files))
    r = re.compile(r"-((\d|-)*)_")
    machines = set()
    for file in files:
        machine = r.findall(file)[0][0]
        machines.add(machine)
    print(len(machines), machines)
    input()
    make_summary(files)

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        import glob
        print("no argument")
        make_summary(glob.glob("*interpol.zip"), no_download=True)
    else:
        make_summary_from_s3(sys.argv[1])