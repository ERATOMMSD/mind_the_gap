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

def reset_workingdir():
    if os.path.exists(workingdir):
        shutil.rmtree(workingdir)
    os.mkdir(workingdir)

def make_summary(files):
    reset_workingdir()
    pickings = []
    for i, file in enumerate(files):
        row = {"zipname": file}
        print(i, file)
        util_boto3.download(file, workingdir)
        # try:
        #     shutil.move(file, workingdir)
        # except shutil.Error as e:
        #     print("Ignoring existing error")
        with zipfile.ZipFile(os.path.join(workingdir, file)) as myzip:
            with myzip.open("task.pickle", "r") as f:
                task = pickle.load(f)
                setting = {"filename": task["filename"], "config": task["config"].to_dict()}
                row["targetname"] = task["filename"]
                row["theory"] = task["config"].theory
                row["bitwidth"] = task["config"].bitwidth
                row["lia2bv"] = task["config"].smtutilopt.lia2bv
                row["id"] = task["id"]
                if row["id"] != 0:
                    continue
                filename = setting["filename"]
                if "annot.c" in filename:
                    if "VeriMAP" in filename:
                        continue
                    if "svcomp16/locks" in filename:
                        continue
                    if "benchmarks/llreve" in filename:
                        continue
            pickings.append(file)
    pathlib.Path("picked.txt").write_text("\n".join(pickings))


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
    make_summary_from_s3(sys.argv[1])