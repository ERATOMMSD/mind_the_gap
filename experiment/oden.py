from flask import Flask, make_response, request, abort
import pickle
import threading
import requests
import datetime
import hashlib
import time
import sys
import logging
import pathlib
import copy
import os
import os.path
import impact #ERASEHERE
import itertools
import hashlib


try:
    with open("slack_webhook.txt", "r") as f:
        slack_url = f.readline().strip()
except Exception as e:
    pass


def notify_slack(message: str) -> bool:
    import json
    try:
        requests.post(slack_url, data=json.dumps({
            'text': message,  # 投稿するテキスト
            'username': u'rnn2wfa',  # 投稿のユーザー名
            'icon_emoji': u':ghost:',  # 投稿のプロフィール画像に入れる絵文字
            'link_names': 1,  # メンションを有効にする
        }))
        return True
    except Exception as e:
        print(e)
        return False

def get_time_hash():
    return datetime.datetime.now().strftime("%Y%m%d%H%M%S")


# State of worker.  It can be ("idle",), ("working", task), ("done", task, res) or
lock_state = threading.Lock()
state_worker = ("idle",)


# ------ User write below ------
# Hostnames are written separated by breaks.
# hosts = """
# 999.999.999.999
# 999.999.999.998
# 999.999.999.997
# """
# The following try-except block reads hostnames from hosts.txt.
# If you wrote the hostnames above, please disable the block.
try:
    hosts = pathlib.Path("hosts.txt").read_text()
except FileNotFoundError:
    hosts = ""
    
name = "sample"
interval_polling = 5
timeout = 30

# Returns the list of the tasks
def make_tasks_usual():
    paths = pathlib.Path("smaller.txt").read_text().split("\n")
    paths = [p for p in paths if p != ""]
    #temp
    # paths = ["'samples/nfm2017/benchmarks/llreve/barthe_merged_safe.c'"]
    print(paths[:5])
    modes = ["lia", "liabv"]
    widths = [8]
    simps = [False]
    strategies = ["strategy2"]
    expand_floor_in_interpolations = [False]
    disable_euf = True
    lia2bvs = ["naive", "boxing"]
    omit_print = False
    to = 10*60 #temp
    tasks = []
    for id in range(1):
        for path in paths:
            for mode, simp in itertools.product(modes, simps):
                if mode == "lia":
                    config = impact.Configuration("lia", 0, simp, "dummy", False, False, "dummy")
                    tasks.append({"filename": path, "config": config, "timeout": to, "id": id})
                elif mode == "liabv":
                    for width, strategy, expand_floor_in_interpolation, lia2bv in itertools.product(widths, strategies, expand_floor_in_interpolations, lia2bvs):
                        config = impact.Configuration("liabv", width, simp, strategy, expand_floor_in_interpolation, disable_euf, lia2bv)
                        tasks.append({"filename": path, "config": config, "timeout": to, "id": id})
                else:
                    assert False
            
    return tasks

def make_tasks_string():
    import json
    to = 10*60
    str = """{'filename': 'samples/nfm2017/benchmarks/llreve/barthe_merged_safe.c', 'config': 0, 'timeout': 600, 'id': 0}"""
    strs = str.split("\n")
    # strs = str.replace("'", '"')
    print(strs)
    jsons = [json.loads(s) for s in strs]
    tasks = []
    for j in jsons:
        con = j["config"]
        theory = con["theory"]
        width = int(con["bitwidth"])
        simp = bool(con["simplification"])
        expand_floor_in_interpolation = bool(con["smtutilopt"]["expand_floor_in_interpolation_error"])
        disable_euf = bool(con["smtutilopt"]["disable_euf"])
        floor_reduce = con["smtutilopt"]["floor_reduce"]
        lia2bv = con["smtutilopt"]["lia2bv"]
        config = impact.Configuration(theory, width, simp, floor_reduce, expand_floor_in_interpolation, disable_euf, lia2bv)
        tasks.append({"filename": j["filename"], "config": config, "timeout": to})
    return tasks

def make_tasks():
    return make_tasks_usual()

# Takes a task and return the result
def calc(task, no_upload=False):
    import os
    import pickle
    import subprocess
    if no_upload:
        import util_boto3_dummy as util_boto3
    else:
        import util_boto3
    import glob

    outerst = time.time()

    with open("task.pickle", "wb") as f:
        pickle.dump(task, f)
    cmd = f"pipenv run python impact_from_task.py"
    try:
        output = subprocess.check_output(cmd.split(), shell=False, timeout=task["timeout"], cwd=os.getcwd(), stderr=subprocess.STDOUT).decode()
    except subprocess.TimeoutExpired as e:
        output = "timeout"
    except subprocess.CalledProcessError as e:
        output = f"ProcessError\n{e.returncode}\n{e.cmd}\n{e.output}"

    outer = time.time() - outerst
    if output == "timeout":
        outer = str(outer) + "(timeout)"
    with open("out/time.txt", "w") as f:
        f.write(str(outer))


    taskhash = hashlib.md5(str(task).encode()).hexdigest()[:12]
    n = util_boto3.zip_and_upload(f"{taskhash}_interpol.zip", glob.glob("out/*"), True)
    print(n)

    return n


# Called when no tasks are assgined to a worker
def handle_finish_machine(uri, name):
    pass


# Called when all the tasks are processed
def handle_finish_tasks():
    notify_slack("finished all!")


# Returns a string that is shown when 'http://(hostname):8080/' is accessed
def show_status():
    return state_worker[0]


# ------ User write above ------

def get_hash(o):
    return hashlib.md5(pickle.dumps(o)).hexdigest()


# Prepare Flask
app = Flask(__name__)


# Make pages
@app.route("/", methods=["GET"])
def respond_home():
    return show_status()


def invoke_calc(task):
    global state_worker
    try:
        res = calc(copy.copy(task))
        lock_state.acquire()
        state_worker = ("done", task, res)
        lock_state.release()
    except Exception as e:
        import traceback, io
        with io.StringIO() as f:
            traceback.print_exc(file=f)
            app.logger.error(f.getvalue())
            lock_state.acquire()
            state_worker = ("error", task, f.getvalue())
            lock_state.release()


@app.route("/calc", methods=["POST"])
def respond_calc():
    global state_worker
    app.logger.info("Got calculation request".format())
    task = pickle.loads(request.data)
    try:
        lock_state.acquire()
        if state_worker[0] == "idle":
            state_worker = ("working", task)
            threading.Thread(target=invoke_calc, args=(task,)).start()
            app.logger.info("Accepting task".format())
        else:
            lock_state.release()
            app.logger.info("Rejecting the request because the state is {0}".format(state_worker[0]))
            abort(503, {})
        lock_state.release()
    except Exception as e:
        import traceback, io
        with io.StringIO() as f:
            traceback.print_exc(file=f)
            app.logger.error(f.getvalue())
            state_worker = ("error", task, f.getvalue())
    return "Accepted your task"


@app.route("/retrieve", methods=["POST"])
def respond_retrieve():
    global state_worker
    response = make_response()
    app.logger.info("Got retrieval request".format())
    task_request = pickle.loads(request.data)
    task_request_hash = get_hash(task_request)
    if state_worker[0] == "idle":
        app.logger.error("The state was idle".format())
        abort(404, {}) #404
    elif state_worker[0] == "working":
        app.logger.info("The state was working".format())
        abort(503, {})  # Service Unavailable
    elif state_worker[0] == "done":
        app.logger.info("The state was done".format())
        lock_state.acquire()
        _, task, res = state_worker
        task_hash = get_hash(task)
        if task_hash != task_request_hash:
            app.logger.error("The task we have done and the task of the request are different".format(),
                             extra={"who": "retrieve"})
            app.logger.error("Task we have: {0}".format(task),
                             extra={"who": "retrieve"})
            app.logger.error("Task of request: {0}".format(task_request),
                             extra={"who": "retrieve"})
            lock_state.release()
            abort(404, {})
        res = pickle.dumps({"task": task, "result": pickle.dumps(res)})
        response.data = res
        response.mimetype = "application/octet-stream"
        state_worker = ("idle",)
        app.logger.info("Returning the result".format())
        lock_state.release()
        return response
    elif state_worker[0] == "error":
        app.logger.info("The state was error".format())
        lock_state.acquire()
        _, task, error = state_worker
        res = pickle.dumps({"task": task, "error": error})
        response.data = res
        response.mimetype = "application/octet-stream"
        state_worker = ("idle",)
        lock_state.release()
        return response
    app.logger.info("Unexpected state {0}".format(state_worker))
    abort(500, {}) # Internal server error


@app.route("/status", methods=["GET"])
def respond_status():
    global state_worker
    return state_worker[0]


def caller(server, tasks, finisheds, lock, total):
    time_start = time.time()
    uri_server = server[0]
    name_server = server[1]
    rootLogger.info("Starting {0}@{1}".format(name_server, uri_server), extra={"who": name_server})
    while True:
        lock.acquire()
        if tasks == []:
            lock.release()
            break
        task = tasks.pop()
        rootLogger.info("Popped".format(), extra={"who": name_server})
        lock.release()
        try:
            filename = "{0}_{1}.done.pickle".format(name_server, get_time_hash())
            filename_error = "{0}_{1}.error.pickle".format(name_server, get_time_hash())
            data = pickle.dumps(task)
            res = requests.post(uri_server + "calc", data=data, timeout=timeout)
            if res.status_code == 200:
                rootLogger.info("Request is accepted".format(), extra={"who": name_server})
                while True:
                    time.sleep(interval_polling)
                    # rootLogger.info("Polling".format(), extra={"who": name_server})
                    res2 = requests.post(uri_server + "retrieve", data=data, timeout=timeout)
                    if res2.status_code == 200:
                        res2.raw.decode_content = True
                        res = pickle.loads(res2.content)
                        if "result" in res:
                            with open(filename, "wb") as f:
                                rootLogger.info("Saving the result as {0}".format(filename),
                                                extra={"who": name_server})
                                f.write(res2.content)
                            break
                        elif "error" in res:
                            rootLogger.info("Internal error occurred in the remote machine on the task: {0}".format(res["error"]),
                                            extra={"who": name_server})
                            with open(filename_error, "wb") as f:
                                rootLogger.info("Saving the error result as {0}".format(filename_error),
                                                extra={"who": name_server})
                                f.write(res2.content)
                            break
                        else:
                            raise Exception("Invalid result is given")
                    elif res2.status_code == 503:
                        pass  # the remote is working
                    elif res2.status_code == 404:
                        raise Exception("The remote machine is in idle.  The task was gone away...")
                    else:
                        raise Exception("Got unexpected error code {0}".format(res2.status_code))
                time_elapsed = time.time() - time_start
                lock.acquire()
                finisheds.append(task)
                lock.release()
                speed = time_elapsed / len(finisheds)
                eta = speed*(total - len(finisheds))
                rootLogger.info("Finished {0}/{1} tasks ({2} in the queue).  ETA is {3}".format(len(finisheds), total, len(tasks), eta),
                                extra={"who": name_server})
            else:
                rootLogger.info("Retrieving failed with {1}".format(res.status_code), extra={"who": name_server})
        except Exception as e:
            import traceback, io
            with io.StringIO() as f:
                traceback.print_exc(file=f)
                rootLogger.error("Request failed with the following error: {0}".format(f.getvalue()),
                                 extra={"who": name_server})
            # put the failed task back to the queue
            rootLogger.info("Putting back the failed task to the queue", extra={"who": name_server})
            lock.acquire()
            tasks.append(task)
            lock.release()
            break

    rootLogger.info("Closing".format(), extra={"who": name_server})
    handle_finish_machine(uri_server, name_server)


def get_servers():
    servers = [x.strip() for x in hosts.split("\n") if x.strip() != ""]
    servers = ["http://{0}:8080/".format(x) for x in servers]
    servers = [(server, name + str(i)) for i, server in enumerate(servers)]
    return servers

def get_status_remotes():
    xs = []
    servers = get_servers()
    for server in servers:
        try:
            res = requests.get(server[0] + "status")
            xs.append((server[0], res.content.decode()))
        except:
            xs.append((server[0], "error"))
    return xs

def tweak_result(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

if __name__ == "__main__":
    # Prepare logger
    global rootLogger
    global logFile

    if len(sys.argv) < 2:
        print("No modes specified")
        # sys.argv = [sys.argv[0], "test"]
        sys.exit(1)

    if sys.argv[1] == "worker":
        logFormatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    else:
        logFormatter = logging.Formatter("%(asctime)s [%(levelname)s] [%(who)s]  %(message)s")

    rootLogger = logging.getLogger("oden")
    rootLogger.setLevel(logging.INFO)

    logFile = "{0}.log".format(get_time_hash())
    fileHandler = logging.FileHandler(logFile)
    fileHandler.setFormatter(logFormatter)
    rootLogger.addHandler(fileHandler)

    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(logFormatter)
    rootLogger.addHandler(consoleHandler)

    # Main
    # print(sys.argv[1])

    if sys.argv[1] in ["manager", "test", "resume", "local"]:
        tasks = make_tasks()
        resume = False
        if sys.argv[1] == "resume":
            resume = True
            rootLogger.info("Starting resume mode".format(), extra={"who": "resume"})
            # hoge
            hash2task = {get_hash(v): v for v in tasks}
            tasks_hash = [get_hash(t) for t in tasks]
            tasks_mset = {h: tasks_hash.count(h) for h in hash2task.keys()}
            dones = []
            for i in pathlib.Path(".").glob("{0}*.done.pickle".format(name)):
                with open(i, "rb") as f:
                    dones.append(pickle.load(f)["task"])
            dones_hash = [get_hash(t) for t in dones]
            dones_mset = {h: dones_hash.count(h) for h in hash2task.keys()}
            remaining_mset = {h: tasks_mset[h] - dones_mset[h] for h in hash2task.keys()}
            tasks = []
            for k, v in remaining_mset.items():
                tasks += [copy.copy(hash2task[k]) for _ in range(v)]

            rootLogger.info("Loaded {0} tasks".format(len(tasks)), extra={"who": "resume"})
            sys.argv[1] = "manager"
        if sys.argv[1] == "manager":
            rootLogger.info("Starting manager mode", extra={"who": "manager"})
            # check if the remotes are ready
            for i in get_status_remotes():
                if i[1] != "idle":
                    rootLogger.error("Machine {0} is not in idle ({1}).  Cannot start the calculation.".format(i[0], i[1]))
                    sys.exit(1)
            #check if the previous calculation remaining
            if any(pathlib.Path(".").glob("{0}*.pickle".format(name))) and (not resume):
                ans = ""
                while True:    
                    ans = input("The previous calculations seems remaining.  Do you really start the calculation? (y/n)")
                    if ans.lower().startswith("y"):
                        break
                    elif ans.lower().startswith("n"):
                        sys.exit(1)

            #
            servers = get_servers()
            rootLogger.info("Servers: " + str(servers), extra={"who": "manager"})

            lock = threading.Lock()
            num_tasks = len(tasks)
            finisheds = []
            rootLogger.info("We have {0} tasks.".format(num_tasks), extra={"who": "manager"})
            threads = []
            for server in servers:
                t = threading.Thread(target=caller, args=(server, tasks, finisheds, lock, num_tasks))
                t.start()
                threads.append(t)
            while True:
                if all([(not t.is_alive()) for t in threads]):
                    handle_finish_tasks()
                    break
        elif sys.argv[1] == "test":
            tasks = make_tasks()
            rootLogger.info("Starting test mode", extra={"who": "test"})
            rootLogger.info("There are " + str(len(tasks)) + " tasks.", extra={"who": "test"})
            input()
            for i in tasks:
                rootLogger.info("Starting task {0}".format(i), extra={"who": "manager"})
                res = calc(i)
                filename = "{0}_{1}.done.pickle".format("odentest", get_time_hash())
                with open(filename, "wb") as f:
                    pickle.dump(res, f)
                input("waiting...")
        elif sys.argv[1] == "local":
            tasks = make_tasks()
            rootLogger.info("Starting local mode", extra={"who": "test"})
            rootLogger.info("There are " + str(len(tasks)) + " tasks.", extra={"who": "test"})
            for i in tasks:
                rootLogger.info("Starting task {0}".format(i), extra={"who": "manager"})
                res = calc(i, no_upload=True)
                filename = "{0}_{1}.done.pickle".format("odentest", get_time_hash())
                with open(filename, "wb") as f:
                    pickle.dump(res, f)
                # input("waiting...")

    elif sys.argv[1] == "worker":
        if len(sys.argv) > 2:
            port = int(sys.argv[2])
        else:
            port = 8080
        app.run(host='0.0.0.0', port=port)
    elif sys.argv[1] == "status":
        for i in get_status_remotes():
            print("{0}\t{1}".format(i[0], i[1]))
    else:
        rootLogger.fatal("Invalid argument {0}".format(sys.argv[1]), extra={"who": "error"})
