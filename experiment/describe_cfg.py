import glob
import pathlib
import impact
import shutil
import sys
import traceback
import os.path
import subprocess
import html

bw = 8


template_index = pathlib.Path("cfgs/index.template.html").read_text()
template_cfg = pathlib.Path("cfgs/cfg.template.html").read_text()


if len(sys.argv) < 2:
    files = pathlib.Path("tasks.txt").read_text().split("\n")
    files = [x for x in files if x != ""]
else:
    d = sys.argv[1]
    files = list(glob.glob(os.path.join(d, "*.c")))
    print(d)

files.sort()
lines = []
sucs = []
for i in files:
    print(i)
    # code = pathlib.Path(i).read_text()
    code = subprocess.check_output(f"gcc -E {i}", shell=True).decode()
    filename = os.path.basename(i)
    # if "annot.c" in filename:
    #     continue
    html_cfg = template_cfg.replace("#TITLE#", filename).replace("#CODE#", html.escape(code))
    error_happend = False
    try:
        impact.construct_cfg(bw, i, "lia")
        pic_lia = f"cfgs/{filename}.lia.png"

        shutil.copy("out/impact.png", pic_lia)
        pic_lia = f"cfgs/{filename}.lia.pred.png"
        shutil.copy("out/cfg_pred.png", pic_lia)
        # impact.construct_cfg(bw, i, "liabv")
        impact.construct_cfg(bw, i, "liabv")
        pic_liabv = f"cfgs/{filename}.liabv.pred.png"
        shutil.copy("out/cfg_pred.png", pic_liabv)
    except Exception as e:
        t, v, tb = sys.exc_info()
        emes = ""
        emes += ("\n".join(traceback.format_exception(t, v, tb)))
        emes += ("\n")
        emes += ("\n".join(traceback.format_tb(e.__traceback__)))
        html_cfg = html_cfg.replace("#ERROR#", emes)
        error_happend = True
    pathlib.Path(f"cfgs/{filename}.html").write_text(html_cfg)
    line = f"<li><a href={filename}.html>{filename}</a></li>"
    if error_happend:
        line += " ERROR"
    else:
        sucs.append(i)
    lines.append(line)

html_index = template_index.replace("#MAIN#", "\n".join(lines))
pathlib.Path(f"cfgs/index.html").write_text(html_index)

with open("parsed.txt", "a") as f:
    for i in sucs:
        f.writelines(i + "\n")
