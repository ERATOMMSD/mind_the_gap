import json
import sys
import zipfile


with zipfile.ZipFile(sys.argv[1]) as z:
    with z.open("result.json", "r") as f:
        r = json.load(f)
print(r)