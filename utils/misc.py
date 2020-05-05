import sys
import json
from pathlib import Path
from collections import OrderedDict

MAX_PART_SIZE = 10 * (1024**2)

def fetch_arg(_pos):
    """ Fetch arg from sys args
    """
    if len(sys.argv) <= _pos:
        raise Exception(
            "Missing arguments: <test_file> <output_file>")
    return sys.argv[_pos]

def parse_predit_args():
    """ Parse arguments
    """
    args = OrderedDict({
        "test_file": str, "output_file": str
    })
    for i, a in enumerate(args):
        args[a] = args[a](fetch_arg(i + 1))
    return args

def log(*msg, lvl="INFO"):
    """ Log message with visual help
    """
    print("-"*50)
    print("[{}]".format(lvl), end=" ")
    for m in msg:
        print("{}".format(m), end=" ")
    print("\n" + "-"*50)

def read_file(sc, fpath):
    """ Read a file
    """
    _fsize = Path(fpath).stat().st_size
    return sc.textFile(fpath, _fsize // MAX_PART_SIZE )

def read_json(sc, fpath):
    """ Read JSON-rowed file parsed in to RDD
    """
    data = read_file(sc, fpath)\
            .map(lambda x: json.loads(x))
    return data

def read_csv(sc, fpath, with_heads=False):
    """ Read and parse CSV into RDD
    """
    def filter_heads(z): return z[1] > 0
    data = read_file(sc, fpath)\
        .zipWithIndex()\
        .filter(lambda z: True if with_heads else filter_heads(z))\
        .map(lambda z: tuple(z[0].split(',')))
    return data

def debug():
    """ Local debug
    """
    import ipdb; ipdb.set_trace()