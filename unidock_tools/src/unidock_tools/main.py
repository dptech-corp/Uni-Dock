from pathlib import Path
import os
import sys
import glob
import logging
import importlib
import argparse

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from unidock_tools import application


def main_cli():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="cmd", help="Uni-Dock-related applications")
    name_module_dict = dict()

    app_files = glob.glob(os.path.join(next(p for p in application.__path__), "*.py"))
    for app_file in app_files:
        app_name = str(Path(app_file).stem)
        if app_name in ["__init__", "base"]:
            continue
        module = importlib.import_module(f".{app_name}", package=application.__name__)
        subparsers.add_parser(app_name, parents=[module.get_parser()], add_help=False)
        name_module_dict[app_name] = module

    args = parser.parse_args().__dict__
    logging.info(f"[Params] {args}")
    assert args["cmd"] in name_module_dict, \
        f"Invalid module name for unidocktools cmd: {args['cmd']}, choose from {list(name_module_dict.keys())}"

    name_module_dict[args["cmd"]].main(args)


if __name__ == "__main__":
    main_cli()
