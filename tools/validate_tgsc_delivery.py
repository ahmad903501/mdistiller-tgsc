"""Validate thin notebooks and the runtime in an isolated, dependency-preserving copy."""
import ast
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import uuid


def main():
    root = Path(__file__).resolve().parents[1]
    scratch = (root / ".test-artifacts").resolve()
    stage = scratch / ("delivery_" + uuid.uuid4().hex)
    stage.mkdir(parents=True)
    try:
        for folder in ("mdistiller", "configs", "tools", "tests"):
            shutil.copytree(root/folder, stage/folder, ignore=shutil.ignore_patterns("__pycache__"))
        names = []
        for path in sorted((root/"notebooks").glob("tgsc_*.ipynb")):
            nb = json.loads(path.read_text(encoding="utf-8"))
            assert nb["nbformat"] == 4
            for cell in nb["cells"]:
                if cell["cell_type"] == "code":
                    ast.parse("".join(cell["source"]))
                    assert cell["outputs"] == [] and cell["execution_count"] is None
            names.append(path.name)
        assert len(names) == 4
        env = dict(os.environ, PYTHONPATH="")
        # Keep user-site dependencies (torch is installed there on Windows), exclude sibling source imports.
        command = ("import pathlib,sys; sys.path.insert(0,str(pathlib.Path.cwd())); "
                   "import mdistiller.tgsc.runtime as r; "
                   "assert pathlib.Path(r.__file__).resolve().is_relative_to(pathlib.Path.cwd())")
        check = subprocess.run([sys.executable,"-c",command], cwd=stage,env=env,capture_output=True,text=True)
        if check.returncode:
            raise RuntimeError(check.stderr)
        tests = subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-q"],
                               cwd=stage,env=env,capture_output=True,text=True)
        if tests.returncode:
            raise RuntimeError(tests.stderr[-6000:])
        count = int(re.search(r"Ran (\d+) tests", tests.stderr).group(1))
        invalid = subprocess.run([sys.executable,"tools/run_tgsc.py","train","--gpu","-1","--config",
                        "configs/tgsc/cifar100/dkd_resnet32x4_to_resnet8x4.json"],
                        cwd=stage,env=env,capture_output=True,text=True)
        assert invalid.returncode != 0 and "nonnegative" in invalid.stderr
        cpu = subprocess.run([sys.executable,"tools/run_tgsc.py","check","--cpu-check","--config",
                        "configs/tgsc/cifar100/dkd_resnet32x4_to_resnet8x4.json"],
                        cwd=stage,env=dict(env,CUDA_VISIBLE_DEVICES=""),capture_output=True,text=True)
        assert cpu.returncode != 0 and "--teacher-checkpoint" in cpu.stderr
        report = {"self_contained_tests": count,"result":"pass","notebooks_parsed":names,
                  "isolated_import":"pass","isolated_tests":"pass","cli_invalid_gpu":"pass",
                  "cli_cpu_check_reaches_input_validation":"pass","python":sys.version.split()[0],
                  "cuda_verified":False,"kaggle_executed":False}
        (root/"docs/TGSC_VALIDATION.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
        print(json.dumps(report,indent=2))
    finally:
        # Only remove the unique scratch directory verified to be within this checkout.
        if stage.resolve().is_relative_to(scratch):
            shutil.rmtree(stage)


if __name__ == "__main__":
    main()
