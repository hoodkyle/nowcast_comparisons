"""Check repository contents and reproduce every paper table without source data."""
from pathlib import Path
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReproductionTests(unittest.TestCase):
    def test_repository_reproduces_all_tables_without_local_files(self):
        names = subprocess.check_output(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z"],
            cwd=ROOT,
        ).decode().split("\0")
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory)
            for name in sorted(set(names) - {""}):
                source = ROOT / name
                if not source.exists():
                    continue  # Pending deletions in a working checkout.
                parts = Path(name).parts
                self.assertFalse(set(parts) & {"paper", "data", "output", ".git", "__pycache__"})
                self.assertNotIn(Path(name).suffix, {".tex", ".bib", ".pdf", ".gz", ".xlsx", ".zip"})
                destination = checkout / name
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
            # Regenerate from repository files without local source observations.
            bootstrap = '''
import importlib.abc, runpy, sys
class TableDependenciesOnly(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in {'torch', 'scipy', 'sklearn', 'threadpoolctl', 'catboost', 'xgboost'}:
            raise ModuleNotFoundError('Estimation dependency unavailable: ' + fullname)
sys.meta_path.insert(0, TableDependenciesOnly())
sys.argv = ['reproduce.py', *sys.argv[1:]]
runpy.run_path('reproduce.py', run_name='__main__')
'''
            for command in ("tables", "mrts-lag-tables"):
                subprocess.run([sys.executable, "-c", bootstrap, command, "--output", "output/tables"],
                               cwd=checkout, check=True, capture_output=True, text=True)
            expected = set(json.loads((checkout / "config/tables.json").read_text()))
            self.assertEqual(len(expected), 21)
            for extension in ("csv", "tex"):
                self.assertEqual({p.stem for p in (checkout / "output/tables").glob("*." + extension)}, expected)
            with (checkout / "output/tables/main_headline.csv").open() as stream:
                rows = {row["Specification"]: row for row in csv.DictReader(stream)}
            self.assertEqual(round(float(rows["Single-month CatBoost"]["RMSE"]), 3), 2.230)
            self.assertEqual(round(float(rows["Single-month CatBoost"]["MAE"]), 3), 1.616)


if __name__ == "__main__":
    unittest.main()
