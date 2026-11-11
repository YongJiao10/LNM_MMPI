import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "molecular_figures"
    / "analysis"
    / "prepare_metascape_upload.py"
)
SPEC = importlib.util.spec_from_file_location("prepare_metascape_upload", SCRIPT_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class PrepareMetascapeUploadTest(unittest.TestCase):
    def test_write_multiple_list_csv_uses_one_column_per_gene_list(self):
        inputs = [
            ("List_A", ["GENE1", "GENE2"]),
            ("List_B", ["GENE3"]),
        ]
        with tempfile.TemporaryDirectory() as temp_dir:
            output = Path(temp_dir) / "lists.csv"

            MODULE.write_multiple_list_csv(inputs, output)

            with output.open(newline="") as handle:
                rows = list(csv.reader(handle))

        self.assertEqual(
            rows,
            [
                ["List_A", "List_B"],
                ["GENE1", "GENE3"],
                ["GENE2", ""],
            ],
        )


if __name__ == "__main__":
    unittest.main()
