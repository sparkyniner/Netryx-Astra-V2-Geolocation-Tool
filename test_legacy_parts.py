"""Legacy part file checks: python -m unittest test_legacy_parts -v."""

import contextlib
import io
import os
from pathlib import Path
import tempfile
import time
import unittest
import uuid
import zipfile

import numpy as np
from numpy.lib import format as npy_format

from test_index_safety import PickleCanary, source_functions


def write_part(path, paths):
    """Write a part file the way older versions did."""
    np.savez(path, descriptors=np.array([[1., 2.], [3., 4.]], dtype=np.float32),
             paths=paths, lats=np.array([1., 2.], dtype=np.float32),
             lons=np.array([3., 4.], dtype=np.float32))


def app_functions(names, **namespace):
    return source_functions("test_super.py", {"legacy_files", "legacy_parts_message"} | names,
                            {"np": np, "npy_format": npy_format, "os": os,
                             "zipfile": zipfile, **namespace})


class LegacyPartTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.marker = self.root / "executed"
        self.parts = self.root / "megaloc_parts"
        self.parts.mkdir()
        self.part = self.parts / "megaloc_part_1.npz"
        write_part(self.part, np.array([PickleCanary(self.marker), "b_90.npz"], dtype=object))
        self.before = self.part.read_bytes()

    def assert_untouched(self):
        self.assertFalse(self.marker.exists())
        self.assertEqual(self.part.read_bytes(), self.before)

    def test_only_object_array_files_are_reported(self):
        current = self.parts / "megaloc_part_2.npz"
        write_part(current, np.array(["a_0.npz", "b_90.npz"]))
        broken = self.parts / "megaloc_part_3.npz"
        broken.write_bytes(b"not a zip file")
        app = app_functions(set())
        self.assertEqual(app["legacy_files"]([str(self.part), str(current), str(broken)]),
                         [str(self.part)])
        self.assert_untouched()

    def test_index_build_stops_and_names_the_old_part_files(self):
        app = app_functions({"build_compact_index"}, uuid=uuid, ACTIVE_ENCODER="megaloc",
                            INDEXES_DIR=str(self.root / "indexes"),
                            MEGALOC_PARTS_DIR=str(self.parts), INDEX_TARGET_DIM=1)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertFalse(app["build_compact_index"]())
        message = output.getvalue()
        self.assertIn(str(self.parts), message)
        self.assertIn("megaloc_part_1.npz", message)
        self.assertIn("can only be read by unpickling", message)
        self.assertIn("Move or delete them, then run Create again", message)
        self.assert_untouched()

    def test_old_index_metadata_gets_a_rebuild_message(self):
        descriptors = self.root / "descriptors.npy"
        np.save(descriptors, np.array([[1., 2.]]))
        metadata = self.root / "metadata.npz"
        np.savez_compressed(metadata, lats=np.array([0.]), lons=np.array([0.]),
                            headings=np.array([0]),
                            panoids=np.array([PickleCanary(self.marker)], dtype=object),
                            paths=np.array(["a_0.npz"], dtype=object))
        before = metadata.read_bytes()
        app = app_functions({"load_compact_index"}, time=time, _compact_cache=None,
                            COMPACT_DESCS_PATH=str(descriptors),
                            COMPACT_META_PATH=str(metadata), has_active_index=lambda: True)
        with self.assertRaisesRegex(ValueError, "unpickling.*Rebuild this index"):
            app["load_compact_index"]()
        self.assertIsNone(app["_compact_cache"])
        self.assertEqual(metadata.read_bytes(), before)
        self.assertFalse(self.marker.exists())


if __name__ == "__main__":
    unittest.main()
