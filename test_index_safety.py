"""Synthetic index safety checks: python -m unittest test_index_safety -v."""

import ast
import importlib.util
import io
import json
import os
from pathlib import Path
import pickle
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock
import uuid
import zipfile

import numpy as np


ROOT = Path(__file__).resolve().parent
with mock.patch.dict(sys.modules, {"huggingface_hub": None}):
    spec = importlib.util.spec_from_file_location("index_hub_under_test", ROOT / "netryx_hub.py")
    hub = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hub)


def source_functions(filename, names, namespace):
    """Use the production functions without starting the GUI or ML models."""
    tree = ast.parse((ROOT / filename).read_text(encoding="utf-8"), filename=filename)
    tree.body = [node for node in tree.body
                 if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    exec(compile(tree, filename, "exec"), namespace)
    return namespace


def npz_bytes(**arrays):
    data = io.BytesIO()
    np.savez_compressed(data, **arrays)
    return data.getvalue()


class PickleCanary:
    def __init__(self, marker):
        self.marker = str(marker)

    def __reduce__(self):
        return eval, (f"__import__('pathlib').Path({self.marker!r}).write_text('executed')",)


class IndexSafetyTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / "output"

    def manifest(self, index_id="safe-index", **changes):
        manifest = dict(format_version=hub.BUNDLE_FORMAT_VERSION, index_id=index_id,
                        name="Synthetic", center_lat=0.0, center_lon=0.0,
                        radius_km=1, num_entries=1, descriptor_dim=2,
                        descriptor_model="MegaLoc")
        manifest.update(changes)
        return manifest

    def bundle(self, manifest, **members):
        bundle = self.root / "fixture.netryx"
        with zipfile.ZipFile(bundle, "w") as archive:
            archive.writestr("manifest.json", json.dumps(manifest))
            for name, data in members.items():
                archive.writestr(name, data)
        return bundle

    def pca_namespace(self):
        return source_functions("megaloc_utils.py",
                                {"_SavedPCA", "save_pca", "load_pca", "apply_pca"},
                                {"np": np, "pickle": pickle, "_pca_model": None})

    def test_import_and_individual_download_reject_invalid_ids_before_mkdir(self):
        bad_ids = ["../escape", "../../escape", str(self.root / "absolute"),
                   "a/b", "a\\b", "C:escape", ".", "..", "", None, 12,
                   "x" * 129, "name.", "CON", "nul", "LPT1", "a\x00b"]
        for index_id in bad_ids:
            for individual in (False, True):
                with self.subTest(index_id=index_id, individual=individual):
                    manifest = self.manifest(index_id)
                    bundle = self.bundle(manifest)
                    manifest_path = self.root / "manifest.json"
                    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
                    with mock.patch.object(hub.os, "makedirs", side_effect=AssertionError("mkdir attempted")), \
                            mock.patch.object(hub, "hf_hub_download", return_value=str(manifest_path), create=True) as download:
                        with self.assertRaises(ValueError):
                            if individual:
                                hub.NetryxHub.__new__(hub.NetryxHub)._download_individual("synthetic/repo", self.output)
                            else:
                                hub.extract_bundle(bundle, self.output)
                        self.assertEqual(download.call_count, int(individual))
                    self.assertFalse(self.output.exists())

    def test_both_import_paths_keep_valid_files_under_the_index_root(self):
        for individual in (False, True):
            with self.subTest(individual=individual):
                output = self.root / ("individual" if individual else "bundle")
                manifest = self.manifest()
                if individual:
                    manifest_path = self.root / "manifest.json"
                    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

                    def fetch(**kwargs):
                        if kwargs["filename"] == "manifest.json":
                            return str(manifest_path)
                        raise FileNotFoundError("Optional synthetic member absent")

                    with mock.patch.object(hub, "hf_hub_download", side_effect=fetch, create=True):
                        result = hub.NetryxHub.__new__(hub.NetryxHub)._download_individual("synthetic/repo", output)
                else:
                    result = hub.extract_bundle(self.bundle(manifest), output)
                self.assertEqual(result["index_id"], "safe-index")
                self.assertEqual(list(output.rglob("manifest.json")),
                                 [output / "indexes" / "safe-index" / "manifest.json"])

    def test_symlinked_index_or_member_cannot_redirect_writes(self):
        self.output.mkdir()
        outside = self.root / "outside"
        outside.mkdir()
        for member in (False, True):
            with self.subTest(member=member):
                indexes = self.output / "indexes"
                if member:
                    indexes.mkdir()
                    destination = indexes / "safe-index"
                    destination.mkdir()
                    target = outside / "manifest.json"
                    target.write_text("preserved", encoding="utf-8")
                    link = destination / "manifest.json"
                else:
                    link, target = indexes, outside
                try:
                    link.symlink_to(target, target_is_directory=not member)
                except OSError as exc:
                    self.skipTest(f"Symlink creation unavailable: {exc}")
                try:
                    with self.assertRaises(ValueError):
                        hub.extract_bundle(self.bundle(self.manifest()), self.output)
                    if member:
                        self.assertEqual(target.read_text(encoding="utf-8"), "preserved")
                    else:
                        self.assertEqual(list(outside.iterdir()), [])
                finally:
                    link.unlink()

    def test_legacy_bundle_is_refused_before_any_output(self):
        bundle = self.bundle(self.manifest(format_version="2.0"))
        with self.assertRaises(ValueError):
            hub.extract_bundle(bundle, self.output)
        self.assertFalse(self.output.exists())

    def test_object_metadata_and_pickle_models_are_not_extracted_or_executed(self):
        marker = self.root / "executed"
        payloads = {
            "metadata.npz": npz_bytes(panoids=np.array([PickleCanary(marker)], dtype=object)),
            "pca_model.pkl": pickle.dumps(PickleCanary(marker)),
        }
        for name, payload in payloads.items():
            with self.subTest(name=name):
                with self.assertRaises(ValueError):
                    hub.extract_bundle(self.bundle(self.manifest(), **{name: payload}), self.output)
                self.assertFalse(self.output.exists())
                self.assertFalse(marker.exists())

    def test_individual_download_does_not_publish_unsafe_arrays(self):
        manifest_path = self.root / "manifest.json"
        manifest_path.write_text(json.dumps(self.manifest()), encoding="utf-8")
        marker = self.root / "executed"
        payload_path = self.root / "object-metadata.npz"
        payload_path.write_bytes(npz_bytes(panoids=np.array([PickleCanary(marker)], dtype=object)))

        def fetch(**kwargs):
            if kwargs["filename"] == "manifest.json":
                return str(manifest_path)
            if kwargs["filename"] == "metadata.npz":
                return str(payload_path)
            raise FileNotFoundError("Optional synthetic member absent")

        with mock.patch.object(hub, "hf_hub_download", side_effect=fetch, create=True):
            with self.assertRaises(ValueError):
                hub.NetryxHub.__new__(hub.NetryxHub)._download_individual("synthetic/repo", self.output)
        self.assertFalse(list(self.output.rglob("manifest.json")))
        self.assertFalse(list(self.output.rglob("metadata.npz")))
        self.assertFalse(marker.exists())

    def test_pca_round_trip_preserves_transform_and_normalisation(self):
        namespace = self.pca_namespace()
        components = np.array([[1., 0., 0.], [0., 1., 0.]])
        mean = np.array([1., 2., 3.])
        variance = np.array([4., 9.])
        samples = np.array([[4., 8., 5.], [6., 9., 7.]])
        for whiten in (False, True):
            with self.subTest(whiten=whiten):
                namespace["_pca_model"] = SimpleNamespace(components_=components, mean_=mean,
                    explained_variance_=variance, whiten=whiten, n_components_=2)
                path = self.root / "model.pkl"
                namespace["save_pca"](path)
                with np.load(path, allow_pickle=False) as data:
                    self.assertEqual(data["format_version"].item(), 1)
                model = namespace["load_pca"](path)
                expected = (samples - mean) @ components.T
                if whiten:
                    expected /= np.sqrt(variance)
                np.testing.assert_allclose(model.transform(samples), expected)
                expected /= np.linalg.norm(expected, axis=1, keepdims=True)
                np.testing.assert_allclose(namespace["apply_pca"](samples), expected)

    def test_pca_legacy_pickle_is_refused_without_changing_active_state(self):
        namespace = self.pca_namespace()
        previous = object()
        namespace["_pca_model"] = previous
        marker = self.root / "executed"
        path = self.root / "model.pkl"
        path.write_bytes(pickle.dumps(PickleCanary(marker)))
        with self.assertRaises(ValueError):
            namespace["load_pca"](path)
        self.assertIs(namespace["_pca_model"], previous)
        self.assertFalse(marker.exists())

    def test_index_builder_writes_reloadable_numeric_pca(self):
        namespace = self.pca_namespace()
        parts = self.root / "parts"
        parts.mkdir()
        np.savez_compressed(parts / "megaloc_part_0.npz",
            descriptors=np.array([[1., 2.], [3., 4.]]),
            paths=np.array(["synthetic-a_0.npz", "synthetic-b_90.npz"]),
            lats=np.array([0., 0.]), lons=np.array([0.001, 0.002]))

        class SyntheticPCA:
            def __init__(self, n_components, whiten):
                self.whiten = whiten
                self.n_components_ = n_components

            def fit(self, samples):
                self.components_ = np.array([[1., 0.]])
                self.mean_ = samples.mean(axis=0)
                self.explained_variance_ = np.array([2.])
                self.explained_variance_ratio_ = np.array([1.])

            def transform(self, samples):
                return (samples - self.mean_) @ self.components_.T / np.sqrt(2.)

        writer = mock.Mock()
        builder = source_functions("test_super.py", {"build_compact_index"},
            {"np": np, "os": os, "uuid": uuid, "ACTIVE_ENCODER": "megaloc",
             "INDEXES_DIR": str(self.output), "MEGALOC_PARTS_DIR": str(parts),
             "INDEX_TARGET_DIM": 1, "EMB_CSV": str(self.root / "absent.csv"),
             "save_pca": namespace["save_pca"], "load_pca": namespace["load_pca"],
             "write_index_manifest": writer,
             "haversine_vec": lambda *args: np.zeros(2)})
        with mock.patch.dict(sys.modules,
                {"sklearn": SimpleNamespace(),
                 "sklearn.decomposition": SimpleNamespace(PCA=SyntheticPCA)}):
            self.assertTrue(builder["build_compact_index"]())
        writer.assert_called_once()
        model_path = Path(builder["COMPACT_PCA_PATH"])
        with np.load(model_path, allow_pickle=False) as data:
            self.assertEqual(data["format_version"].item(), 1)
        model = namespace["load_pca"](model_path)
        np.testing.assert_allclose(model.transform(np.array([[3., 4.]])), [[1. / np.sqrt(2.)]])
        with np.load(builder["COMPACT_META_PATH"], allow_pickle=False) as data:
            self.assertEqual(data["panoids"].dtype.kind, "U")

    def test_invalid_pca_parameters_preserve_the_active_model(self):
        namespace = self.pca_namespace()
        valid = dict(format_version=np.array(1), components=np.eye(2),
                     mean=np.zeros(2), variance=np.ones(2), whiten=np.array(True))
        invalid = [dict(mean=np.zeros(3)), dict(variance=np.array([-1., 1.])),
                   dict(components=np.array([[np.nan, 0.], [0., 1.]])),
                   dict(components=np.array([["x", "y"], ["a", "b"]])),
                   dict(whiten=np.array(1)), dict(format_version=np.array(2))]
        for changes in invalid:
            with self.subTest(changes=tuple(changes)):
                path = self.root / "invalid-pca.npz"
                path.write_bytes(npz_bytes(**(valid | changes)))
                previous = object()
                namespace["_pca_model"] = previous
                with self.assertRaises(ValueError):
                    namespace["load_pca"](path)
                self.assertIs(namespace["_pca_model"], previous)

    def test_safe_export_import_round_trip(self):
        source = self.root / "source"
        source.mkdir()
        descriptors = np.array([[1., 0.], [0., 1.]], dtype=np.float32)
        np.save(source / "megaloc_descriptors.npy", descriptors)
        np.savez_compressed(source / "metadata.npz", lats=np.array([0., 0.001]),
            lons=np.array([0., 0.001]), headings=np.array([0., 90.]),
            panoids=np.array(["synthetic-a", "synthetic-b"]),
            paths=np.array(["synthetic/a.jpg", "synthetic/b.jpg"]))
        namespace = self.pca_namespace()
        namespace["_pca_model"] = SimpleNamespace(components_=np.eye(2), mean_=np.zeros(2),
            explained_variance_=np.ones(2), whiten=False, n_components_=2)
        namespace["save_pca"](source / "megaloc_pca.pkl")
        bundle, exported = hub.create_bundle(source, self.root / "safe.netryx", "Synthetic", "Fixture", 0., 0., 1.)
        imported = hub.extract_bundle(bundle, self.output)
        self.assertEqual(imported["index_id"], exported["index_id"])
        target = self.output / "indexes" / imported["index_id"]
        np.testing.assert_array_equal(np.load(target / "megaloc_descriptors.npy", allow_pickle=False), descriptors)
        with np.load(target / "metadata.npz", allow_pickle=False) as data:
            self.assertEqual(data["panoids"].dtype.kind, "U")
        np.testing.assert_allclose(namespace["load_pca"](target / "megaloc_pca.pkl").transform(descriptors), descriptors)

    def test_export_refuses_legacy_pca_before_publishing(self):
        source = self.root / "source"
        source.mkdir()
        np.save(source / "megaloc_descriptors.npy", np.array([[1., 0.]], dtype=np.float32))
        np.savez_compressed(source / "metadata.npz", lats=np.array([0.]),
            lons=np.array([0.]), headings=np.array([0.]),
            panoids=np.array(["synthetic"]), paths=np.array(["synthetic/a.jpg"]))
        marker = self.root / "executed"
        (source / "megaloc_pca.pkl").write_bytes(pickle.dumps(PickleCanary(marker)))
        output = self.root / "legacy.netryx"
        with self.assertRaises(ValueError):
            hub.create_bundle(source, output, "Synthetic", "Fixture", 0., 0., 1.)
        self.assertFalse(output.exists())
        self.assertFalse(marker.exists())

    def test_cached_object_metadata_is_refused_without_executing_it(self):
        marker = self.root / "executed"
        descriptors = self.root / "descriptors.npy"
        np.save(descriptors, np.array([[1., 2.]]))
        metadata = self.root / "metadata.npz"
        metadata.write_bytes(npz_bytes(lats=np.array([0.]), lons=np.array([0.]),
            headings=np.array([0.]), panoids=np.array([PickleCanary(marker)], dtype=object),
            paths=np.array(["synthetic"])))
        namespace = source_functions("test_super.py", {"load_compact_index"},
            {"np": np, "os": os, "time": time, "_compact_cache": None,
             "COMPACT_DESCS_PATH": str(descriptors), "COMPACT_META_PATH": str(metadata),
             "has_active_index": lambda: True})
        with self.assertRaises(ValueError):
            namespace["load_compact_index"]()
        self.assertIsNone(namespace["_compact_cache"])
        self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main()
