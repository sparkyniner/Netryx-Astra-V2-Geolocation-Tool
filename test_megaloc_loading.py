"""Local checkpoints must use the restricted loader, without real models or providers."""
import ast
import contextlib
import importlib.util
import io
import os
import pickle
import sys
import tempfile
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


CALLBACKS = []


def checkpoint_callback():
    CALLBACKS.append("unpickled")
    return {}


class CustomCheckpoint:
    def __reduce__(self):
        return checkpoint_callback, ()


class ModelLoadingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.weights = self.root / "megaloc_weights.pth"
        self.weights.write_bytes(b"synthetic fixture")
        source = Path(__file__).with_name("megaloc_utils.py")
        tree = ast.parse(source.read_text(encoding="utf-8"))
        self.function = next(node for node in tree.body
                             if isinstance(node, ast.FunctionDef) and node.name == "get_megaloc_model")

    def load(self, loader, model=None, hub_model=None):
        model = model if model is not None else Mock()
        if isinstance(model, Mock):
            model.mock_add_spec(["load_state_dict", "eval", "to", "parameters"])
            model.eval.return_value = model
            model.to.return_value = model
            model.parameters.return_value = []
        hub = Mock()
        if hub_model is None:
            hub.load.side_effect = RuntimeError("synthetic offline hub")
        else:
            hub.load.return_value = hub_model
        namespace = {"__file__": str(self.root / "megaloc_utils.py"), "os": os,
                     "torch": SimpleNamespace(hub=hub, load=loader, bfloat16="synthetic dtype"),
                     "_megaloc_model": None, "_megaloc_lock": threading.Lock(), "_device": "cpu",
                     "autocast": lambda *args, **kwargs: contextlib.nullcontext(), "MEGALOC_RAW_DIM": 8448}
        exec(compile(ast.Module(body=[self.function], type_ignores=[]), "megaloc_utils.py", "exec"), namespace)
        with patch.dict(sys.modules, {"megaloc_model": SimpleNamespace(MegaLoc=lambda: model)}):
            with contextlib.redirect_stdout(io.StringIO()):
                result = namespace["get_megaloc_model"]("cpu")
        return result, model

    def test_local_state_dict_uses_explicit_restricted_cpu_loader(self):
        state = {"synthetic.weight": object()}
        loader = Mock(return_value=state)
        result, model = self.load(loader)
        loader.assert_called_once_with(str(self.weights), map_location="cpu", weights_only=True)
        model.load_state_dict.assert_called_once_with(state)
        self.assertIs(result, model)
        model.to.assert_called_once_with("cpu")

    def test_restricted_load_failure_has_no_unsafe_retry(self):
        loader = Mock(side_effect=pickle.UnpicklingError("synthetic refused checkpoint"))
        with self.assertRaisesRegex(pickle.UnpicklingError, "synthetic refused checkpoint"):
            self.load(loader)
        self.assertEqual(loader.call_count, 1)
        self.assertIs(loader.call_args.kwargs.get("weights_only"), True)

    def test_missing_local_weights_does_not_load(self):
        self.weights.unlink()
        loader = Mock()
        with self.assertRaisesRegex(RuntimeError, "Could not load MegaLoc"):
            self.load(loader)
        loader.assert_not_called()

    def test_successful_hub_load_does_not_open_local_checkpoint(self):
        loader = Mock()
        model = Mock()
        result, _ = self.load(loader, model, model)
        self.assertIs(result, model)
        loader.assert_not_called()
        model.load_state_dict.assert_not_called()

    def test_supported_tensor_state_dict_round_trip(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("PyTorch is not installed in this isolated runner")
        import torch
        expected = torch.nn.Linear(2, 1)
        torch.save(expected.state_dict(), self.weights)
        result, _ = self.load(torch.load, torch.nn.Linear(2, 1))
        for key, value in expected.state_dict().items():
            self.assertTrue(torch.equal(result.state_dict()[key], value))
        self.assertTrue(all(not parameter.requires_grad for parameter in result.parameters()))

    def test_custom_checkpoint_cannot_override_restricted_loading(self):
        if importlib.util.find_spec("torch") is None:
            self.skipTest("PyTorch is not installed in this isolated runner")
        import torch
        torch.save(CustomCheckpoint(), self.weights)
        CALLBACKS.clear()
        error = None
        with patch.dict(os.environ, {"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"}):
            try:
                self.load(torch.load, torch.nn.Module())
            except pickle.UnpicklingError as caught:
                error = caught
        self.assertEqual(CALLBACKS, [], "checkpoint callback must never execute")
        self.assertIsInstance(error, pickle.UnpicklingError)


if __name__ == "__main__":
    unittest.main()
