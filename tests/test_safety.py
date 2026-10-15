"""Offline regression tests; no checkpoint, PyTorch, or network required."""
import hashlib
import importlib.util
import io
from pathlib import Path
import sys
import tempfile
import threading
import types
import unittest
from unittest.mock import Mock, patch

from orthrus_parallel.integrity import checked_path, verify_file
from scripts.download_model import download_file, HTTPSRedirectHandler


def manifest(data):
    return {'size': len(data), 'oid': hashlib.sha1(
        f'blob {len(data)}\0'.encode() + data).hexdigest()}


class DownloadTests(unittest.TestCase):
    def test_manifest_and_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / 'model'
            path.write_bytes(b'abc')
            self.assertEqual(verify_file(path, manifest(b'abc')), hashlib.sha256(b'abc').hexdigest())
            with self.assertRaises(ValueError):
                verify_file(path, manifest(b'abd'))
            for name in ('../escape', '/absolute', ''):
                with self.assertRaises(ValueError):
                    checked_path(root, name)
            (root / 'link').symlink_to(path)
            with self.assertRaises(ValueError):
                checked_path(root, 'link')

    def test_download_atomicity_and_cleanup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model'
            for data, succeeds in [(b'ab', False), (b'abcd', False), (b'abd', False), (b'abc', True)]:
                response = io.BytesIO(data)
                response.geturl = lambda: 'https://cdn.example/model'
                opener = Mock()
                opener.open.return_value = response
                with patch('scripts.download_model.urllib.request.build_opener', return_value=opener):
                    if succeeds:
                        download_file('https://example/model', path, manifest(b'abc'))
                        self.assertEqual(path.read_bytes(), b'abc')
                    else:
                        with self.assertRaises(ValueError):
                            download_file('https://example/model', path, manifest(b'abc'))
                        self.assertFalse(path.exists())
                self.assertEqual(list(Path(directory).glob('*.part')), [])

    def test_existing_destination_survives_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model'
            path.write_bytes(b'original')
            opener = Mock()
            opener.open.side_effect = OSError('connection interrupted')
            with patch('scripts.download_model.urllib.request.build_opener', return_value=opener):
                with self.assertRaises(OSError):
                    download_file('https://example/model', path, manifest(b'abc'))
            self.assertEqual(path.read_bytes(), b'original')
            self.assertEqual(list(Path(directory).glob('*.part')), [])

    def test_reject_http_and_downgrade(self):
        with self.assertRaises(ValueError):
            download_file('http://example/model', Path('unused'), manifest(b''))
        with self.assertRaises(ValueError):
            HTTPSRedirectHandler().redirect_request(None, None, 302, '', {}, 'http://example/model')


def session_class():
    """Import the real Session lifecycle with heavyweight inference dependencies stubbed."""
    generation = types.ModuleType('orthrus_parallel.generation')
    generation.baseline = generation.generate = Mock()
    selector = types.ModuleType('orthrus_parallel.selector')
    selector.Selector = Mock()
    name = 'orthrus_parallel._session_safety_test'
    spec = importlib.util.spec_from_file_location(name, Path(__file__).parents[1] / 'orthrus_parallel/session.py')
    module = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {'torch': types.ModuleType('torch'),
            'orthrus_parallel.generation': generation, 'orthrus_parallel.selector': selector}):
        spec.loader.exec_module(module)
    return module.Session


class SessionTests(unittest.TestCase):
    def make_session(self):
        cls = session_class()
        session = cls.__new__(cls)
        session._lock = threading.RLock()
        session._closed = False
        session.last = object()
        session.selector = types.SimpleNamespace(state=object())
        session.model = Mock()
        return session

    def test_exception_cleanup_and_close(self):
        session = self.make_session()
        session._run = Mock(side_effect=ValueError('inference failed'))
        with self.assertRaises(ValueError):
            session.run('test')
        session.model.model.norm.register_forward_hook.return_value.remove.assert_called_once()
        self.assertIsNone(session.last)
        self.assertIsNone(session.selector.state)
        session._run = Mock(return_value='recovered')
        self.assertEqual(session.run('test'), 'recovered')
        session.close()
        session.close()
        with self.assertRaises(RuntimeError):
            session.run('test')

    def test_requests_are_serialized(self):
        session = self.make_session()
        entered, release, attempted, second_entered = (threading.Event() for _ in range(4))
        errors = []
        def run(prompt, *args):
            if prompt == 'first':
                entered.set()
                if not release.wait(3):
                    raise TimeoutError('test timed out')
            else:
                second_entered.set()
            return prompt
        session._run = run
        def invoke(prompt):
            try:
                if prompt == 'second':
                    attempted.set()
                session.run(prompt)
            except Exception as error:
                errors.append(error)
        first = threading.Thread(target=invoke, args=('first',))
        second = threading.Thread(target=invoke, args=('second',))
        first.start()
        try:
            self.assertTrue(entered.wait(2))
            second.start()
            self.assertTrue(attempted.wait(2))
            self.assertFalse(second_entered.wait(0.05))
        finally:
            release.set()
            first.join(3)
            if second.ident is not None:
                second.join(3)
        self.assertFalse(errors)
        self.assertTrue(second_entered.is_set())
        self.assertFalse(first.is_alive() or second.is_alive())


if __name__ == '__main__':
    unittest.main()
