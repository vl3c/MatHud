"""The server suite must not see real provider keys unless live tests are requested.

Assertions never include key values in their messages, so a failure cannot leak a key.
"""

from __future__ import annotations

import os
import unittest

from server_tests.conftest import PROVIDER_KEY_VARS, live_tests_enabled


def _looks_real(value: str) -> bool:
    return value.startswith("sk-") or len(value) > 32


class TestLiveTestsGuard(unittest.TestCase):
    def setUp(self) -> None:
        if live_tests_enabled():
            self.skipTest("MATHUD_LIVE_TESTS is set; real keys are allowed")

    def _assert_no_real_keys(self) -> None:
        leaked = [key for key in PROVIDER_KEY_VARS if _looks_real(os.environ.get(key) or "")]
        self.assertEqual(leaked, [], "real provider keys visible to tests")

    def test_provider_keys_are_not_real_without_opt_in(self) -> None:
        self._assert_no_real_keys()

    def test_env_files_do_not_restore_keys(self) -> None:
        from static.env_config import load_env_files

        load_env_files()
        self._assert_no_real_keys()

    def test_env_files_do_not_restore_a_deleted_key(self) -> None:
        # Tests that save and restore a key may delete it; a later .env load must not refill it.
        from static.env_config import load_env_files

        saved = {key: os.environ.get(key) for key in PROVIDER_KEY_VARS}
        try:
            for key in PROVIDER_KEY_VARS:
                os.environ.pop(key, None)
            load_env_files()
            self._assert_no_real_keys()
        finally:
            for key, value in saved.items():
                if value is None:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value
