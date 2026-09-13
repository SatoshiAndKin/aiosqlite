# Copyright Amethyst Reese
# Licensed under the MIT license

import asyncio
import sqlite3
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from unittest import TestCase
from unittest.mock import patch

import aiosqlite


class QueueShutdownTest(TestCase):
    def test_closed_loop_drains_queued_statements(self):
        for fail in (False, True):
            with self.subTest(fail=fail):
                self._check_queue(fail=fail, race=False)

    def test_error_delivery_races_loop_close(self):
        self._check_queue(fail=True, race=True)

    def _check_queue(self, *, fail, race):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        started, release = Event(), Event()
        connection = None
        tasks = []
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "queued.sqlite"

            async def prepare():
                nonlocal connection
                connection = await aiosqlite.connect(
                    path, isolation_level=None, check_same_thread=False
                )
                await connection.execute("CREATE TABLE writes(value INTEGER)")

                def gated_value():
                    started.set()
                    if not release.wait(5):
                        raise TimeoutError("queued statement was not released")
                    if fail:
                        raise ValueError("failed queued statement")
                    return 1

                await connection.create_function("gated_value", 0, gated_value)

                async def write(statement):
                    await connection.execute(statement)

                tasks.extend(
                    (
                        asyncio.create_task(
                            write("INSERT INTO writes VALUES (gated_value())")
                        ),
                        asyncio.create_task(write("INSERT INTO writes VALUES (2)")),
                    )
                )
                await asyncio.sleep(0)

            with patch("threading.excepthook") as thread_error:
                try:
                    loop.run_until_complete(prepare())
                    self.assertTrue(started.wait(5))
                    self.assertEqual(connection._tx.qsize(), 1)
                    for task in tasks:
                        task.cancel()
                    loop.run_until_complete(
                        asyncio.gather(*tasks, return_exceptions=True)
                    )
                    notify = loop.call_soon_threadsafe

                    def close_before_delivery(*args, **kwargs):
                        loop.close()
                        return notify(*args, **kwargs)

                    if not race:
                        loop.close()
                    with patch.object(
                        loop,
                        "call_soon_threadsafe",
                        side_effect=close_before_delivery if race else notify,
                    ):
                        connection.stop()
                        release.set()
                        connection._thread.join(5)
                    self.assertFalse(connection._thread.is_alive())
                    self.assertIsNone(connection._connection)
                    thread_error.assert_not_called()
                    with sqlite3.connect(path) as reader:
                        rows = reader.execute(
                            "SELECT value FROM writes ORDER BY value"
                        ).fetchall()
                    self.assertEqual(rows, [(2,)] if fail else [(1,), (2,)])
                finally:
                    release.set()
                    if connection is not None:
                        connection.stop()
                        connection._thread.join(5)
                        if connection._connection is not None:
                            connection._connection.close()
                    loop.close()
                    asyncio.set_event_loop(None)

    def test_stop_delivery_races_loop_close(self):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        connection = loop.run_until_complete(aiosqlite.connect(":memory:"))
        notify = loop.call_soon_threadsafe

        def close_before_delivery(*args, **kwargs):
            loop.close()
            return notify(*args, **kwargs)

        with patch("threading.excepthook") as thread_error:
            try:
                with patch.object(
                    loop, "call_soon_threadsafe", side_effect=close_before_delivery
                ):
                    connection.stop()
                    connection._thread.join(5)
                self.assertFalse(connection._thread.is_alive())
                self.assertIsNone(connection._connection)
                thread_error.assert_not_called()
            finally:
                connection.stop()
                connection._thread.join(5)
                loop.close()
                asyncio.set_event_loop(None)
