"""Flask route tests (test_client, fake service, no live server, no EGL)."""

from __future__ import annotations

import unittest

from code.apps.fleet_web.app import create_app, mjpeg_frames
from code.apps.fleet_web.service import FleetService
from code.apps.fleet_web.tests.fakes import FakeEngine


class _TinyFrameService:
    """Minimal service exposing just what the MJPEG generator reads."""

    stopped = False

    def latest_jpeg(self):
        return b"ABC123"


class RouteTest(unittest.TestCase):
    def setUp(self):
        # Unstarted service (routes only read snapshots / validate commands).
        self.svc = FleetService(FakeEngine())
        self.app = create_app(self.svc)
        self.app.testing = True
        self.client = self.app.test_client()

    def test_index_serves_page(self):
        r = self.client.get("/")
        self.assertEqual(r.status_code, 200)
        self.assertIn("text/html", r.content_type)
        self.assertIn(b"Warehouse Fleet", r.data)

    def test_state_shape(self):
        r = self.client.get("/state")
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertEqual(set(body),
                         {"robots", "transcript", "next_id", "status", "mission"})

    def test_state_accepts_after_param(self):
        r = self.client.get("/state?after=3")
        self.assertEqual(r.status_code, 200)
        self.assertIsInstance(r.get_json()["transcript"], list)

    def test_command_valid(self):
        r = self.client.post("/command",
                             json={"text": "Alpha, fetch the red cube to the pad"})
        self.assertEqual(r.status_code, 200)
        body = r.get_json()
        self.assertTrue(body["ok"])
        self.assertFalse(body["queued"])

    def test_command_unknown_callsign(self):
        body = self.client.post(
            "/command", json={"text": "Zulu, fetch the red cube"}).get_json()
        self.assertFalse(body["ok"])
        self.assertIn("Zulu", body["error"])

    def test_command_unresolvable_object(self):
        body = self.client.post(
            "/command", json={"text": "Alpha, hello there"}).get_json()
        self.assertFalse(body["ok"])

    def test_command_missing_text(self):
        body = self.client.post("/command", json={}).get_json()
        self.assertFalse(body["ok"])

    def test_command_queued_notice(self):
        self.svc._active = True
        body = self.client.post(
            "/command", json={"text": "Bravo, bring the blue ball"}).get_json()
        self.assertTrue(body["ok"])
        self.assertTrue(body["queued"])

    def test_stream_content_type(self):
        r = self.client.get("/stream")
        self.assertEqual(r.mimetype, "multipart/x-mixed-replace")
        self.assertIn("boundary=frame", r.headers["Content-Type"])
        r.close()

    def test_mjpeg_frames_chunk_is_well_formed(self):
        gen = mjpeg_frames(_TinyFrameService(), max_frames=1)
        chunk = next(gen)
        self.assertTrue(chunk.startswith(b"--frame"))
        self.assertIn(b"Content-Type: image/jpeg", chunk)
        self.assertIn(b"ABC123", chunk)


if __name__ == "__main__":
    unittest.main()
