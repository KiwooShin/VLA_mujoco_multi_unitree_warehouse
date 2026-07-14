"""EGL smoke: compose 3 real frames on a tiny rooms mission (oracle+teacher).

Asserts the canvas size and that EVERY robot's ego tile is non-black (the hard
"every camera visible at all times" requirement). Heavy (builds a 4-robot fleet
+ MuJoCo renderers) but bounded: it cancels the run after 3 composed frames.
"""

from __future__ import annotations

import unittest

import numpy as np

# Importing the package triggers code/__init__'s NVIDIA EGL vendor pin (before
# any MuJoCo context is created during discovery).
import code.apps.demos  # noqa: F401
from code.apps.demos import style
from code.apps.demos.composer import DemoComposer
from code.apps.demos.layout import ego_area
from code.apps.demos.runner_adapter import frame_state_from_runner


class TestComposeSmoke(unittest.TestCase):
    def test_compose_three_real_frames_every_tile_nonblack(self):
        try:
            import mujoco  # noqa: F401
        except Exception:
            self.skipTest("mujoco unavailable")
        from code.fleet.mission import MissionRunner
        from code.warehouse.layout import callsigns_for_layout, rooms_layout

        layout = rooms_layout()
        callsigns = callsigns_for_layout(layout)
        mr = MissionRunner(layout=layout, callsigns=callsigns, seed=0,
                           use_gpu=True, perception_mode="oracle",
                           locomotion="teacher", search_deadline_steps=200)
        # Title card off so the three sampled frames are the live composition
        # (the card path is covered by test_effects); capture after a few steps
        # so the egos show settled scene content.
        composer = DemoComposer(mr.fleet.viz, layout.hall_x, layout.hall_y,
                                title="Smoke Test", description="tiny rooms mission",
                                title_card_secs=0.0)
        card_composer = DemoComposer(mr.fleet.viz, layout.hall_x, layout.hall_y,
                                     title="Smoke Test", description="card path",
                                     title_card_secs=2.0)
        frames = []
        card_frame = None
        try:
            mr.submit("Alpha, fetch the red cube to the delivery pad")

            def on_step(runner, t):
                nonlocal card_frame
                if t == 0:  # exercise the title-card blend path on a real frame
                    card_frame = card_composer.compose(
                        frame_state_from_runner(runner, t))
                if t >= 5 and t % 3 == 0 and len(frames) < 3:
                    frames.append(
                        composer.compose(frame_state_from_runner(runner, t)))
                return len(frames) < 3   # cancel once we have 3 frames

            mr.run(60, on_step=on_step)
        finally:
            composer.close()
            card_composer.close()
            mr.close()

        self.assertEqual(len(frames), 3)
        self.assertEqual(card_frame.shape, (style.CANVAS_H, style.CANVAS_W, 3))
        tiles = composer.tiles(len(callsigns))
        for frame in frames:
            self.assertEqual(frame.shape, (style.CANVAS_H, style.CANVAS_W, 3))
            self.assertEqual(frame.dtype, np.uint8)
            for tile in tiles:
                ego = ego_area(tile)
                patch = frame[ego.y:ego.y1, ego.x:ego.x1]
                self.assertGreater(patch.mean(), 15.0)   # a real render, non-black
                self.assertGreater(patch.std(), 10.0)    # actual scene content


if __name__ == "__main__":
    unittest.main()
