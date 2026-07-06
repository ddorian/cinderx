# Copyright (c) Meta Platforms, Inc. and affiliates.

# pyre-strict

"""A long-lived frame must not consume C stack per executed instruction.
A tail-call interpreter built without working musttail pins a frame per
dispatch edge; this test catches that at runtime."""

import subprocess
import sys
import unittest

import cinderx

cinderx.init()

from cinderx.test_support import ENCODING, subprocess_env


def run_snippet(source: str) -> "subprocess.CompletedProcess[str]":
    return subprocess.run(
        [sys.executable, "-c", source],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        encoding=ENCODING,
        env=subprocess_env(),
    )


class TailCallStackTests(unittest.TestCase):
    @unittest.skipUnless(sys.version_info >= (3, 14), "tail-call interpreter")
    def test_fstring_loop_does_not_leak_c_stack(self) -> None:
        # A broken build loses thousands of frames of headroom in this
        # loop; the threshold only allows for measurement noise.
        proc = run_snippet(
            """if 1:
            import cinderx.jit
            cinderx.jit.auto()
            import sys

            sys.setrecursionlimit(200_000)

            depth = 0

            def probe():
                global depth
                depth += 1
                probe()

            def headroom():
                global depth
                depth = 0
                try:
                    probe()
                except RecursionError:
                    pass
                return depth

            def main():
                acc = 0
                start = None
                for i in range(30_000):
                    acc += len(f"x{i}")
                    if i == 1_000:
                        start = headroom()
                lost = start - headroom()
                assert lost < 500, f"lost {lost} frames of C stack headroom"
                return acc

            main()
            """
        )
        self.assertEqual(proc.returncode, 0, proc.stdout)


if __name__ == "__main__":
    unittest.main()
