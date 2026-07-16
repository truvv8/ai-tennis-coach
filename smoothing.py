"""Exponential smoothing for keypoints. Raw trt_pose output jitters a few
pixels per frame, which turns into +-10 degrees on the angles and makes the
feedback flicker."""


class KeypointSmoother:
    def __init__(self, alpha=0.5, max_gap=5):
        """alpha: EMA weight of the new frame (1.0 = no smoothing).
        max_gap: hold a briefly-lost keypoint for this many frames before
        dropping it, so angles don't blink on single-frame detection misses."""
        self.alpha = alpha
        self.max_gap = max_gap
        self._state = {}  # index -> [x, y, frames_since_seen]

    def reset(self):
        self._state.clear()

    def update(self, kps):
        if kps is None:
            self.reset()
            return None

        for j, (x, y) in kps.items():
            if j in self._state:
                s = self._state[j]
                s[0] += self.alpha * (x - s[0])
                s[1] += self.alpha * (y - s[1])
                s[2] = 0
            else:
                self._state[j] = [x, y, 0]

        for j in list(self._state):
            if j not in kps:
                self._state[j][2] += 1
                if self._state[j][2] > self.max_gap:
                    del self._state[j]

        return {j: (s[0], s[1]) for j, s in self._state.items()}
