from collections import OrderedDict
import numpy as np

class _NPYCache:
    def __init__(self, max_items=8, mmap_mode=None):
        self.max_items = int(max_items)
        self.mmap_mode = mmap_mode
        self._cache = OrderedDict()

    def get(self, path):
        path = str(path)
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]

        arr = np.load(path, mmap_mode=self.mmap_mode, allow_pickle=False)
        self._cache[path] = arr

        if len(self._cache) > self.max_items:
            self._cache.popitem(last=False)  # nic nie zamykamy

        return arr
