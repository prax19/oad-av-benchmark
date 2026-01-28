from collections import OrderedDict
import numpy as np

class _NPZCache:
    def __init__(self, max_items=8, mmap_mode=None):
        self.max_items = max_items
        self.mmap_mode = mmap_mode
        self._cache = OrderedDict()

    def get(self, path):
        path = str(path)
        if path in self._cache:
            self._cache.move_to_end(path)
            return self._cache[path]
        data = np.load(path, mmap_mode=self.mmap_mode, allow_pickle=False)
        self._cache[path] = data
        if len(self._cache) > self.max_items:
            _, old = self._cache.popitem(last=False)
            try:
                old.close()
            except Exception:
                pass
        return data
