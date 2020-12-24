class TabuList:
    def __init__(self, max_length=None, *, max_count=None):
        self._length = max_length
        self._max_count = max_count
        self._tabu = list()
        self._counter = 0

    def is_tabu(self, move):
        if len(self._tabu) < 1:
            return False
        if isinstance(move, dict):
            if move['WP'] in set(self._tabu):
                return True
        if move in set(self._tabu):
            return True
        return False

    def keys_are_tabu(self, multiple_keys):
        intersection = set(self._tabu).intersection(set(multiple_keys))
        return len(intersection) > 0

    def add(self, move):
        if isinstance(move, dict):
            self._tabu.append(move['WP'])
        else:
            self._tabu.append(move)
        while len(self._tabu) > self._length:
            self._tabu.pop(0)

