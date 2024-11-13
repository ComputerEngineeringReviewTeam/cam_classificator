from json import JSONEncoder


def _default(self, obj):
    return getattr(obj.__class__, "__json__", _default.default)(obj)

_default.default = JSONEncoder().default
JSONEncoder.default = _default
