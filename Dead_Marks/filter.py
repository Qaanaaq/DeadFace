# filter.py – Simplified EMA smoothing for blendshapes

class EMAFilter1D:
    """Exponential Moving Average filter for a single value."""
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.value = None  # No previous value yet

    def update(self, new_value):
        if self.value is None:
            self.value = new_value
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value


class BlendshapeEMAFilter:
    """EMA filter applied to all blendshapes."""
    def __init__(self, alpha=0.5):
        self.alpha = alpha
        self.filters = {}

    def set_alpha(self, alpha):
        """Update smoothing factor for all filters."""
        self.alpha = alpha
        for f in self.filters.values():
            f.alpha = alpha

    def apply(self, blendshape_dict):
        """Apply EMA filter to each blendshape value."""
        return {
            name: self.filters.setdefault(name, EMAFilter1D(self.alpha)).update(value)
            for name, value in blendshape_dict.items()
        }

    def reset(self):
        """Reset all filters (e.g., after neutral pose recalibration)."""
        self.filters.clear()

