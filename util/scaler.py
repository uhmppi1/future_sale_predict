import numpy as np

class MinMaxScaler():
    def __init__(self, min_value, max_value):
        assert (max_value > min_value)
        self.min_value = min_value
        self.max_value = max_value

    def scale_value(self, val):
        return (val - self.min_value) / (self.max_value - self.min_value)

    def inv_scale_value(self, scaled_val):
        return self.min_value + scaled_val * (self.max_value - self.min_value)


class LogMinMaxScaler():
    def __init__(self, min_value, max_value, pad_avoid_log0=1):
        assert (max_value > min_value)
        self.value_shift = -min_value + pad_avoid_log0
        self.min_value = min_value + self.value_shift # to avoid np.log(0).. assert(self.min_value > 0)
        self.max_value = max_value + self.value_shift
        assert (self.min_value > 0)
        self.min_logvalue = np.log(self.min_value)
        self.max_logvalue = np.log(self.max_value)

    def scale_value(self, val):
        val_log = np.log(val + self.value_shift)
        return (val_log - self.min_logvalue) / (self.max_logvalue - self.min_logvalue)

    def inv_scale_value(self, scaled_val):
        val_log = self.min_logvalue + scaled_val * (self.max_logvalue - self.min_logvalue)
        return np.exp(val_log) - self.value_shift