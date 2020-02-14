class fitVal():
    def __init__(self):
        self.fit_val = {}
        self.fit_val['rotation_range']        = 10
        self.fit_val['horizontal_flip']       = True
        self.fit_val['vertical_flip']         = False
        self.fit_val['width_shift_range']     = 0.0
        self.fit_val['height_shift_range']    = 0.0
        self.fit_val['zoom_range']            = [1.0, 1.0]
        self.fit_val['brightness_range']      = [0.9, 1.1]
