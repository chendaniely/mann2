from mann2 import model_lens


class ModelLensCascade(model_lens.ModelLens):

    def __init__(self, config_file, logger_name):
        super(ModelLensCascade, self).__init__()
