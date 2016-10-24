import logging

from mann2 import model_lens


class ModelLensRecurrent(model_lens.ModelLens):

    def __init__(self, config_file, logger_name):
        super(ModelLensRecurrent, self).__init__()

    def setup_model_meta(self, config):
        self.description = config['meta']['description']

    def setup_logging(self, config, logger_name):
        log_fname = config['logging']['file_name']
        f = open(log_fname, 'w')
        f.close()

        logger = logging.getLogger(logger_name)
        logger.setLevel(config['logging']['base_level'])

        fh = logging.FileHandler(log_fname)
        fh.setLevel(config['logging']['file_level'])

        ch = logging.StreamHandler()
        ch.setLevel(config['logging']['console_level'])

        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)

        return logger
