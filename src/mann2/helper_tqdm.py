class HelperTqdm():

    def parse_tqdm_config(self, config):
        show = not config['misc']['progress_bar']['show']
        position = config['misc']['progress_bar']['base_position']
        sim_number = config['meta']['sim_number']
        return((sim_number, show, position))
