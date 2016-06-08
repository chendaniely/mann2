import model_watts


def main():
    # initialize the model
    model = model_watts.ModelWatts('config_model_watts.yaml', 'model_watts')

    # setup model
    model.setup_model_graph()
    # model.setup_model_run()

if __name__ == '__main__':
    main()
