class Evaluate:
    def __init__(self, estimator, generator, metrics):
        self.estimator = estimator
        self.generator = generator
        self.metrics = metrics

    def evaluate(self, n_samples, n_repetitions):
        # run the estimator n_repetitions times
        # and compute the metrics for each run
        raise NotImplementedError()

    def save_resutls(self, results, path):
        # save the results to a file
        raise NotImplementedError()
