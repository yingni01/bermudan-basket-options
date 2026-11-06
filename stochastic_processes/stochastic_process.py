class StochasticProcess():

    def __init__(self, option_parameters, simulation_parameters):
        self.S0 = option_parameters["S0"]
        self.K = option_parameters["K"]
        self.r = option_parameters["r"]
        self.volatility = option_parameters["volatility"]
        self.div = option_parameters["div"]
        self.T = option_parameters["T"]
        self.dim = option_parameters["dim"]
        self.correlation_matrix = option_parameters["correlation_matrix"]

        self.n = simulation_parameters["n"]
        self.num_paths = simulation_parameters["num_paths"]
        self.exercise_n = simulation_parameters["exercise_n"]

        self.h = self.T/self.n # or delta T
