import numpy as np

from stochastic_processes.stochastic_process import StochasticProcess

class Heston(StochasticProcess):

    def __init__(self, option_parameters, simulation_parameters, heston_parameters):
        super().__init__(option_parameters, simulation_parameters)
        self.theta = heston_parameters["theta"]
        self.xi = heston_parameters["xi"]
        self.kappa = heston_parameters["kappa"]
        self.correlation_matrix = heston_parameters["correlation_matrix"]


    def simulate_underlying_prices(self):
        # Simulates the heston model
        # Only resturns the prices at the possible exercise times
        # This is the only data that is used for the pricing

        heston_prices = np.zeros((self.num_paths, self.exercise_n + 1, self.dim))
        exercise_times = np.array(range(self.exercise_n+1))*int(self.n/self.exercise_n)
        j=1

        for i in range(1, self.n + 1):
            if i == 1:
              price = self.S0 #Initiating the start of the prices to be the specified start price S0
              heston_prices[:, 0, :] = self.S0
              sigma = np.ones((self.num_paths, self.dim))*self.volatility

            Z = np.random.multivariate_normal(np.zeros(self.dim*2), self.correlation_matrix, size=self.num_paths)
            Z_S = Z[:, :self.dim]
            Z_sigma = Z[:, self.dim:]

            new_sigma = sigma + self.kappa*(self.theta - sigma.clip(0))*self.h + self.xi * np.sqrt(sigma.clip(0)) * np.sqrt(self.h)*Z_sigma
            bias = np.mean(np.sqrt(sigma) * Z_S * np.sqrt(self.h))

            decay = np.exp((self.r - self.div - 0.5 * sigma) * self.h + np.sqrt(sigma) * np.sqrt(self.h)*Z_S - bias)  # moment matching
            if i in exercise_times:
                heston_prices[:, j, :] = decay * price
                j+=1

            price = decay * price
            sigma = new_sigma.clip(0)

        return heston_prices
