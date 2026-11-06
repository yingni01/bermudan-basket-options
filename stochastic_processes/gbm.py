import numpy as np
from stochastic_processes.stochastic_process import StochasticProcess

class GBM(StochasticProcess):

    def __init__(self, option_parameters, simulation_parameters):
        super().__init__(option_parameters, simulation_parameters)

    def simulate_underlying_prices(self):
        # Simulates the GBM
        # Only resturns the prices at the possible exercise times
        # This is the only data that is used for the pricing

        GBM_prices = np.zeros((self.num_paths, self.exercise_n + 1, self.dim))
        exercise_times = np.array(range(self.exercise_n+1))*int(self.n/self.exercise_n)
        j=1

        for i in range(1, self.n + 1):
            if i == 1:
              price = self.S0 #Initiating the start of the prices to be the specified start price S0
              GBM_prices[:, 0,:] = self.S0 #Initiating the start of the prices to be the specified start price S0

            Z = np.random.multivariate_normal(np.zeros(self.dim), self.correlation_matrix, size=self.num_paths)
            
            # Ying (may 09 2025): commented this line because sigma is not an attribute in superclass
            #decay = np.exp((self.r - self.div - 0.5 * np.square(self.sigma)) * self.h + self.sigma * np.sqrt(self.h)*Z)
            
            #Ying (updated to the superclass attribute volatility)
            decay = np.exp((self.r - self.div - 0.5 * np.square(self.volatility)) * self.h + self.volatility * np.sqrt(self.h)*Z)

            if i in exercise_times:
                GBM_prices[:, j, :] = decay * price
                j+=1
            price = decay * price

        return GBM_prices
