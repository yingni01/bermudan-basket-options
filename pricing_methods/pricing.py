import numpy as np
import time

class PricingModel():

    def __init__(self, option_parameters, simulation_parameters, pricing_parameters, neural_network):
        self.dim = option_parameters["dim"]
        self.K = option_parameters["K"]
        self.T = option_parameters["T"]
        self.r = option_parameters["r"]
        self.payoff_type = option_parameters["payoff_type"]
        self.option_type = option_parameters["option_type"]

        self.n = simulation_parameters["n"]
        self.num_paths = simulation_parameters["num_paths"]
        self.exercise_n = simulation_parameters["exercise_n"]

        self.polynominal_deg = pricing_parameters["polynominal_deg"]
        self.pricing_type = pricing_parameters["pricing_type"]

        self.h = self.T/self.n # or delta T
        self.neural_network = neural_network

    # Function that returns the payoff in point in time of the option, based on the exercise price and zero vaues
    def get_payoff(self, underlying_prices):

        if self.payoff_type == 'a_average':
            price = np.dot(underlying_prices, self.weights).T
        elif self.payoff_type == 'g_average':
            price = np.power(underlying_prices.prod(axis=-1), 1.0/self.dim).T
        elif self.payoff_type == 'max':
            price = np.amax(underlying_prices, 2).T
        elif self.payoff_type == 'min':
            price = np.amin(underlying_prices, 2).T
        else:
            print("This is not a possible payoff type: ", self.payoff_type)
            exit()

        if self.option_type == "call":
            return np.maximum(price - self.K, np.zeros((self.exercise_n + 1, self.num_paths)))
        elif self.option_type == "put":
            return np.maximum(self.K - price, np.zeros((self.exercise_n + 1, self.num_paths)))


    def get_option_value(self, underlying_prices):
        # Returns the value of the Bermudan option

        payoff = self.get_payoff(underlying_prices)
        discount = np.exp(self.h*(-self.r))

        values = np.zeros((self.n + 1, self.num_paths))
        values[-1, :] = payoff[-1, :]


        tn = np.array(range(self.exercise_n+1))*int(self.n/self.exercise_n)
        exercise_times = tn[1:-1] # remove start and end time

        #print(tn)
        j = self.exercise_n-1
        for i in range(self.n-1, 0 ,-1):
            #print(i)

            if i in exercise_times:          # since we do Bermudan options

                # Prediciting the future value of the path, given a regression line.
                # This is the same as in the literature.
                # Fit   GBM_prices[i, :]   to   values[i + 1, :]*self.discount

                if self.pricing_type == "neural_network":
                    # Divide by K since we use the moneyness
                    X = np.concatenate((underlying_prices[:, j, :]/self.K, payoff[j, :].reshape(-1, 1)/self.K), axis=1)
                    print('start training at exercise time i=', i)
                    start_training = time.time()
                    self.neural_network.train(X, values[i + 1, :]*discount/self.K, i)
                    done_training = time.time()
                    print("Time it took to train: ", round(done_training - start_training, 1))
                    
                    print('start predicting at exercise time i=', i)
                    start_predicting = time.time()
                    future_value = self.neural_network.predict(X).squeeze()*self.K
                    done_predicting = time.time()
                    print("Time it took to predict: ", round(done_predicting - start_predicting, 1))
                    values[i, :] = np.where(payoff[j, :] > future_value, payoff[j, :], values[i + 1, :]*discount)

                elif self.pricing_type == "synthetic":
                    print('start training at exercise time i=', i)
                    if self.payoff_type == "max":
                        X = np.amax(underlying_prices[:, j, :], 1).T
                    elif self.payoff_type == "min":
                        X = np.amin(underlying_prices[:, j, :], 1).T
                    elif self.payoff_type == "g_average":
                        X = np.power(underlying_prices[:, j, :].prod(axis=-1), 1.0/self.dim).T
                    elif self.payoff_type == "a_average":
                        X = np.dot(underlying_prices[:, j, :], np.ones(size=(self.dim))*1/self.dim).T
                    start_training = time.time()
                    coeff= np.polyfit(X, values[i + 1, :] * discount, self.polynominal_deg)
                    done_training = time.time()
                    print("Time it took to train: ", round(done_training - start_training, 1))
                    print('start predicting at exercise time i=', i)
                    start_predicting = time.time()
                    future_value = np.polyval(coeff, X)
                    done_predicting = time.time()
                    print("Time it took to predict: ", round(done_predicting - start_predicting, 1))
                    values[i, :] = np.where(payoff[j, :] > future_value, payoff[j, :], values[i + 1, :] * discount)

                j -= 1
            else:
              values[i, :] = values[i + 1, :] * discount

        final_values = values[1,:] * discount
        return np.sum(final_values / self.num_paths)
