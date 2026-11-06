import time
import numpy as np

from stochastic_processes.gbm import GBM
from stochastic_processes.heston import Heston

from pricing_methods.neural_network import NeuralNetwork
from pricing_methods.pricing import PricingModel

def get_GBM_correlation(s_corr, dim):
    # Returns a matrix with correlation coefficients between S
    # Assumes that the correlation between all S is equal s_corr
    corr = np.ones((dim,dim))*s_corr
    np.fill_diagonal(corr, 1)
    return corr

def get_heston_correlation(s_corr, vol_corr, dim):
    # Returns a matrix with correlation coefficients between S and vol
    # Assumes that the correlation between all S is equal s_corr
    # Assumes that the correlation between all S are equal s_corr, and all volatilities are vol_corr
    corr = np.ones((dim,dim))*s_corr
    np.fill_diagonal(corr, 1)
    vol_corrs = np.ones(dim)*vol_corr
    vol_matrix = np.diag(vol_corrs)
    heston_corr = np.vstack((np.hstack((corr, vol_matrix)), np.hstack((vol_matrix, np.eye(dim)))))
    return heston_corr

def define_parameters():
    parameters = {}

    parameters["option_parameters"] = {}
    #parameters["option_parameters"]["dim"] = 50  # 50 very slow
    parameters["option_parameters"]["dim"] = 50
    parameters["option_parameters"]["S0"] = 100
    parameters["option_parameters"]["K"] = 100
    parameters["option_parameters"]["r"] = 0.05
    parameters["option_parameters"]["div"] = 0.1
    parameters["option_parameters"]["volatility"] = np.ones((parameters["option_parameters"]["dim"]))*0.2
    parameters["option_parameters"]["T"] = 3 # 
    parameters["option_parameters"]["s_corr"] = 0   # correlation between S_i and S_j
    parameters["option_parameters"]["correlation_matrix"] = get_GBM_correlation(parameters["option_parameters"]["s_corr"], parameters["option_parameters"]["dim"])
    parameters["option_parameters"]["payoff_type"] = "max" # max, min, a_average, g_average #min-call has zero price
    parameters["option_parameters"]["option_type"] = "call"

    parameters["simulation_parameters"] = {}
    parameters["simulation_parameters"]["n"] = 300*parameters["option_parameters"]["T"]
    parameters["simulation_parameters"]["num_paths"] = 300000  # 500000  originally /Ying 300000 was claimed in project thesis, but syn. gave lower values
    parameters["simulation_parameters"]["exercise_n"] = 9

    parameters["heston_parameters"] = {}
    parameters["heston_parameters"]["theta"] = np.ones((parameters["option_parameters"]["dim"]))*0.04
    parameters["heston_parameters"]["xi"] = np.ones((parameters["option_parameters"]["dim"]))*0.9
    parameters["heston_parameters"]["kappa"] = np.ones((parameters["option_parameters"]["dim"]))*2
    parameters["heston_parameters"]["correlation_matrix"] = get_heston_correlation(0, -0.7, parameters["option_parameters"]["dim"])

    parameters["neural_network_parameters"] = {}
    parameters["neural_network_parameters"]["num_neurons"] = 50
    parameters["neural_network_parameters"]["input_dim"] = parameters["option_parameters"]["dim"] + 1 # +1 since we also use the payoff
    parameters["neural_network_parameters"]["epochs1"] = 1000
    parameters["neural_network_parameters"]["epochs2"] = 200
    parameters["neural_network_parameters"]["batch_size"] = 10000
    parameters["neural_network_parameters"]["activation_function"] = "elu"

    parameters["pricing_parameters"] = {}
    parameters["pricing_parameters"]["pricing_type"] = "synthetic" # neural_network or synthetic
    parameters["pricing_parameters"]["polynominal_deg"] = 5

    parameters["experiments"] = {}
    parameters["experiments"]["num_experiments"] = 10 # 15 in the paper
    parameters["experiments"]["heston"] = True

    return parameters

def main():
    print("_"*50)
    print("Starts pricing the option")
    parameters = define_parameters()
    heston = parameters["experiments"]["heston"]
    num_expr = parameters["experiments"]["num_experiments"]
    price_array = np.zeros(num_expr)

    for i in range(num_expr):
        print("_"*50)
        print("Start experiment ", i)
        print("Simulates the underlying...")
        start = time.time()
        if heston:
            print('We are under Heston model')
            heston = Heston(parameters["option_parameters"],
                            parameters["simulation_parameters"],
                            parameters["heston_parameters"])
            underlying_prices = heston.simulate_underlying_prices()
        else:
            print('We are under GBM model')
            gbm = GBM(parameters["option_parameters"], parameters["simulation_parameters"])
            underlying_prices = gbm.simulate_underlying_prices()
        done_simulating_underlying = time.time()
        print("Time it took to simulate the underlying: ", round(done_simulating_underlying - start, 1))

        print("Pricing the option...")
        start_pricing = time.time()

        neural_network = None
        if parameters["pricing_parameters"]["pricing_type"] == "neural_network":
            neural_network = NeuralNetwork(parameters["neural_network_parameters"])

        pricing_model = PricingModel(parameters["option_parameters"],
                                    parameters["simulation_parameters"],
                                    parameters["pricing_parameters"],
                                    neural_network)
        price = pricing_model.get_option_value(underlying_prices)
        done_pricing = time.time()
        print("Time it took to price the option: ", round(done_pricing - start_pricing, 1))
        print("Total time: ", done_pricing - start)

        print("Price of {} {} option with {} dimentions and {} regression: {}".format(parameters["option_parameters"]["payoff_type"],
                                                                    parameters["option_parameters"]["option_type"],
                                                                    parameters["option_parameters"]["dim"],
                                                                    parameters["pricing_parameters"]["pricing_type"],
                                                                    round(price, 8)))
        price_array[i]= price
        print('test ',  price_array[i])
    print('average price from 10 experiments is', price_array.mean())
    print('standard deviation from these 10 experiments is', price_array.std())
    array_message = "Price of {} {} option with {} dimentions and {} regression".format(parameters["option_parameters"]["payoff_type"],
                                                                parameters["option_parameters"]["option_type"],
                                                                parameters["option_parameters"]["dim"],
                                                                parameters["pricing_parameters"]["pricing_type"])
    np.savez('price_array.npz', array=price_array, message=np.array(array_message))


if __name__ == "__main__":
    main()
