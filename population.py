import numpy as np
from utils import Utils

class Population:
    def __init__(self, period, mean, numOfTowns):
        self.period = period
        self.meanPop = mean
        self.wave = []

        self.computeBasePopWave()
        self.populationGame()

    def computeBasePopWave(self):
        for x in range(-3, self.period+3):
            gauss = np.random.normal(0,1,1)[0]
            self.wave.append(self.meanPop*((1 + ((1/3)*np.cos(0.0172*x) + (1/3)*np.cos(0.0172*x*2 - 0.6) ))))   

    def populationGame(self):
        for i in range(-3, self.period+3):
            gauss = np.random.normal(0,1,1)[0]
            #if i >= 1 and i <= self.period:
            self.wave[i] = self.wave[i] + (self.meanPop/10) * gauss
        popToReturn = Utils.smooth(self.wave, 6)[3:self.period+3]

        cov_matrix = np.array(  [[0.1, 0, 0],
                                [0, 0.1, 0],
                                [0, 0, 0.1]])
        mean_vector = np.array([popToReturn[0]/2, popToReturn[0]/4, popToReturn[0]/4])

        num_samples = 100
        data = np.random.multivariate_normal(mean_vector, cov_matrix, num_samples)

        # Define the autoregressive model parameters
        phi = np.array([[0.7, 0.2, 0.1],
                        [0.2, 0.7, 0.1],
                        [0.35, 0.15, 0.5]])
        predicted_data = np.zeros_like(data)

        predicted_data[0] = data[0]
        for i in range(1, num_samples):
            predicted_data[i] = Utils.computeAR1(predicted_data[i], phi, cov_matrix)   #np.dot(phi, predicted_data[i-1]) + np.random.multivariate_normal(mean_vector, cov_matrix)/1000

        # Print the predicted data
        print('Predicted data:', predicted_data)

        return popToReturn


if __name__ == "__main__":
    sym = Population(360*100, 1000, 3)
    # sym.getRainAmountProb()
