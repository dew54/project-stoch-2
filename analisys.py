from sym import Sym
import pandas as pd
from random import randint
import random
import numpy as np
import math
from utils import Utils
import matplotlib.pyplot as plt
class Analisys:
    def __init__(self):

        eqThreshold = 6
        eqA, eqB = 1.8, 0.4
        hRainThreshold = 800
        visibilityThreshold = 700
        self.period = 360 * 50
        self.forecastingPeriod = 360*10
        numSamples = 15

        self.sym = Sym(self.period)
        data = self.sym.getDf()
        stats = self.sym.getDfStats()
        
        self.dfAn = pd.DataFrame(columns=['day', 'month', 'season', 'year'])

        days = [randint(1,self.forecastingPeriod) for _ in range(numSamples)]
        self.dfAn['day']= days
        self.dfAn['year'] = ([math.ceil(t/30/12) for t in days])
        self.dfAn['month'] = ([math.floor((t)/30)%12 +1 for t in days])
        self.dfAn['season'] = (([math.floor((t)/90)%4 +1 for t in days]))
        floodReturnPeriod = [stats.floodReturnPeriod[month-1] for month in self.dfAn.month.values]

        print("FLOOD RETURN PERIOD IS:", floodReturnPeriod)

        # print([1 - (1-1/floodReturnPeriod[self.dfAn.month.values[i]])**self.dfAn.year.values[i] for i in range(len(self.dfAn.year.values))])
        
        # self.dfAn['FloodRisk'] = [1 - (1-1/floodReturnPeriod[i])** self.dfAn.year[i] for i in self.dfAn.year.values ]
        test = [self.dfAn.loc[self.dfAn['year'] == i, 'month'] for i in self.dfAn.year.values][1]

        self.dfAn['FloodRisk'] = [1 - (1-1/floodReturnPeriod[self.dfAn.loc[self.dfAn['year'] == i, 'month'][0]])** self.dfAn.year[i] for i in self.dfAn.year.values ]

        numberOfEQ = 10**(eqA - eqB*eqThreshold)
        eqFrequency = numberOfEQ / self.period

        self.dfAn['EqRisk'] = [1- math.exp(-1*eqFrequency*self.dfAn.year.values[i]) for i in self.dfAn.year.values]
        print(data)

        print("##########################################################################")

        heavyRainRisk = ([Utils.chebyshev(hRainThreshold, stats.meanRain[i], math.sqrt(stats.varRain[i])) for i in range(12)])
        self.dfAn['HeavyRainRisk'] = [heavyRainRisk[i-1] for i in self.dfAn.month.values]
        
        poorVisibilityRisk = ([Utils.chebyshev(visibilityThreshold, stats.meanRad[i], math.sqrt(stats.varRad[i])) for i in range(12)])
        self.dfAn['poorVisibilityRisk'] = [poorVisibilityRisk[i-1] for i in self.dfAn.month.values]

        self.dfAn['belowZeroRisk'] = [stats.belowZeroMean[i-1] for i in self.dfAn.month.values]

        self.dfAn['expectedPopulation'] = ([np.mean([data.population.loc[data['day'] == math.ceil(i%360)+1]]) for i in self.dfAn['day'].values])
    
        self.dfAn.sort_values(by=['day'])

        # self.dfAn['HRainRisk'] = data.rainAmount.loc[data['day'] == self.dfAn.day[0] ] 
        print(self.dfAn)
        normalizedDF = self.dfAn.iloc[:,4:].apply(lambda x: (abs(x))/sum(x), axis=0)
        print(normalizedDF)
        normalizedDF.plot()

        plt.show()
 


if __name__ == "__main__":
    analisys = Analisys()