from weather import Weather
from population import Population
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from utils import Utils
import math


class Sym:
    def __init__(self, period):
        mean = 2000
        numOfTowns = 1  #Useless for the moment
        self.Ia = 11
        self.S = 12
        self.qThreshold = 500
        self.period = period
        self.weather = Weather(period)
        self.population = Population(period, mean, numOfTowns)
        self.utils = Utils()
        self.dailyRain, t_min, t_max, radiaz = self.weather.weatherGame()
        popWave = self.population.populationGame()

        self.df = pd.DataFrame(columns=['day', 'month', 'season', 'year', 'population', 'rainAmount', 'rainOff',  't_min', 't_max', 'radiation'])
        self.dfStats = pd.DataFrame(columns=['meanRain', 'varRain', 'meanTmin', 'varTmin', 'meanTmax', 'varTmax', 'meanRad', 'varRad', 'meanPop', 'varPop'])
        #self.seasonalStats = pd.DataFrame(columns=['meanRain', 'varRain', 'meanTmin', 'varTmin', 'meanTmax', 'varTmax', 'meanRad', 'varRad'])


        self.df['day'] = ([math.ceil(t%360)+1 for t in range(period)])
        self.df['year'] = ([math.ceil(t/30/12) for t in range(1, period+1)])
        self.df['month'] = ([math.floor((t)/30)%12 +1 for t in range(period)])
        self.df['season'] = (([math.floor((t)/90)%4 +1 for t in range(period)]))
        self.df['population'] = popWave
        # self.df['LocA'] = 
        # self.df['LocB'] = 
        # self.df['LocC'] = 
        
        self.df['rainAmount'] = self.dailyRain
        self.df['t_min'] = t_min
        self.df['t_max'] = t_max
        self.df['radiation'] = radiaz
        self.df['rainOff'] = ([(((p - self.Ia)**2)/(p - self.Ia + self.S) if p > 0 else 0) for p in self.dailyRain ])#, 1)

        self.dfStats.meanRain = ([np.mean([self.df.rainAmount.loc[self.df['month'] == i]]) for i in range(1,13)])
        self.dfStats.varRain = ([np.var([self.df.rainAmount.loc[self.df['month'] == i]]) for i in range(1,13)])
        self.dfStats.meanTmin = ([np.mean([self.df.t_min.loc[self.df['month'] == i]]) for i in range(1,13)])
        self.dfStats.varTmin = ([np.var([self.df.t_min.loc[self.df['month'] == i]]) for i in range(1,13)])
        self.dfStats.meanTmax = ([np.mean([self.df.t_max.loc[self.df['month'] == i]]) for i in range(1,13)])
        self.dfStats.varTmax = ([np.var([self.df.t_max.loc[self.df['month'] == i]]) for i in range(1,13)])
        self.dfStats.meanRad = ([np.mean([self.df.radiation.loc[self.df['month'] == i]]) for i in range(1,13)])
        self.dfStats.varRad = ([np.var([self.df.radiation.loc[self.df['month'] == i]]) for i in range(1,13)])
        self.dfStats.meanPop = ([np.mean([self.df.population.loc[self.df['month'] == i]]) for i in range(1,13)])
        self.dfStats.varPop = ([np.var([self.df.population.loc[self.df['month'] == i]]) for i in range(1,13)])

        self.dfStats.insert(10, 'belowZeroMean', [self.meanBelowValue(0, (self.df.t_min.loc[self.df['month'] == i]).values) for i in range(1, 13)] )
        self.dfStats.insert(11, 'belowZeroVar', [self.varBelowValue(0, (self.df.t_min.loc[self.df['month'] == i]).values) for i in range(1, 13)] )

        self.dfStats.insert(12, 'poorVisibilityMean', [self.meanBelowValue(500, (self.df.radiation.loc[self.df['month'] == i]).values) for i in range(1, 13)] )
        self.dfStats.insert(13, 'poorVisibilityVar', [self.varBelowValue(500, (self.df.radiation.loc[self.df['month'] == i]).values) for i in range(1, 13)] )

        # self.dfStats.insert(10, 'belowZeroVar', [self.meanBelowValue(0, (self.df.t_min.loc[self.df['month'] == i]).values) for i in range(1, 13)] )

        self.dfStats.insert(14, 'floodReturnPeriod', [self.getFloodReturnPeriod(self.df.rainOff.loc[self.df['month'] == i], self.qThreshold) for i in range(1,13)])

        fig1, axs1 = plt.subplots(3, 2)

        X = range(0, period)

        axs1[0][0].plot(X, self.dailyRain )
        axs1[0][0].set_title('Rainy days')

        axs1[1][0].plot(X, t_min, t_max)
        axs1[1][0].set_title('t max/min')

        axs1[0][1].plot(X,  popWave)
        axs1[0][1].set_title('pop')

        axs1[1][1].plot(X,  radiaz )
        axs1[1][1].set_title('radiation')

        axs1[2][0].plot(X,  self.df['rainOff'] )
        axs1[2][0].set_title('rainOff')

        # plt.show()
        
    def getDf(self):
        return self.df
    def getDfStats(self):
        return self.dfStats


    def extractDescriptors(self):
        self.returnPeriod = self.getFloodReturnPeriod(self.qThreshold)

    # def splitInPeriods(self, days, nPeriods):
    #     periods = []
    #     daysInAPeriod = len(days)
    #     for n in range(nPeriods):


                
        
    

    def getFloodReturnPeriod(self, data, threshold):
        events = []
        periods = []
        [events.append(i) for i in range(len(data.values)) if data.values[i] > threshold]
        print("events are", events)
        if len(events) < 1:
            return 100
        else:
            return (self.period/len(events))/360


    def meanBelowValue(self, value, data):
        count = 0
        count = sum([1 if data[i] < value else 0 for i in range(len(data))])
        return np.mean([1 if data[i] < value else 0 for i in range(len(data))])
    
    def varBelowValue(self, value, data):
        count = 0
        count = sum([1 if data[i] < value else 0 for i in range(len(data))])
        return np.var([1 if data[i] < value else 0 for i in range(len(data))])


#(count/(len(data)/(self.period/360)))
    # def getRainAmountProb(self):
    #     distribution = [0, 0, 0, 0]
    #     variance = [0, 0, 0, 0]

        
    #     for amount in self.dailyRain:
    #         if 0<amount and amount < 250:
    #             distribution[0] += 1
    #         if 250<amount and amount < 500:
    #             distribution[1] += 1
    #         if 500<amount and amount < 750:
    #             distribution[2] += 1
    #         if 750<amount and amount < 1000:
    #             distribution[3] += 1
    #     distribution = Utils.normalizeProbs(distribution)
    #     variance[0] = np.var([rain for rain in self.dailyRain if rain < 250])
    #     variance[1] = np.var([rain for rain in self.dailyRain if rain > 250 and rain <500])
    #     variance[2] = np.var([rain for rain in self.dailyRain if rain < 500 and rain < 750])
    #     variance[3] = np.var([rain for rain in self.dailyRain if rain > 750])
        
    #     plt.plot(range(4), distribution )
    #     print("var 1 = ", variance[0])
    #     print("var 2 = ", variance[1])
    #     print("var 3 = ", variance[2])
    #     print("var 4 = ", variance[3])
    #     # plt.show()
        
    def printHist(self, data, nBins = 20):
        H, edges = np.histogram(data, bins=nBins)
        plt.figure(figsize=(10, 4))
        ax = plt.subplot(111)
        ax.bar(edges[:-1], H / float(sum(H)), width=edges[1] - edges[0])
        ax.set_xlabel("Data")
        ax.set_ylabel("Probability")
        ax.minorticks_on()
        plt.show()


if __name__ == "__main__":
    sym = Sym(360*100)
    # sym.getRainAmountProb()
