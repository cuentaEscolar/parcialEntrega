import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from pandas import DataFrame
from statsmodels.tsa.seasonal import seasonal_decompose, seasonal_mean
import seaborn as sns
import numpy as np
from scipy.fft import fft, fftfreq, ifft
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf  
from Utils import Utils
def fork(a):
    return a.copy()
class Grapher:
    @staticmethod
    def MAP(mapper, series):
        return [
            np.array([mapper[_][0] for _ in series]),
            np.array([mapper[_][1] for _ in series])
        ]
    @staticmethod
    def origenDestDist( mapper , fullData, name):
        countS, countD =  fork([0]*len(mapper))
        sns.jointplot(x="Orig")
        st.pyplot()
        plt.clf()
    @staticmethod
    def plotPos( mapper, posData, name):
        origenPos =  Grapher.MAP(mapper,posData)    
        plt.cla()
        plt.scatter(origenPos[-2], origenPos[1])
        plt.title(name)
        st.pyplot()
        plt.clf()

    @staticmethod
    def plotHist2d(mapper, series):

        plt.cla()
        seriesPos = Grapher.MAP(mapper, series)  
        plt.hist2d(seriesPos[0], seriesPos[1], bins=50)
        plt.savefig("originHist.png")
        plt.close()

    @staticmethod
    def plotArrowMap(mapper, first, second, name):
        plt.cla() 
        plt.figure(figsize=(10,10))
        i = 0
        for _ in zip(first, second):
            if i == 1000: break
            (_1, _2) = _
            x, y = mapper[_1]
            x2, y2 = mapper[_2]
            dx, dy = x2 - x, y2 - y 
            plt.arrow(
                x=x, 
                y=y,
                dx=dx,
                dy=dy,
                width=0.0001,
                alpha=0.01,
                edgecolor=(0.0,0.0,0.0,0.00001),
                antialiased=True,
                head_length=0,
                head_width=0
            )

        plt.title(name)
        plt.savefig(name)

    @staticmethod
    def travelPerYear(mapper, bigAssData):
        pass 
    @staticmethod 
    def genderTravelTimeCorrel(bigAssData):
        pass
    @staticmethod
    def correlEdadTiempo(Data, name):
        plt.figure(figsize=(18,16))
        print(Data.columns)
        return Grapher.correl(Data, name, "Edad", "Duracion",xlim=(18, 80), 
                ylim=(0,35))
    @staticmethod
    def correlSemana(Data,name):
        fig, axs = plt.subplots(nrows=2, ncols=7, figsize=(16,9))
        lessThan50 = [0]*511

        for i in range(7):
            sns.violinplot( 
                ax=axs[0,i],
                data=Data,
                y="Duracion",
                split=True,
                hue="Genero",
                inner="quart"
            )
            sns.boxplot(
                ax = axs[1,i],
                data = Data[ Data["Semana"] == i  ]["Duracion"]
            )
        plt.show()
        plt.savefig(name)

         

    @staticmethod
    def correl(Data, name, x, y, xlim=None, ylim=None):
        graph = sns.jointplot(
            data=Data,
            x=x,
            y=y,
            xlim=xlim,
            ylim=ylim,
            cmap="rocket",
            kind="kde",
            fill=True,
            thresh=0
            # ylim=(0, 12),
        )
        graph.plot_marginals(sns.histplot, color="#03051A", alpha=1, bins=32)
        st.pyplot()
        plt.clf()

    @staticmethod
    def correlComp(Data, name,x, y, cmp=None):
        ax = sns.kdeplot(
            data=Data,
            x=x,
            y=y,
            hue=cmp,
            alpha=0.5,
            levels=10,
            fill=False,
            xlim=(18, 80), 
            linewidth=0
        )
        
        graph = ax.get_figure()
        graph.savefig(name)
        plt.clf()
    
    @staticmethod
    def correlCompCapture(Data, name):
        return lambda x,y : lambda cmp : Grapher.correlComp(Data, name, x, y , cmp=cmp)

    @staticmethod
    def correlCompEdadDuracion(Data, name):
        return Grapher.correlCompCapture(Data,name)("Edad","Duracion")

    @staticmethod 
    def plotHist1d(Data, by_, column):
        Data.hist(by=by_, column=column)
        plt.savefig("histTest.png")

    @staticmethod
    def tiempoVsEstacion(Data: DataFrame, name):
        Data.dropna(inplace=True)
        fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(32,32))
        sns.kdeplot(
            ax = axs[0,0],
            data=Data[Data["Genero"]== "M"],
            x="Origen_Id",
            y="Duracion",
            cmap="mako", fill=True,
            thresh=0, levels=32,
            legend=True
        )

        sns.kdeplot(
            ax = axs[0,1],
            data=Data[Data["Genero"] == "M"],
            x="Destino_Id",
            y="Duracion",
            cmap="mako", fill=True,
            thresh=0, levels=32,
            legend=True
        )

        sns.kdeplot(
            ax = axs[1,0],
            data=Data[Data["Genero"]== "F"],
            x="Origen_Id",
            y="Duracion",
            cmap="mako", fill=True,
            thresh=0, levels=32,
            legend=True
        )

        sns.kdeplot(
            ax = axs[1,1],
            data=Data[Data["Genero"] == "F"],
            x="Destino_Id",
            y="Duracion",
            cmap="mako", fill=True,
            thresh=0, levels=32,
            legend=True
        )
        st.pyplot()
        plt.clf()

    
    @staticmethod
    def plotAvgTime(Data, name):
        fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(10,4))
        M = Data[Data["Genero"]=="M"]["Duracion"]
        F = Data[Data["Genero"]=="F"]["Duracion"]

        axs[0].violinplot(
            M, 
            side="low",
            showmedians=True,
            showextrema=False
        )
        axs[0].violinplot(
            F,
            side="high",
            showmedians=True,
            showextrema=False
        )
        #axs[0].violinplot(
        #    Data[Data["Genero"]=="H"]["Duracion"],
        #    side="high"
        #)
        axs[0].set_title("Distribucion")
        axs[1].boxplot(
            Data["Duracion"]
        )
        axs[1].set_title("Cuartiles")
        for ax in axs:
            ax.set_ylabel("Tiempo de viaje")
        st.pyplot()
        plt.savefig(name)
        plt.clf()

    @staticmethod 
    def groupStations(raw,name):
        Data = raw.copy()
        for column in Data.columns: continue
        Data.drop(inplace=True, axis=1)
        graph = sns.pairplot(Data, kind="scatter", hue="Genero")
        graph.savefig(name)
        plt.clf()

    @staticmethod
    def timeSeries(Data, COL ):
        for GROUP in ["DateInicio"]:
            mean, sum = Utils.getAvg(Data, COL,by_=GROUP), Utils.getSum(Data, COL, by_=GROUP)
            count = Utils.getCount(Data, COL, by_=GROUP)
            Grapher.compareDecomposedToFT(mean, f"{COL} Promedio by {GROUP }")
            Grapher.compareDecomposedToFT(sum, f"{COL} Acumulado by {GROUP }")
            Grapher.compareDecomposedToFT(count, f"Cuenta de {COL}  by {GROUP }")
            st.pyplot()


    @staticmethod
    def compareDecomposedToFT(series, name):
        fig, axs = plt.subplots(nrows=2, ncols=4, figsize=(32,18))
        rollmean = series.rolling(7).median()
        dec = seasonal_decompose(series)
        trend, seasonal, residual = dec.trend, dec.seasonal, dec.resid
        # [rows, cols ]
        axs[0,0].plot(series, label="original")
        axs[0,0].plot(rollmean, label="rollAvg") 
        axs[0,0].set_ylabel(f"{name}, por dia, en escala Logaritmica")
        
        axs[0,1].plot(trend, label="Tendencia")
        axs[0,0].set_ylabel(f"Tendencia de la {name}, por dia, en escala Logaritmica")
        axs[0,2].plot(seasonal)
        axs[0,3].plot(residual)
        for i in range(4):
            axs[0,i].set_xlabel("Mes")

        axs[1,0].plot(fft(series))
        axs[1,1].plot(fft(trend))
        axs[1,2].plot(fft(seasonal))
        axs[1,3].plot(fft(residual))

        for i in range(4):
            axs[1,i].set_xlabel("Mes")

        plt.title("Serie y Transformada de la Serie")

        plt.savefig(name + ".png")
        plt.clf()
    @staticmethod
    def arimaComp(ts, name):
        ts.dropna(inplace=True)
        print(ts)
        ts_log = np.log(ts)

        ts_log_diff = ts_log - ts_log.shift()
        ts_log_diff.dropna(inplace=True)
        modelo_ARIMA = ARIMA(ts_log_diff, order = (2, 1, 2), freq = None)
        resulta_ARIMA = modelo_ARIMA.fit()
        RSS_ARIMA = sum((resulta_ARIMA.fittedvalues - ts_log_diff)**2)

        predic_ARIMA_diff = pd.Series(resulta_ARIMA.fittedvalues)

        predic_ARIMA_diff_cumsum =  predic_ARIMA_diff.cumsum()

        predic_ARIMA_log = pd.Series(ts_log.loc[ts_log.index[0]], index = ts_log.index)
        predic_ARIMA_log = predic_ARIMA_log.add(predic_ARIMA_diff_cumsum, fill_value = 0)

        predic_ARIMA = np.exp(predic_ARIMA_log).astype('int')

        fig, axs = plt.subplots(nrows=1, ncols=1)

        axs.plot(ts, label="Original")
        axs.plot(predic_ARIMA, label="Arima")
        plt.legend(loc="best")
        plt.savefig(name +".png")
        return 
        trend_log, seasonal_log, residual_log, =  (
            np.log(trend), np.log(seasonal), np.log(residual)
        )
        series_log.dropna(inplace=True)
        trend_log.dropna(inplace=True)
        seasonal_log.dropna(inplace=True)
        residual_log.dropna(inplace=True)
        shifted = [ x - x.shift(freq="infer") for x in 
            (series_log, trend_log, seasonal_log, residual_log)
        ]
        modelos_ARIMA = [ ARIMA(x, order=(2,1,2), freq=None ) for x in shifted]
        resulta_ARIMAs = [ modelo.fit() for modelo in modelos_ARIMA]
        RSS_ARIMAs = [ sum((resulta_ARIMAs[i].fittedvalues - shifted[i])**2) 
            for i in range(len(shifted))]
        predic_ARIMA_diffs = [pd.Series(resulta_ARIMA.fittedvalues) for 
                              resulta_ARIMA in resulta_ARIMAs]
        predic_ARIMA_diffs_cumsum = [predic_ARIMA_diff.cumsum() for 
                                     predic_ARIMA_diff in predic_ARIMA_diffs]
        predic_ARIMA_logs = [pd.Series(x.loc[series.index[0]], index = x.index) for x in 
                (series, trend, seasonal, residual)]
        predic_ARIMA_logs = [predic_ARIMA_logs[i].add(predic_ARIMA_diffs_cumsum[i], fill_value = 0) for i in range(len(shifted))]
        predic_ARIMAs = [ np.exp(x) for x in predic_ARIMA_logs]

        axs = plt.plot(series, label="Original")
        axs = plt.plot(predic_ARIMAs[1], label="Arima")
        plt.legend(loc="best")
        plt.title(f"Predicciones de {name}, Metodo Arima")
        #axs[0,1] = 
        #axs[0,1] = 
 
        fig, axs = plt.subplots(nrows=1, ncols=1)
        axs = plt.plot(series.copy(), label="Original")
        axs = plt.plot(predic_ARIMAs[1] + predic_ARIMAs[2], label="Arima")
        plt.legend(loc="best")
        plt.title(f"Predicciones de {name}, Metodo Arima + Decomposicion")
        plt.savefig(name+"EstimacionDescomp.png")
