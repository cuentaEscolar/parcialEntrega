import streamlit as st
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt 
import seaborn as sns
from scipy import stats
from datetime import datetime
from Grapher import Grapher
from Utils import Utils  
unaBici = Utils.cleanUp(pd.read_csv(f"test.csv", encoding="latin1")) #test has data for multiple months 
mapper = Utils.loadMapVals("./coords.csv")

st.title("Entrega Parcial")
st.write("Tiempo Vs Estacion, por genero")
st.write("mm")
st.write("ff")
Grapher.tiempoVsEstacion( unaBici, "TiempoVSEstacion" )
st.write("Serie de Tiempo de Duracion Total y Promedio")
Grapher.timeSeries(unaBici, "Duracion")
st.write("Tiempos Promedio")
Grapher.plotAvgTime(unaBici, "AvgTimeTest.png")
st.write("Serie de Tiempo de Costo del viaje Total y Promedio ")
Grapher.timeSeries(unaBici, "Costo")
st.write("Edad Vs Tiempo de Viaje")
Grapher.correlEdadTiempo(unaBici, "correlEdadTiempo.png")