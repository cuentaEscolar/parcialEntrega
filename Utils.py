from datetime import datetime
import pandas as pd
from pandas.core.api import to_datetime
from scipy.fft import fft, ifft
class Utils:


    @staticmethod 
    def loadMapVals(fileText):
        file = open(fileText,"r", encoding="ISO-8859-1")
        file.readline()
        mapper = [(None,None)]*391
        while (line:=file.readline()):
            i, latitude, longitude = line.split(",") 
            i = int(i)
            longitude = longitude[:-2]
            mapper[i] = (float(latitude), float(longitude))

        return mapper

    @staticmethod 
    def MAP(mapper, series):
        return [
            np.array([mapper[_][0] for _ in series]),
            np.array([mapper[_][1] for _ in series])
        ]

    @staticmethod
    def getUnaryFunctions():
        return (
                {
                    "Semana": ( Utils.getWeek ), 
                    "Mes": ( Utils.getMonth), 
                    "Edad" : (    lambda x: 2024 -  x),
                    "SegundosO": (Utils.getSeconds),
                    "SegundosD": (Utils.getSeconds),
                },
                {
                    "Semana" : "Inicio_del_viaje", 
                    "Mes" : "Inicio_del_viaje", 
                    "Edad": "AÃ±o_de_nacimiento",
                    "SegundosO": "Inicio_del_viaje",
                    "SegundosD": "Fin_del_viaje"
                },
        )
    @staticmethod
    def ymd(stringy):
        ymd = stringy.split(" ")[0]
        return [int(x) for x in ymd.split("-")]
    @staticmethod
    def hms(stringy):
        hms = stringy.split(" ")[1]
        return [int(x) for x in hms.split(":")]
    @staticmethod
    def getWeek(stringy):
        #dateFormat = "%Y-%m-%d %H:%M:%S"
        y, m, d = Utils.ymd(stringy)
        return datetime(y,m,d).isoweekday()%7
    @staticmethod
    def getSeconds(stringy):
        return Utils.hms(stringy)[2]
    @staticmethod
    def getMinutes(stringy):
        return Utils.hms(stringy)[2]
        
    @staticmethod
    def getYear(stringy):
        return Utils.ymd(stringy)[0]
    @staticmethod
    def getMonth(stringy):
        return Utils.ymd(stringy)[1]

    @staticmethod
    def calcCost(duracionEnSegundos, weekday):
        if weekday==6:
            return  478 + Utils.calcPar(duracionEnSegundos, 45)
        return 478 + Utils.calcPar(duracionEnSegundos, 30)
    @staticmethod
    def calcPar(duracionS, freeM)-> int: 
        if duracionS <= freeM*60: return 0 
        if duracionS <= 60*60: return 28 
        res = 28 
        duracionS -= 3600 
        res += 39*duracionS//(60*30)
        if duracionS % (60*30): res+= 39
        return min(res, 1052)
                
    @staticmethod
    def createColumns(Data: pd.DataFrame):

        dateFormat = "%Y-%m-%d %H:%M:%S"
        start = datetime(1970,1,1)
        Data["DateInicio"] = pd.to_datetime(Data["Inicio_del_viaje"])
        Data["DateInicio"] = Data["DateInicio"].apply(func=(lambda x: x.date()))
        Data["DateFin"] = pd.to_datetime(Data["Inicio_del_viaje"])
        Data["DateFin"] = Data["DateFin"].apply(func=(lambda x: x.date()))

        TRANS = lambda x: pd.to_datetime(x, format=dateFormat)
        Data["InicioS"] = Data["Inicio_del_viaje"].apply(func=TRANS)
        Data["InicioS"] = Data["InicioS"].apply(func=(lambda x: (x-start).total_seconds()))

        Data["FinS"] = Data["Fin_del_viaje"].apply(func=TRANS)
        Data["FinS"] = Data["FinS"].apply(func=(lambda x: (x-start).total_seconds()))

        Data["InicioH"] = Data["InicioS"].apply(func=(lambda x: x//3600))
        Data["FinH"] = Data["FinS"].apply(func=(lambda x: x//3600))


        unaryF, unaryC = Utils.getUnaryFunctions()
        for param in unaryF:Data[param] = Data[unaryC[param]].apply(unaryF[param])
        Data["Mes"] = pd.to_datetime(Data["Mes"])
        
        linearTrans = {
            "DuracionS": (lambda D: 
                (lambda x, y: D[y].sub(D[x]))
                         ) ,
            "Ruta": ( lambda D:
                (lambda x, y: D[y].add(D[x]))
            )
        }
        linearTransArgs = {
            "DuracionS": ("InicioS", "FinS" ),
            "Ruta" : ("Origen_Id", "Destino_Id")
        }
        for param in linearTrans:
            Data[param] = linearTrans[param](Data)(*linearTransArgs[param])
        Data["Duracion"] = Data["DuracionS"]//60
        Data["Costo"] = Data.apply(lambda x: Utils.calcCost( x.Duracion,x.Semana), axis=1)
        return Data
    @staticmethod
    def getPreDateTime(Data, by_="DateInicio"):
        """
        ['Viaje_Id', 'Usuario_Id', 'Genero', 'AÃ±o_de_nacimiento',
       'Inicio_del_viaje', 'Fin_del_viaje', 'Origen_Id', 'Destino_Id',
       'DateInicio', 'DateFin', 'InicioS', 'FinS', 'InicioH', 'FinH', 'Semana',
       'Mes', 'Edad', 'SegundosO', 'SegundosD', 'DuracionS', 'Ruta',
       'Duracion']
       """
        Copy = Data.copy()
        Copy["Datetime"] = pd.to_datetime(Copy[by_])
        for toDrop in ("Viaje_Id", "Usuario_Id", "Genero", "AÃ±o_de_nacimiento", "Origen_Id", "Destino_Id", "Inicio_del_viaje", "Fin_del_viaje", "DateFin", "DateInicio", "Mes"):
            Copy.drop(toDrop, axis=1, inplace=True)
        return Copy
    @staticmethod
    def getAvg(Data, X, by_="DateInicio"):
        Copy = Utils.getPreDateTime(Data, by_)
        CopyGrouped = Copy.groupby("Datetime").mean().reset_index()
        Copy = CopyGrouped.set_index("Datetime")
        return Copy[X]
    @staticmethod 
    def getSum(Data, X,by_="DateInicio"):
        Copy = Utils.getPreDateTime(Data)
        CopyGrouped = Copy.groupby("Datetime").sum().reset_index()
        Copy = CopyGrouped.set_index("Datetime")
        return Copy[X]
    @staticmethod
    def getCount(Data, X,by_="DateInicio"):
        Copy = Utils.getPreDateTime(Data, by_)
        CopyGrouped = Copy.groupby("Datetime").count().reset_index()
        Copy = CopyGrouped.set_index("Datetime")
        return Copy[X]

    @staticmethod
    def cleanUp( Data : pd.DataFrame):
        Data.dropna(subset=["Origen_Id"], inplace=True)
        Data.dropna(subset=["Destino_Id"], inplace=True)
        Data.dropna(axis="columns")
     
        Data = Utils.createColumns(Data)
        col = "DuracionS"
        q_low = Data[col].quantile(0.01)
        q_hi  = Data[col].quantile(0.99)

        anomalInd = (Data[(Data[col] < q_low) | (Data[col] > q_hi)]).index

        Data.drop(
            index = Data[ (Data["Edad"]<18 ) ].index,
             inplace=True
        )

        Data.drop(
            index = Data[ 
                (Data["InicioH"] == 0) & (Data["FinH"] == 0) & 
                (Data["InicioS"] == 0) & (Data["FinS"] == 0 )
            ].index ,
            inplace=True
        )

        Data.drop(
            index=(Data[Data["Duracion"]<0]).index,
            inplace=True 
        )

        Data.drop(
            index=(anomalInd),
            inplace=True
        )
        Data.drop(
            index=(Data[Data["Duracion"]>1440].index)
            ,inplace=True
        )
        Data.drop(
            index=((Data[Data["Edad"]>100]).index)
            ,inplace=True
        )
        #Data.drop(
        #    index=((Data[Data["SegundosO"]==0 and Data["SegundosD"]==0]).index),
        #    inplace=True
        #)
        return Data

