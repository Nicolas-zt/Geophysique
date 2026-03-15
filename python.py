import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import pyproj
import re

def df_datetime(data):
    # Convertir en string avec zéros en tête si nécessaire
    data["date_str"] = data["*YYYYMMDD"].astype(str)
    data["heure_str"] = data["HHMMSS"].astype(str).str.zfill(6)
    
    data["datetime"] = pd.to_datetime(
        data["date_str"] + data["heure_str"],
        format="%Y%m%d%H%M%S"
    )
    
    # Calcul année décimale
    year = data["datetime"].dt.year
    start_year = pd.to_datetime(year.astype(str) + "-01-01")
    next_year = pd.to_datetime((year + 1).astype(str) + "-01-01")
    
    data["datetime"] = year + (data["datetime"] - start_year) / (next_year - start_year)  
    return data


def plot_df(datas, names, changement):
    fig, ax = plt.subplots(len(datas), 3, figsize=(10, 4*len(datas)), sharex=True)
    j=0
    for d in [("dE","Se"),("dN","Sn"),("dU","Su")]:
        for i in range(len(datas)):
            ax[i,j].plot(
                datas[i]["datetime"],
                datas[i][d[0]],
                linestyle="-",
                linewidth=1
            )
            
            ax[i,j].fill_between(datas[i]["datetime"],
                                 datas[i][d[0]]+2*datas[i][d[1]],
                                 datas[i][d[0]]-2*datas[i][d[1]],
                                 facecolor='red', alpha=0.5)
            
            
            for dates in changement[j]:
                if dates[1] != 9999:
                    ax[i,j].axvline(x=dates[1],c="red",linestyle="--")
                    
            ax[i,j].axvline(x=2010+58/365.25,c="green",linestyle="--")
            ax[i,j].axvline(x=2015+259/365.25,c="green",linestyle="--")
            ax[i,j].axvline(x=2014+91/365.25,c="green",linestyle="--")
            
            ax[i,j].set_ylabel(f"{names[i]} {d[0]} (m)")
            ax[i,j].set_xlabel("t (Decimal Year)")
            ax[i,j].grid(True)
        j+=1


    plt.tight_layout()
    plt.show()
    
def changement_antenne(name, file):
    fichier = open(file, "r")
    liste_chgmt = []
    for ligne in fichier.readlines():
        if ligne[1:5]==name:
            split = re.split(r'  +', ligne)
            annee_debut = split[2].split()[0]
            jour_debut = split[2].split()[1]
            annee_fin = split[3].split()[0]
            if annee_fin == "9999":
                jour_fin = "0"
            else:
                jour_fin = split[3].split()[1]
            liste_chgmt.append([(int(annee_debut) + int(jour_debut)/365),(int(annee_fin) + int(jour_fin)/365)])
    return liste_chgmt


def H(x):
    if x<0:
        return 0
    else:
        return 1
    
def H_(t,t0):
    t = np.where(t<t0,0,1)
    return t

def MC_lineaire(A,B,P):
    N = A.T@P@A
    K = A.T@P@B
    X_ = np.linalg.inv(N)@K
    return X_
    
def MC(A,B,P,X):
    
    dX = 1e6
    i = 0
    while dX.any() > 1e6 or i>=50:
        
        N = A.T@P@A
        K = A.T@P@B
        
        dX = np.linalg.inv(N)@K
        X += dX
        i += 1
    
    V = B - A@X
    sigma_0 = np.sqrt(V.T@P@V/(A.shape[0]-B.shape[0]))
    
    return X,V,sigma_0

def MC_saisonnier(station, coord,plot=False):
    """Station est le df de la station
    coord est soit "dN", "dE" ou "dU", càd la série temporelle à laquelle on s'interesse"""

    #Pour récupérer les écart-types dans la bonne colonne
    table_ecart_type = {"dN":"Sn", "dE":"Se", "dU":"Su"}
    ecart_type = table_ecart_type[coord]
    
    #Définition des tailles de matrice
    A = np.zeros((station.shape[0], 6))
    P = np.identity(station.shape[0])
    B = np.zeros((station.shape[0], 1))
    
    #Remplissage des matrices
    for i in range(0,station.shape[0]):
        ligne = station.iloc[i]
        t = ligne["datetime"]
        A[i,:] = [t,1,np.cos(2*np.pi*t),np.sin(2*np.pi*t),np.cos(4*np.pi*t), np.sin(4*np.pi*t)]
        P[i,i] = 1/ligne[ecart_type]**2
        B[i,:] = ligne[coord]
        
    X_ = MC_lineaire(A,B,P)
    
    if plot:
        a,b,c,d,e,f = X_[0,0],X_[1,0],X_[2,0],X_[3,0],X_[4,0],X_[5,0]
        t = station["datetime"]
        fig, ax = plt.subplots()
        ax.plot(t, station[coord])
        ax.plot(t,a*t+b+c*np.cos(2*np.pi*t)+d*np.sin(2*np.pi*t)+e*np.cos(4*np.pi*t)+f*np.sin(4*np.pi*t))
        plt.show()
        
    return X_


def MC_sauts(station, coord,dates,plot = False):
    
    #Pour récupérer les écart-types dans la bonne colonne
    table_ecart_type = {"dN":"Sn", "dE":"Se", "dU":"Su"}
    ecart_type = table_ecart_type[coord]
    
    #Définition des tailles de matrice
    A = np.zeros((station.shape[0], 6 + len(dates)))
    P = np.identity(station.shape[0])
    B = np.zeros((station.shape[0], 1))
    
    #Remplissage des matrices
    for i in range(0,station.shape[0]):
        ligne = station.iloc[i]
        t = ligne["datetime"]
        A[i,:] = np.hstack(([t,1,np.cos(2*np.pi*t),np.sin(2*np.pi*t),np.cos(4*np.pi*t), np.sin(4*np.pi*t)]
                            , np.array([H(t-date)for date in dates])))
        P[i,i] = 1/ligne[ecart_type]**2
        B[i,:] = ligne[coord]
    X_ = MC_lineaire(A,B,P)
   
    if plot:
        a,b,c,d,e,f = X_[0,0],X_[1,0],X_[2,0],X_[3,0],X_[4,0],X_[5,0]
        t = station["datetime"]
        fig, ax = plt.subplots()
        ax.plot(t, station[coord])
        ax.plot(t,a*t+b+c*np.cos(2*np.pi*t)+d*np.sin(2*np.pi*t)+e*np.cos(4*np.pi*t)+f*np.sin(4*np.pi*t)+np.sum(X_[6+i,0]*H_(t,date)for i,date in enumerate(dates)))
        plt.show()
       
    return X_
 
if __name__ == "__main__":
    data = np.genfromtxt("query.csv",usecols=[0,1,2,3,4],encoding = "utf-8",
                         delimiter = ",",skip_header = 1)
    gnss = pd.read_csv("cGPS.dat",delimiter = "\s+")
    sta = ["ANTC","ATJN","AZUL","CMBA","CNBA","CONS","CONZ","DINO","IGM1","IQQE","MAUL","MNMI","PEDR","PICC","SRLP","UCOR","URUS"]
    gnss = gnss[gnss["site"].isin(sta)]
    mag = data[:,4]
    data = data[data[:,4]>8]
    
    ### Séries temporelles
    names = ["CNBA","DINO","UCOR"]
    CNBA = pd.read_csv("time-series/CNBA.pos", header=36, delimiter=r"\s+")
    DINO = pd.read_csv("time-series/DINO.pos", header=36, delimiter=r"\s+")
    UCOR = pd.read_csv("time-series/UCOR.pos", header=36, delimiter=r"\s+")
    #Conversion date
    CNBA = df_datetime(CNBA)
    DINO = df_datetime(DINO)
    UCOR = df_datetime(UCOR)
    
    ### Changements antennes
    chgmt_UCOR = changement_antenne("UCOR", "materiel.dat")
    chgmt_CNBA = changement_antenne("CNBA", "materiel.dat")
    chgmt_DINO = changement_antenne("DINO", "materiel.dat")
    
    changement = [chgmt_UCOR,chgmt_CNBA,chgmt_DINO]
    plot_df([CNBA,DINO,UCOR], names, changement)
    
    fig,ax = plt.subplots(2,1)
    ax[0].hist(mag,bins = 50)
    
    
    ax[1].hist(mag,bins=50,log=True)
    fig.show()
    
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.scatter(data[:,2],data[:,1],color = 'green')
    ax.scatter(gnss["long."].values,gnss["lat."].values,marker = "^", color = "red")
    for index,row in gnss.iterrows():
        ax.text(row["long."],row["lat."],row["site"])
    """"ax.text(data[4,2],data[4,1],"SEISME")"""
    fig.show()
    
    # print(MC_saisonnier(CNBA, "dN",True))
    print(MC_sauts(CNBA,"dN",[2010+58/365.25,2015+259/365.25,2014+91/365.25],True))
    
