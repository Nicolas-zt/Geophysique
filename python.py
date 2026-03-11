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
    
    # Combiner puis convertir
    data["datetime"] = pd.to_datetime(
        data["date_str"] + data["heure_str"],
        format="%Y%m%d%H%M%S"
    )
    return data

def plot_df(datas, names):
    fig, ax = plt.subplots(len(datas), 3, figsize=(10, 4*len(datas)), sharex=True)
    j=0
    for d in ["dE","dN","dU"]:
        for i in range(len(datas)):
            ax[i,j].plot(
                datas[i]["datetime"],
                datas[i][d],
                marker="+",
                linestyle="-"
            )
            ax[i,j].set_ylabel(f"{names[i]} {d} (m)")
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
            print(annee_debut,annee_fin)
            liste_chgmt.append([(int(annee_debut) + int(jour_debut)/365),(int(annee_fin) + int(jour_fin)/365)])
    return liste_chgmt


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
    
    plot_df([CNBA,DINO,UCOR], names)
    
    fig,ax = plt.subplots(2,1)
    ax[0].hist(mag,bins = 50)
    
    
    ax[1].hist(mag,bins=50,log=True)
    fig.show()
    
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.coastlines()
    ax.scatter(data[:,2],data[:,1],color = 'green')
    ax.scatter(gnss["long."].values,gnss["lat."].values,marker = "^", color = "pink")
    for index,row in gnss.iterrows():
        ax.text(row["long."],row["lat."],row["site"])
    ax.text(data[4,2],data[4,1],"SEISME")
    fig.show()
    
    #axvline chgmt antenne
