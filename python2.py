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
            
            
            for dates in changement[i]:
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
    if isinstance(x, (pd.Series, pd.DataFrame)):
        return (x >= 0).astype(int)
    else:
        return 0 if x < 0 else 1

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

def MC_saisonnier(Station, coord, date_debut, date_fin, plot=False):
    """Station est le df de la station
    date_debut, date_fin sont les dates sur laquelle on estime la composante saisonnière et la tendance linéaire
    coord est soit "dN", "dE" ou "dU", càd la série temporelle à laquelle on s'interesse"""

    #Pour récupérer les écart-types dans la bonne colonne
    table_ecart_type = {"dN":"Sn", "dE":"Se", "dU":"Su"}
    ecart_type = table_ecart_type[coord]
    
    station = Station.copy()
    station = station[station["datetime"] > date_debut]
    station = station[station["datetime"] < date_fin]
    
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
    
    
    a,b,c,d,e,f = X_[0,0],X_[1,0],X_[2,0],X_[3,0],X_[4,0],X_[5,0]
    t = Station["datetime"]
    f = a*t+b+c*np.cos(2*np.pi*t)+d*np.sin(2*np.pi*t)+e*np.cos(4*np.pi*t)+f*np.sin(4*np.pi*t)
    Station[coord+"_signal_saisonnier"] = Station[coord] - f
    
    if plot:
        fig, ax = plt.subplots()
        ax.plot(t, Station[coord])
        ax.plot(t,f)
        ax.plot(t,Station[coord+"_signal_saisonnier"])
        plt.show()
        
    return round(X_[0,0],3),round(X_[2,0],3),round(X_[3,0],3),round(X_[4,0],3),round(X_[5,0],3)
def MC_cosismic(station, coord, date_chgmt_antenne, date_seisme, plot=False):
    """ Station est le df de la station
    coord est soit "dN", "dE" ou "dU", càd la série temporelle à laquelle on s'interesse
    date_chgmt_antenne : Liste de date
    date_seisme : Liste de date """

    #Pour récupérer les écart-types dans la bonne colonne
    table_ecart_type = {"dN":"Sn", "dE":"Se", "dU":"Su"}
    ecart_type = table_ecart_type[coord]
    
    #Création de variables
    n=len(date_chgmt_antenne)
    m=len(date_seisme)
    
    
    #Définition des tailles de matrice
    A = np.zeros((station.shape[0], n+m))
    P = np.identity(station.shape[0])
    B = np.zeros((station.shape[0], 1))
    
    #Remplissage des matrices
    for i in range(0,station.shape[0]):
        ligne = station.iloc[i]
        t = ligne["datetime"]
        A[i,:] = [H(t-date_chgmt_antenne[j]) for j in range(n)] + [H(t-date_seisme[j]) for j in range(m)]
        P[i,i] = 1/ligne[ecart_type]**2
        B[i,:] = ligne[coord+"_signal_saisonnier"]
        
    X_ = MC_lineaire(A,B,P)
    
    t = station["datetime"]
    new_signal = station[coord+"_signal_saisonnier"]
    saut = 0
    
    for j in range(n):
        new_signal -= X_[saut,0]*H(t-date_chgmt_antenne[j])
        saut+=1
    for j in range(m):
        new_signal -= X_[saut,0]*H(t-date_seisme[j])
        saut+=1
        
    station[coord + "_cosismic"] = new_signal
    
    if plot:
        fig, ax = plt.subplots()
        ax.plot(t, station[coord])
        ax.plot(t, station[coord + "_cosismic"])
        plt.show()
    
    return X_


#Estimation de post-sismique et co-sismique en même temps
def MC_postsismic(station, coord, date_chgmt_antenne, date_seisme, sol_init, plot=False):
    table_ecart_type = {"dN":"Sn", "dE":"Se", "dU":"Su"}
    ecart_type = table_ecart_type[coord]
    
    n = len(date_chgmt_antenne)
    m = len(date_seisme)
    
    sol = sol_init.copy().reshape(len(sol_init), 1)
    
    A = np.zeros((station.shape[0], n + 3*m))
    P = np.identity(station.shape[0])
    B = np.zeros((station.shape[0], 1))
    
    def f(time, solution):
        s = 0
        
        # Sauts antenne
        for j in range(n):
            ck = solution[j, 0]
            s += ck * H(time - date_chgmt_antenne[j])
            
        # Sauts séisme
        for j in range(m):
            ck = solution[n + j, 0]
            dt = time - date_seisme[j]
            s += ck * H(dt)
        
        # Post-sismique
        for j in range(m):
            Ak = solution[n + m + j, 0]
            ak = solution[n + 2*m + j, 0]
            dt = time - date_seisme[j]
            s += Ak * np.exp(-ak * dt) * H(dt)
            
        return s
    
    for k in range(20):
        for i in range(station.shape[0]):
            ligne = station.iloc[i]
            t = ligne["datetime"]
            
            # paramètres ak (taux)
            a0 = sol[n + 2*m : n + 3*m]
            
            A[i, :] = (
                [H(t - date_chgmt_antenne[j]) for j in range(n)]
                + [H(t - date_seisme[j]) for j in range(m)]
                + [np.exp(-a0[j, 0] * (t - date_seisme[j])) * H(t - date_seisme[j]) for j in range(m)]
                + [-(t - date_seisme[j]) * np.exp(-a0[j, 0] * (t - date_seisme[j])) * H(t - date_seisme[j]) for j in range(m)]
            )
            
            P[i, i] = 1 / ligne[ecart_type]**2
            B[i, :] = ligne[coord+"_signal_saisonnier"] - f(t, sol)

        dX_ = MC_lineaire(A, B, P)
        sol += dX_
    
    if plot:
        fig, ax = plt.subplots()
        l = []
        
        for k in range(station.shape[0]):
            ligne = station.iloc[k]
            t = ligne["datetime"]
            l.append(f(t, sol))
        
        t = station["datetime"]
        ax.plot(t, station[coord+"_signal_saisonnier"])
        ax.plot(t, l)
        plt.show()
    
    return sol


#Estimation du séisme sur une période de X années après le séisme
def Estimation_POST(station,coord,periode_evaluation,A,a):
    Amplitude_t0 = A*np.exp(0)
    Amplitude_t = A*np.exp(-a*periode_evaluation)
    return Amplitude_t - Amplitude_t0
    
    
    



if __name__ == "__main__":
#%%
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
    
    changement = [chgmt_CNBA,chgmt_DINO,chgmt_UCOR]
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
        
    extent = ax.get_extent()
    fig.show()
    
    #Estimation long terme CNBA
    A,C,D,E,F = MC_saisonnier(CNBA, "dE",2006, 2015.66)
    E_Long_terme = A
    print(f"CNBA_Est \n Vitesse long terme : {A} m/an \n Amplitude annuelle : E : {C} m ; N : {D} m \n Amplitude semi-annuelle : E : {E} m ; N : {F} m")
    A,C,D,E,F = MC_saisonnier(CNBA, "dN",2016, 2022)
    N_Long_terme = A
    print(f"CNBA_Nord \n Vitesse long terme : {A} m/an \n Amplitude annuelle : E : {C} m ; N : {D} m \n Amplitude semi-annuelle : E : {E} m ; N : {F} m")
    CNBA_Long_terme = float(E_Long_terme), float(N_Long_terme)
    print(CNBA_Long_terme)
    
    #Estimation long terme DINO
    A,C,D,E,F = MC_saisonnier(DINO, "dE",2012.00, 2015.66)
    E_Long_terme = A
    print(f"DINO_Est \n Vitesse long terme : {A} m/an \n Amplitude annuelle : E : {C} m ; N : {D} m \n Amplitude semi-annuelle : E : {E} m ; N : {F} m")
    A,C,D,E,F = MC_saisonnier(DINO, "dN",2017.08, 2021.9)
    N_Long_terme = A
    print(f"DINO_Nord \n Vitesse long terme : {A} m/an \n Amplitude annuelle : E : {C} m ; N : {D} m \n Amplitude semi-annuelle : E : {E} m ; N : {F} m")
    DINO_Long_terme = float(E_Long_terme), float(N_Long_terme)
    
    #Estimation long terme UCOR
    A,C,D,E,F = MC_saisonnier(UCOR, "dE",2010.20, 2015.66)
    E_Long_terme = A
    print(f"UCOR_Est \n Vitesse long terme : {A} m/an \n Amplitude annuelle : E : {C} m ; N : {D} m \n Amplitude semi-annuelle : E : {E} m ; N : {F} m")
    A,C,D,E,F = MC_saisonnier(UCOR, "dN",2010.4, 2017.46)
    N_Long_terme = A
    print(f"UCOR_Nord \n Vitesse long terme : {A} m/an \n Amplitude annuelle : E : {C} m ; N : {D} m \n Amplitude semi-annuelle : E : {E} m ; N : {F} m")
    UCOR_Long_terme = float(E_Long_terme), float(N_Long_terme)
    
    #Element pour carto
    lat_CNBA = gnss.loc[gnss["site"] == "CNBA", "lat."].iloc[0]
    lon_CNBA = gnss.loc[gnss["site"] == "CNBA", "long."].iloc[0]
    lat_DINO = gnss.loc[gnss["site"] == "DINO", "lat."].iloc[0]
    lon_DINO = gnss.loc[gnss["site"] == "DINO", "long."].iloc[0]
    lat_UCOR = gnss.loc[gnss["site"] == "UCOR", "lat."].iloc[0]
    lon_UCOR = gnss.loc[gnss["site"] == "UCOR", "long."].iloc[0]
    
    #Carte vitesse long-terme
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.plot()
    ax.scatter([lon_CNBA,lon_DINO,lon_UCOR],[lat_CNBA,lat_DINO,lat_UCOR],color = 'green', transform=ccrs.PlateCarree())
    ax.quiver([lon_CNBA,lon_DINO,lon_UCOR],[lat_CNBA,lat_DINO,lat_UCOR],
              [CNBA_Long_terme[0]*100,DINO_Long_terme[0]*100,UCOR_Long_terme[0]*100],
              [CNBA_Long_terme[1]*100,DINO_Long_terme[1]*100,UCOR_Long_terme[1]*100],
              transform=ccrs.PlateCarree())
    

    CosismicE_CNBA = MC_cosismic(CNBA, "dE",[2021.4493],[2010+58/365.25,2015+259/365.25,2014+91/365.25])
    print(f"Amplitude Cosismique Est Iquique sur CNBA : {round(CosismicE_CNBA[2,0],3)} m")
    CO_E_CNBA = round(CosismicE_CNBA[2,0],3)
    
    CosismicN_CNBA = MC_cosismic(CNBA, "dN",[2021.4493],[2010+58/365.25,2015+259/365.25,2014+91/365.25])
    print(f"Amplitude Cosismique Nord Iquique sur CNBA : {round(CosismicN_CNBA[2,0],3)} m")
    CO_N_CNBA = round(CosismicN_CNBA[2,0],3)
    
    CosismicE_DINO = MC_cosismic(CNBA, "dE",[2016.9644],[2015+259/365.25,2014+91/365.25])
    print(f"Amplitude Cosismique Est Iquique sur DINO : {round(CosismicE_DINO[1,0],3)} m")
    CO_E_DINO = round(CosismicE_DINO[1,0],3)
    
    CosismicN_DINO = MC_cosismic(CNBA, "dN",[2016.9644],[2015+259/365.25,2014+91/365.25])
    print(f"Amplitude Cosismique Nord Iquique sur DINO : {round(CosismicN_DINO[1,0],3)} m")
    CO_N_DINO = round(CosismicN_DINO[1,0],3)
    
    CosismicE_UCOR = MC_cosismic(CNBA, "dE",[2008.8767,2017.6438,2019.4137],[2010+58/365.25,2015+259/365.25,2014+91/365.25])
    print(f"Amplitude Cosismique Est Iquique sur UCOR : {round(CosismicE_UCOR[4,0],3)} m")
    CO_E_UCOR = round(CosismicE_UCOR[4,0],3)
    
    CosismicN_UCOR = MC_cosismic(CNBA, "dN",[2008.8767,2017.6438,2019.4137],[2010+58/365.25,2015+259/365.25,2014+91/365.25])
    print(f"Amplitude Cosismique Nord Iquique sur UCOR : {round(CosismicN_UCOR[4,0],3)} m")
    CO_N_UCOR = round(CosismicN_UCOR[4,0],3)
    
    #Carte cosismique
    fig = plt.figure()
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.plot()
    ax.scatter([lon_CNBA,lon_DINO,lon_UCOR],[lat_CNBA,lat_DINO,lat_UCOR],color = 'green', transform=ccrs.PlateCarree())
    ax.quiver([lon_CNBA,lon_DINO,lon_UCOR],[lat_CNBA,lat_DINO,lat_UCOR],
              [CO_E_CNBA*100,CO_E_DINO*100,CO_E_UCOR*100],
              [CO_N_CNBA*100,CO_N_DINO*100,CO_N_UCOR*100],
              transform=ccrs.PlateCarree())
    
    Estimation_CNBA = MC_postsismic(CNBA, "dE",[2021.4493],[2010+58/365.25,2015+259/365.25,2014+91/365.25],np.array([0]+[0,0,0]+[0.1,0.1,0.1]+[1,1,1]))
    