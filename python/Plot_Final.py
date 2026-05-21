import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob 
import os
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
ST=5
prob = 'NSE_test1'
# ty   = 'initial'
ty   = 'Final'
filename = f"../Data/{prob}/{prob}_*{ty}.csv"
files=glob(filename)

if not os.path.isdir(prob):
    os.mkdir(f'./{prob}')

for f0 in range(len(files)):

# for f0 in [0]:
    print(files[f0])
    f= open(files[f0])
    numline = len(f.readlines())-1
    f.close()
    Nx=int(numline**0.5)

    df=pd.read_csv(files[f0])
    data = {col : df[col].values.reshape(Nx,Nx) for col in df.columns}
    # print(list(data.keys()))

    plt.figure(figsize=(7, 6))
    mesh = plt.pcolormesh(data['x'], data['y'], data['rho'], shading='auto', cmap='jet')
    quiv = plt.quiver(data['x'][::ST,::ST], data['y'][::ST,::ST]
                    ,data['mu'][::ST,::ST],data['mv'][::ST,::ST])
    plt.colorbar(mesh, label='Physical Field U')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'2D Hydrodynamics Simulation | step = {f0}')
    
    savename=files[f0].split('/')[-1].split('.')[0]+ '.png'
    
    plt.savefig(f'./{prob}/{savename}',dpi = 400)
    plt.close()