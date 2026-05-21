import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from glob import glob 
# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
prob = 'NSE'
ty   = 'initial'
# ty   = 'Final'
filename = f"../Data/{prob}/{prob}_*{ty}.csv"
files=glob(filename)

print(files)

for f0 in [0]:
    f= open(files[0])
    numline = len(f.readlines())-1
    f.close()
    Nx=int(numline**0.5)

    df=pd.read_csv(files[f0])
    data = {col : df[col].values.reshape(Nx,Nx) for col in df.columns}
    print(list(data.keys()))
    # data['x']=data['x'][0]
    # data['y']=data['y'][0]
    print(data['rho'][Nx//2,:])

    plt.figure(figsize=(7, 6))
    mesh = plt.pcolormesh(data['x'], data['y'], data['rho'], shading='auto', cmap='jet')
    plt.colorbar(mesh, label='Physical Field U')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.title(f'2D Hydrodynamics Simulation | step =')
    # plt.show()
    # print(files[f0])
    savename=files[f0].split('/')[-1].split('.')[0]+ '.png'
    # print(savename)
    plt.savefig(savename)
