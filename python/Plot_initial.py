import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
prob = 'NSE'
# ty   = 'initial'
ty   = 'Final'

df=pd.read_csv(f'../Data/{prob}_{ty}.csv')

# x,y,rho,u,v,w

Nx,Ny=256,256

data = {col : df[col].values.reshape(-1,Nx,Ny) for col in df.columns}
print(list(data.keys()))
data['x']=data['x'][0]
data['y']=data['y'][0]

t0=10
plt.figure(figsize=(7, 6))
mesh = plt.pcolormesh(data['x'], data['y'], data['rho'][t0], shading='auto', cmap='jet')
plt.colorbar(mesh, label='Physical Field U')
plt.xlabel('X Coordinate')
plt.ylabel('Y Coordinate')
plt.title(f'2D Hydrodynamics Simulation | step = {t0}')

plt.show()

# old
# X = data['x'].values.reshape(Nx,Ny)
# Y = data['y'].values.reshape(Nx,Ny)
# U = data['u0'].values.reshape(Nx, Ny)

# plt.figure(figsize=(7, 6))
# mesh = plt.pcolormesh(X, Y, U, shading='auto', cmap='jet')
# plt.colorbar(mesh, label='Physical Field U')
# plt.xlabel('X Coordinate')
# plt.ylabel('Y Coordinate')
# plt.title('2D Hydrodynamics Simulation Initial State')
# plt.savefig('fluid_density.png', dpi=300)
# plt.show()


