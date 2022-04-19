import torch
import matplotlib.pyplot as plt

def pcd_3d_vis(pcd):
    plt.figure(figsize=(5,5))
    ax = plt.axes(projection='3d')
    ax.axis("on")
    plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)

    ax.scatter(pcd[:,0],pcd[:,1],pcd[:,2], s=0.04)
    ax.set_xlim3d(-1,1)
    ax.set_ylim3d(-1,1)
    ax.set_zlim3d(-1,1)
    ax.grid()
    ax.view_init(-5, 10)
    
    plt.show()
    
