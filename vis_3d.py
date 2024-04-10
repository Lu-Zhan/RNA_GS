import torch
import matplotlib.pyplot as plt
#from mpl_toolkits.mplot3d import Axes3D

data = torch.load('094rjvec/3d_pts.pth')

x = data['xy'][:, 0].cpu().numpy()
x = x / max(x)
y = data['xy'][:, 1].cpu().numpy()
y = y = -((y / max(y) ) - 0.5) + 0.5
z = data['z'].cpu().numpy()
z = (z / 2.) + 0.5
opacity = data['ref_score']
opacity = torch.clamp(opacity, 0, 1).cpu().numpy()
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(y, x, z, c='red', marker='o', alpha=opacity, s=4)

# ax.scatter(x, y, z, c='red', marker='o', alpha=opacity, s=2)

ax.view_init(elev=-90, azim=0)
ax.set_xlim([0, 1])  
ax.set_ylim([0, 1])  
ax.set_zlim([0, 1])
ax.set_xlabel('Y')
ax.set_ylabel('X')
ax.set_zlabel('Z')
ax.set_title('pred_points')
plt.savefig('3d_pts.png')



# 2d
# x = data['xy'][:, 0].cpu().numpy()
# x = x / max(x)
# y = data['xy'][:, 1].cpu().numpy()
# y = -((y / max(y) ) -0.5)+0.5
# z = data['z'].cpu().numpy()
# opacity = data['ref_score']
# opacity = torch.clamp(opacity, 0, 1).cpu().numpy()
# # fig = plt.figure(figsize=(10, 7))
# # ax = fig.add_subplot(111, projection='3d')
# fig, ax = plt.subplots(figsize=(10, 7))
# ax.scatter(x, y, c='red', marker='o', alpha=opacity, s=2)

# # ax.scatter(x, y, z, c='red', marker='o', alpha=opacity, s=2)

# #ax.view_init(elev=90, azim=90)
# ax.set_aspect('equal')
# ax.set_xlim([0, 1])  
# ax.set_ylim([0, 1])  
# # ax.set_zlim([-8, 8])
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# # ax.set_zlabel('Z')
# ax.set_title('pred_points')

# plt.savefig('2d_pts.png')

