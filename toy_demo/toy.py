import torch
import numpy as np
import matplotlib.pyplot as plt


from systems.models import FixGaussModel

gs_model = FixGaussModel(num_primarys=1, num_backups=0, hw=(256, 256), device='cuda:0')
gs_model.rgbs = torch.ones_like(gs_model.rgbs)[..., :1]
gs_model.opacities = torch.ones_like(gs_model.opacities)[..., :1]
gs_model.background = torch.zeros_like(gs_model.background)[..., :1]
gs_model.means_3d = torch.zeros_like(gs_model.means_3d)
gs_model.scales = torch.ones_like(gs_model.scales)

image, _, radii, _ = gs_model.render()
print(radii)

plt.subplot(121)
plt.imshow(image.data.cpu().numpy(), cmap='gray')
plt.title(f'scale=1, radii={int(radii)}')

# gs_model.scales[..., :] = 0.2 * gs_model.scales[..., :]
gs_model.means_3d[:, -2] = -1
# gs_model.scales = torch.ones_like(gs_model.scales)

image, _, radii, _ = gs_model.render()
print(radii)
plt.subplot(122)

plt.imshow(image.data.cpu().numpy(), cmap='gray')
plt.title(f'scale=0.2, radii={int(radii)}')

plt.savefig('scale test.png')
