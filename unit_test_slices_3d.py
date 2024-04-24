import torch

from matplotlib import pyplot as plt
from systems.cameras import SliceCamera
from systems.models_3d import GaussModel


if __name__ == '__main__':
    camera = SliceCamera(
        num_slice=40,
        num_dims=1,
        hw=(128, 128),
        step_z=0.05,
        camera_z=-8,
    )
    gs_model = GaussModel(num_primarys=1, num_backups=0, device='cuda', camera=camera)

    gs_model.rgbs = torch.ones_like(gs_model.rgbs)[..., :1] 
    gs_model.opacities = torch.ones_like(gs_model.opacities)[..., :1]
    gs_model.background = torch.zeros_like(gs_model.background)[..., :1]
    gs_model.means_3d = torch.tensor([[0, 0, 0.5]], device='cuda')

    gs_model.scales = torch.ones_like(gs_model.scales) * 2
    # gs_model.scales[:, -2] = gs_model.scales[:, -2] / 2
    # gs_model.scales[:, -2] /= 2
    gs_model.quats = torch.zeros_like(gs_model.quats)
    # gs_model.scales = gs_model.scales / gs_model.scales.max()

    w = 5
    h = 8
    bias = 2

    import time

    start = time.time()
    images = gs_model.render_slice(camera=camera, index=[x for x in range(h*w)])[0] # (20, h, w, 1)
    end1 = time.time()

    # image_list = []
    # for i in range(0, h * w):
    #     image = gs_model.render_slice(camera=camera, index=[i])[0] # (1, h, w, 1)
    #     image_list.append(image)
    
    # images = torch.cat(image_list, dim=0) # (20, h, w, 1)

    end2 = time.time()

    print(f'end1 - start: {end1 - start}')
    print(f'end2 - end1: {end2 - end1}')

    vmax = images.max()

    for i in range(0, h * w):
        plt.subplot(h, w, i+1)
        # print(images[i, ..., 0].max())
        plt.imshow(images[i, ..., 0].data.cpu().numpy(), cmap='jet', vmax=vmax, vmin=0)
        # plt.title(f'z={i * 0.1 - 2:.1f}')
        plt.tight_layout()
        plt.axis('off')

    plt.savefig('new_test_3d_128_-0.5_new.png')

    print('done.')