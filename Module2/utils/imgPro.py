import torch
from torch import Tensor

def split_image_to_4(image:Tensor):
    out = []
    height, width = image.shape[-2:]
    mid_height = height // 2
    mid_width = width // 2

    if image.dim() ==3:
        image = image.unsqueeze(0)
    top_left = image[:, :, :mid_height, :mid_width]
    top_right = image[:, :, :mid_height, mid_width:]
    bottom_left = image[:, :, mid_height:, :mid_width]
    bottom_right = image[:, :, mid_height:, mid_width:]
    return top_left,top_right,bottom_left,bottom_right

def merge_4_image(top_left,top_right,bottom_left,bottom_right):
    top = torch.cat((top_left, top_right), dim=2)
    bottom = torch.cat((bottom_left, bottom_right), dim=2)
    reconstructed_image = torch.cat((top, bottom), dim=1)
    return reconstructed_image
    
class KrizhevskyColorAugmentation(object):
    def __init__(self, sigma=0.5):
        self.sigma = sigma
        self.mean = torch.tensor([0.0])
        self.deviation = torch.tensor([sigma])

    def __call__(self, img, color_vec=None):
        sigma = self.sigma
        # for color augmentation, computed by origin author
        U = torch.tensor([[-0.56543481, 0.71983482, 0.40240142],
                        [-0.5989477, -0.02304967, -0.80036049],
                        [-0.56694071, -0.6935729, 0.44423429]], dtype=torch.float32)
        EV = torch.tensor([1.65513492, 0.48450358, 0.1565086], dtype=torch.float32)

        if color_vec is None:
            if not sigma > 0.0:
                color_vec = torch.zeros(3, dtype=torch.float32)
            else:
                color_vec = torch.distributions.Normal(self.mean, self.deviation).sample((3,))
            color_vec = color_vec.squeeze()

        alpha = color_vec * EV
        noise = torch.matmul(U, alpha.t())
        noise = noise.view((3, 1, 1))
        return img + noise