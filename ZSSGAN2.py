import torch
from sg2_model import Generator


class ZSSGAN2(torch.nn.Module):
    def __init__(self, device, checkpoint_frozen_path, checkpoint_trainable_path, img_size):
        super(ZSSGAN, self).__init__()
        # gpu设备
        self.device = device

        # Set up frozen (source) generator
        self.generator_frozen = Generator(
            size=img_size, style_dim=512, n_mlp=8
        ).to(device)

        # load model
        checkpoint_frozen = torch.load(checkpoint_frozen_path, map_location=device)
        self.generator_frozen.load_state_dict(checkpoint_frozen["g_ema"], strict=True)

        # 调整为eval模式
        self.generator_frozen.eval()

        # Set up trainable (target) generator
        self.generator_trainable = Generator(
            size=img_size, style_dim=512, n_mlp=8
        ).to(device)

        # load model
        checkpoint_trainable = torch.load(checkpoint_trainable_path, map_location=device)
        self.generator_trainable.load_state_dict(checkpoint_trainable["g_ema"], strict=True)

        self.generator_trainable.eval()

    def forward(self, latents, batch_size, truncation=1, randomize_noise=True):
        # 得到trainable image
        trainable_img = \
            self.generator_trainable(latents, input_is_latent=True, truncation=truncation,
                                     randomize_noise=randomize_noise)[0]

        return trainable_img
