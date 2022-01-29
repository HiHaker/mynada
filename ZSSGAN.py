import torch
from sg2_model import Generator


class ZSSGAN(torch.nn.Module):
    def __init__(self, device, checkpoint_frozen_path, checkpoint_trainable_path):
        super(ZSSGAN, self).__init__()
        # gpu设备
        self.device = device

        # Set up frozen (source) generator
        self.generator_frozen = Generator(
            size=256, style_dim=512, n_mlp=8
        ).to(device)

        # load model
        checkpoint_frozen = torch.load(checkpoint_frozen_path, map_location=device)
        self.generator_frozen.load_state_dict(checkpoint_frozen["g_ema"], strict=True)

        # 调整为eval模式
        self.generator_frozen.eval()

        # Set up trainable (target) generator
        self.generator_trainable = Generator(
            size=256, style_dim=512, n_mlp=8
        ).to(device)

        # load model
        checkpoint_trainable = torch.load(checkpoint_trainable_path, map_location=device)
        self.generator_trainable.load_state_dict(checkpoint_trainable["g_ema"], strict=True)

        self.generator_trainable.eval()

    def forward(self, batch_size, truncation=1, randomize_noise=True):
        sample_z = torch.randn(batch_size, 512, device=self.device)

        w_styles = [self.generator.style(s) for s in [sample_z]]

        # 得到frozen image
        frozen_img = \
        self.generator_frozen(w_styles, input_is_latent=True, truncation=truncation, randomize_noise=randomize_noise)[0]

        # 得到trainable image
        trainable_img = \
        self.generator_trainable(w_styles, input_is_latent=True, truncation=truncation, randomize_noise=randomize_noise)[0]

        return frozen_img, trainable_img