import paddle
import paddle.nn as nn


class MlpBlock(nn.Layer):
    def __init__(self, features_dim, mlp_dim):
        super().__init__()
        self.fc_0 = nn.Linear(features_dim, mlp_dim)
        self.fc_1 = nn.Linear(mlp_dim, features_dim)

    def forward(self, x):
        y = self.fc_0(x)
        y = nn.functional.gelu(y)
        y = self.fc_1(y)
        return y


class MixerBlock(nn.Layer):
    def __init__(self, token_dim, channels_dim, tokens_mlp_dim, channels_mlp_dim):
        super().__init__()
        self.norm_0 = nn.LayerNorm(channels_dim)
        self.token_mixing = MlpBlock(token_dim, tokens_mlp_dim)
        self.norm_1 = nn.LayerNorm(channels_dim)
        self.channel_mixing = MlpBlock(channels_dim, channels_mlp_dim)

    def forward(self, x):
        y = self.norm_0(x)
        y = y.transpose((0, 2, 1))
        y = self.token_mixing(y)
        y = y.transpose((0, 2, 1))
        x = x + y
        y = self.norm_1(x)
        y = self.channel_mixing(y)
        x = x + y
        return x


class MlpMixer(nn.Layer):
    def __init__(self, img_size=(224, 224), patch_size=(16, 16), 
                num_classes=1000, num_blocks=12, hidden_dim=768, 
                tokens_mlp_dim=384, channels_mlp_dim=3072):
        super().__init__()
        self.stem = nn.Conv2D(
            3, hidden_dim, kernel_size=patch_size, stride=patch_size)
        self.blocks = nn.LayerList()
        for _ in range(num_blocks):
            block = MixerBlock(
                (img_size[0] // patch_size[0]) ** 2, hidden_dim, tokens_mlp_dim, channels_mlp_dim)
            self.blocks.append(block)
        self.pre_head_layer_norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, num_classes)

    def forward(self, inputs):
        x = self.stem(inputs)
        x = x.transpose((0, 2, 3, 1))
        x = x.flatten(1, 2)
        for block in self.blocks:
            x = block(x)
        x = x.mean(axis=1)
        x = self.head(x)
        return x


def mixer_b(pretrained=False, **kwargs):
    ''' 
    Model: MLP-mixer-base
    Params:
        pretrained: load the pretrained model
        img_size: input image size
        patch_size: patch size
        num_classes: number of classes
        num_blocks: number of MixerBlock
        hidden_dim: dim of hidden
        tokens_mlp_dim: dim of tokens_mlp
        channels_mlp_dim: dim of channels_mlp
    '''
    model = MlpMixer(
        hidden_dim=768, 
        num_blocks=12, 
        tokens_mlp_dim=384, 
        channels_mlp_dim=3072, 
        **kwargs
    )
    if pretrained:
        path = paddle.utils.download.get_weights_path_from_url('https://bj.bcebos.com/v1/ai-studio-online/8fcd0b6ba98042d68763bbcbfe96375cbfd97ffed8334ac09787ef73ecf9989f?responseContentDisposition=attachment%3B%20filename%3Dimagenet1k_Mixer-B_16.pdparams')
        model.set_dict(paddle.load(path))
    return model


def mixer_l(pretrained=False, **kwargs):
    ''' 
    Model: MLP-mixer-large
    Params:
        pretrained: load the pretrained model
        img_size: input image size
        patch_size: patch size
        num_classes: number of classes
        num_blocks: number of MixerBlock
        hidden_dim: dim of hidden
        tokens_mlp_dim: dim of tokens_mlp
        channels_mlp_dim: dim of channels_mlp
    '''
    model = MlpMixer(
        hidden_dim=1024, 
        num_blocks=24, 
        tokens_mlp_dim=512, 
        channels_mlp_dim=4096, 
        **kwargs
    )
    if pretrained:
        path = paddle.utils.download.get_weights_path_from_url('https://bj.bcebos.com/v1/ai-studio-online/ca74ababd4834e34b089c1485989738de4fdf6a97be645ed81b6e39449c5815c?responseContentDisposition=attachment%3B%20filename%3Dimagenet1k_Mixer-L_16.pdparams')
        model.set_dict(paddle.load(path))
    return model
 
