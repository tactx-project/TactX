# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR model and criterion classes.
"""
import torch
from torch import nn
from torch.autograd import Variable
from tactile_policy.third_party.detr.models.backbone import build_backbone
from tactile_policy.third_party.detr.models.transformer import build_transformer, TransformerEncoder, TransformerEncoderLayer

import numpy as np
from sklearn.decomposition import PCA

def reparametrize(mu, logvar):
    std = logvar.div(2).exp()
    eps = Variable(std.data.new(std.size()).normal_())
    return mu + std * eps


def get_sinusoid_encoding_table(n_position, d_hid):
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    return torch.FloatTensor(sinusoid_table).unsqueeze(0)

class DETRVAE(nn.Module):
    """ This is the DETR module that performs object detection """
    def __init__(self, backbones, transformer, encoder, state_dim, tactile_dim, action_dim, num_queries, camera_names, tactile_names, image_feature_strategy, tactile_image_feature_strategy, use_language_conditioning, tactile_modality='image', tactile_encoding='linear', tactile_hidden_layers=None, tactile_image_encoding=None, pca_n_components=256, pca_mlp_hidden_layers=None):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
            tactile_modality: 'image', 'array', or 'state' - determines how tactile data is processed
            tactile_encoding: encoding strategy for raw tactile arrays ('linear', 'mlp', etc.)
            tactile_hidden_layers: list of hidden layer sizes for MLP encoding
            tactile_image_encoding: encoding strategy for tactile images ('resnet', 'pca', 'pca_mlp')
            pca_n_components: number of PCA components (or fraction of variance to keep)
            pca_mlp_hidden_layers: list of hidden layer sizes for MLP after PCA
        """
        super().__init__()
        self.num_queries = num_queries
        self.camera_names = camera_names
        self.tactile_names = tactile_names
        self.transformer = transformer
        self.encoder = encoder
        hidden_dim = transformer.d_model
        self.action_head = nn.Linear(hidden_dim, action_dim)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.tactile_dim = tactile_dim
        self.tactile_modality = tactile_modality
        self.tactile_encoding = tactile_encoding
        self.tactile_image_encoding = tactile_image_encoding if tactile_image_encoding else 'resnet'
        self.num_tactile = len(tactile_names) if tactile_names else 0
        self.pca_n_components = pca_n_components
        self.pca_mlp_hidden_layers = pca_mlp_hidden_layers

        self.use_language_conditioning = use_language_conditioning
        self.image_feature_strategy = image_feature_strategy

        assert self.image_feature_strategy in ['ACT_linear', 'linear', 'linear4', 'dpt']

        if self.image_feature_strategy == 'ACT_linear':
            self.input_proj = nn.Conv2d(backbones[0].num_channels, hidden_dim, kernel_size=1)
        elif self.image_feature_strategy == 'linear':
            # input proj goes up to 2x for cls token
            self.input_proj = nn.Conv2d(backbones[0].num_channels * 2, hidden_dim, kernel_size=1)
        elif self.image_feature_strategy == 'linear4':
            # input proj goes up to 2x for cls token
            self.input_proj = nn.Conv2d(backbones[0].num_channels * 2 * 4, hidden_dim, kernel_size=1)
        else:
            raise ValueError(f"Strategy {self.image_feature_strategy} not supported")
        
        self.backbones = nn.ModuleList(backbones)

        # Build tactile encoders based on modality
        if self.tactile_modality == 'image':
            # Image-based tactile: use CNN backbone + projection OR PCA-based encoding
            if self.tactile_image_encoding in ['pca', 'pca_mlp']:
                # PCA-based encoding for tactile images
                # PCA components will be fitted during data loading and stored per sensor
                self.tactile_pca_encoders = nn.ModuleDict()
                for tactile_name in tactile_names:
                    # Build encoder for each sensor
                    if self.tactile_image_encoding == 'pca':
                        # Direct PCA to hidden_dim
                        self.tactile_pca_encoders[tactile_name] = nn.Linear(self.pca_n_components, hidden_dim)
                    elif self.tactile_image_encoding == 'pca_mlp':
                        # PCA + MLP encoder
                        self.tactile_pca_encoders[tactile_name] = self._build_tactile_array_encoder(
                            self.pca_n_components, hidden_dim, 'mlp', self.pca_mlp_hidden_layers
                        )

                # PCA transformers will be set later after fitting on data
                self.tactile_pca_transformers = {}
                self.tactile_image_proj = None
                self.tactile_image_feature_strategy = None
            else:
                # ResNet-based encoding (original approach)
                self.tactile_image_feature_strategy = tactile_image_feature_strategy
                tactile_backbone_idx = 1 if len(backbones) > 1 else 0
                assert self.tactile_image_feature_strategy in ['ACT_linear', 'linear', 'linear4', 'dpt']

                if self.tactile_image_feature_strategy == 'ACT_linear':
                    self.tactile_image_proj = nn.Conv2d(backbones[tactile_backbone_idx].num_channels, hidden_dim, kernel_size=1)
                elif self.tactile_image_feature_strategy == 'linear':
                    self.tactile_image_proj = nn.Conv2d(backbones[tactile_backbone_idx].num_channels * 2, hidden_dim, kernel_size=1)
                elif self.tactile_image_feature_strategy == 'linear4':
                    self.tactile_image_proj = nn.Conv2d(backbones[tactile_backbone_idx].num_channels * 2 * 4, hidden_dim, kernel_size=1)
                else:
                    raise ValueError(f"Strategy {self.tactile_image_feature_strategy} not supported")

                self.tactile_pca_encoders = None
                self.tactile_pca_transformers = None

            self.tactile_array_encoder = None
            self.tactile_state_encoder = None

        elif self.tactile_modality == 'array':
            # Array-based tactile: process as separate tokens
            self.tactile_array_encoder = self._build_tactile_array_encoder(tactile_dim, hidden_dim, tactile_encoding, tactile_hidden_layers)
            self.tactile_image_proj = None
            self.tactile_state_encoder = None

        elif self.tactile_modality == 'state':
            # State-based tactile: concatenate with qpos
            # Encode tactile separately, then concat with encoded qpos
            self.tactile_state_encoder = self._build_tactile_array_encoder(self.num_tactile * tactile_dim, hidden_dim, tactile_encoding, tactile_hidden_layers)
            self.tactile_image_proj = None
            self.tactile_array_encoder = None
        else:
            # No tactile data
            self.tactile_image_proj = None
            self.tactile_array_encoder = None
            self.tactile_state_encoder = None

        if self.use_language_conditioning:
            self.input_proj_robot_state = nn.Linear(state_dim + 4096 * 2, hidden_dim)
        else:
            self.input_proj_robot_state = nn.Linear(state_dim, hidden_dim)

        # encoder extra parameters
        self.latent_dim = 32 # final size of latent z # TODO tune
        self.cls_embed = nn.Embedding(1, hidden_dim) # extra cls token embedding
        self.encoder_action_proj = nn.Linear(action_dim, hidden_dim) # project action to embedding
        self.encoder_joint_proj = nn.Linear(state_dim, hidden_dim)  # project qpos to embedding
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim*2) # project hidden state to latent std, var
        self.register_buffer('pos_table', get_sinusoid_encoding_table(1+1+num_queries, hidden_dim)) # [CLS], qpos, a_seq

        # decoder extra parameters
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim) # project latent sample to embedding
        self.additional_pos_embed = nn.Embedding(2, hidden_dim) # learned position embedding for proprio and latent

    def _build_tactile_array_encoder(self, input_dim, output_dim, encoding_strategy, hidden_layers):
        """Build encoder for raw tactile arrays based on encoding strategy."""
        if encoding_strategy == 'linear':
            return nn.Linear(input_dim, output_dim)
        elif encoding_strategy == 'mlp':
            if hidden_layers is None or len(hidden_layers) == 0:
                return nn.Linear(input_dim, output_dim)
            else:
                layers = []
                in_dim = input_dim
                for hidden_dim_layer in hidden_layers:
                    layers.append(nn.Linear(in_dim, hidden_dim_layer))
                    layers.append(nn.ReLU())
                    in_dim = hidden_dim_layer
                layers.append(nn.Linear(in_dim, output_dim))
                return nn.Sequential(*layers)
        else:
            raise ValueError(f"Tactile encoding strategy '{encoding_strategy}' not supported")

    def fit_pca_on_tactile_images(self, tactile_images_dict):
        if self.tactile_image_encoding not in ['pca', 'pca_mlp']:
            return

        for tactile_name in self.tactile_names:
            if tactile_name not in tactile_images_dict:
                continue

            tactile_images = tactile_images_dict[tactile_name]
            N, C, H, W = tactile_images.shape
            tactile_flat = tactile_images.reshape(N, -1)

            pca = PCA(n_components=self.pca_n_components)
            pca.fit(tactile_flat)
            self.tactile_pca_transformers[tactile_name] = pca

    def get_features_and_pos(self, image):
        """
        image (B, num_cam, C, H, W): image observation
        strategy (str): 'ACT_linear', 'linear', 'linear4', 'dpt'.
            - 'ACT_linear': linear projection of image features. Equivalent to implementation in original ACT.
            - 'linear': similar to ACT_linear but with CLS token concatenated along the channel dimension.
                        Same as linear in Dinov2 depth estimation.
            - 'linear4': Similar to linear, but uses 4 layers of features instead of 1.
            - 'dpt': similar to the DPT decoder
        """
        # Image observation features and position embeddings
        all_cam_features = []
        all_cam_pos = []
        B, num_cam, C, H, W = image.shape
        featuress, poss = self.backbones[0](image.reshape(B*num_cam, C, H, W))

        if self.image_feature_strategy in ['ACT_linear', 'linear', 'linear4']:
            featuress = featuress[-1]
            output_B, output_C, output_H, output_W = featuress.shape
            assert output_B == B*num_cam
            pos = poss[-1]
            featuress = featuress.view(B, num_cam, output_C, output_H, output_W) # take the last layer feature

            for cam_id, cam_name in enumerate(self.camera_names):
                features = featuress[:, cam_id]
                projected_features = self.input_proj(features)
                all_cam_features.append(projected_features)
                all_cam_pos.append(pos/2+ cam_id - 0.5)
        else:
            raise ValueError(f"Strategy {self.image_feature_strategy} not supported")
        
        # fold camera dimension into width dimension
        src = torch.cat(all_cam_features, axis=3)
        pos = torch.cat(all_cam_pos, axis=3)

        return src, pos
    
    def get_tactile_image_features_and_pos(self, tactile_image):
        """
        Process tactile images through tactile-specific backbone
        tactile_image (B, num_tactile, C, H, W): tactile sensor observations
        """
        if self.tactile_image_encoding in ['pca', 'pca_mlp']:
            # PCA-based encoding: flatten, transform, and encode per sensor
            B, num_tactile, C, H, W = tactile_image.shape
            tactile_features_list = []

            for tactile_id, tactile_name in enumerate(self.tactile_names):
                images = tactile_image[:, tactile_id]  # (B, C, H, W)
                images_flat = images.reshape(B, -1).cpu().numpy()  # (B, C*H*W)

                # Transform with PCA
                pca = self.tactile_pca_transformers[tactile_name]
                images_pca = pca.transform(images_flat)  # (B, n_components)
                images_pca = torch.from_numpy(images_pca).float().to(tactile_image.device)

                # Encode with linear or MLP
                features = self.tactile_pca_encoders[tactile_name](images_pca)  # (B, hidden_dim)
                tactile_features_list.append(features)

            tactile_features = torch.stack(tactile_features_list, dim=1)  # (B, num_tactile, hidden_dim)
            return tactile_features, None  # Return features directly, no spatial pos
        else:
            # ResNet-based encoding (original)
            all_tactile_features = []
            all_tactile_pos = []
            B, num_tactile, C, H, W = tactile_image.shape
            tactile_backbone = self.backbones[1] if len(self.backbones) > 1 else self.backbones[0]
            featuress, poss = tactile_backbone(tactile_image.reshape(B*num_tactile, C, H, W))

            if self.tactile_image_feature_strategy in ['ACT_linear', 'linear', 'linear4']:
                featuress = featuress[-1]
                output_B, output_C, output_H, output_W = featuress.shape
                assert output_B == B*num_tactile
                pos = poss[-1]
                featuress = featuress.view(B, num_tactile, output_C, output_H, output_W)

                for tactile_id, tactile_name in enumerate(self.tactile_names):
                    features = featuress[:, tactile_id]
                    projected_features = self.tactile_image_proj(features)
                    all_tactile_features.append(projected_features)
                    all_tactile_pos.append(pos/2 + len(self.camera_names) + tactile_id - 0.5)
            else:
                raise ValueError(f"Strategy {self.tactile_image_feature_strategy} not supported")

            src = torch.cat(all_tactile_features, axis=3)
            pos = torch.cat(all_tactile_pos, axis=3)
            return src, pos

    def get_tactile_array_features(self, tactile_array):
        """
        Process raw tactile arrays (B, num_tactile, tactile_dim)
        Returns features for each sensor to be used as spatial tokens
        """
        B, num_tactile, tactile_dim = tactile_array.shape
        # Encode each sensor separately: (B, num_tactile, tactile_dim) -> (B, num_tactile, hidden_dim)
        tactile_features = self.tactile_array_encoder(tactile_array)
        return tactile_features

    def forward(self, qpos, image, env_state, tactile_data=None, actions=None, is_pad=None, conditioning_dict=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        ### Obtain latent z from action sequence
        if is_training:
            # project action sequence to embedding dim, and concat with a CLS token
            action_embed = self.encoder_action_proj(actions) # (bs, seq, hidden_dim)
            qpos_embed = self.encoder_joint_proj(qpos)  # (bs, hidden_dim)
            qpos_embed = torch.unsqueeze(qpos_embed, axis=1)  # (bs, 1, hidden_dim)
            cls_embed = self.cls_embed.weight # (1, hidden_dim)
            cls_embed = torch.unsqueeze(cls_embed, axis=0).repeat(bs, 1, 1) # (bs, 1, hidden_dim)
            encoder_input = torch.cat([cls_embed, qpos_embed, action_embed], axis=1) # (bs, seq+1, hidden_dim)
            encoder_input = encoder_input.permute(1, 0, 2) # (seq+1, bs, hidden_dim)
            # do not mask cls token
            cls_joint_is_pad = torch.full((bs, 2), False).to(qpos.device) # False: not a padding
            is_pad = torch.cat([cls_joint_is_pad, is_pad], axis=1)  # (bs, seq+1)
            # obtain position embedding
            pos_embed = self.pos_table.clone().detach()
            pos_embed = pos_embed.permute(1, 0, 2)  # (seq+1, 1, hidden_dim)
            # query model
            encoder_output = self.encoder(encoder_input, pos=pos_embed, src_key_padding_mask=is_pad)
            encoder_output = encoder_output[0] # take cls output only
            latent_info = self.latent_proj(encoder_output)
            mu = latent_info[:, :self.latent_dim]
            logvar = latent_info[:, self.latent_dim:]
            latent_sample = reparametrize(mu, logvar)
            latent_input = self.latent_out_proj(latent_sample)
        else:
            mu = logvar = None
            latent_sample = torch.zeros([bs, self.latent_dim], dtype=torch.float32).to(qpos.device)
            latent_input = self.latent_out_proj(latent_sample)

        assert self.backbones is not None
        src_cam, pos_cam = self.get_features_and_pos(image)

        # Handle tactile based on modality
        if self.tactile_modality == 'image' and len(self.tactile_names) > 0:
            # tactile_data contains processed tactile images
            if tactile_data is not None and tactile_data.numel() > 0:
                src_tactile, pos_tactile = self.get_tactile_image_features_and_pos(tactile_data)
                if self.tactile_image_encoding in ['pca', 'pca_mlp']:
                    # PCA returns token features directly, not spatial features
                    tactile_features = src_tactile  # (B, num_tactile, hidden_dim)
                    src = src_cam
                    pos = pos_cam
                else:
                    # ResNet returns spatial features
                    src = torch.cat([src_cam, src_tactile], axis=3)
                    pos = torch.cat([pos_cam, pos_tactile], axis=3)
                    tactile_features = None
            else:
                src = src_cam
                pos = pos_cam
                tactile_features = None
        elif self.tactile_modality == 'array' and len(self.tactile_names) > 0:
            # tactile_data contains tactile arrays: (B, num_tactile, tactile_dim)
            if tactile_data is not None and tactile_data.numel() > 0:
                tactile_features = self.get_tactile_array_features(tactile_data)  # (B, num_tactile, hidden_dim)
                # Will pass directly to transformer as local variable
                src = src_cam
                pos = pos_cam
            else:
                tactile_features = None
                src = src_cam
                pos = pos_cam
        else:
            # For 'state', 'none', or no tactile names: use only camera features
            tactile_features = None
            src = src_cam
            pos = pos_cam

        # Proprioception features
        if self.tactile_modality == 'state' and len(self.tactile_names) > 0:
            # tactile_data contains tactile arrays: (B, num_tactile, tactile_dim)
            if tactile_data is not None and tactile_data.numel() > 0:
                B = tactile_data.shape[0]
                tactile_flat = tactile_data.reshape(B, -1)
                tactile_encoded = self.tactile_state_encoder(tactile_flat)
                qpos_encoded = self.input_proj_robot_state(qpos)
                proprio_input = qpos_encoded + tactile_encoded
            else:
                proprio_input = self.input_proj_robot_state(qpos)
        elif self.use_language_conditioning:
            language_embedding = conditioning_dict['language_embeddings'].flatten(start_dim=1)
            state_input = torch.cat([qpos, language_embedding], axis=1)
            proprio_input = self.input_proj_robot_state(state_input)
        else:
            proprio_input = self.input_proj_robot_state(qpos)

        hs = self.transformer(src, None, self.query_embed.weight, pos, latent_input, proprio_input, self.additional_pos_embed.weight, tactile_features=tactile_features)[0]

        a_hat = self.action_head(hs)
        is_pad_hat = self.is_pad_head(hs)
        return a_hat, is_pad_hat, [mu, logvar]

class CNNMLP(nn.Module):
    def __init__(self, backbones, state_dim, camera_names):
        """ Initializes the model.
        Parameters:
            backbones: torch module of the backbone to be used. See backbone.py
            transformer: torch module of the transformer architecture. See transformer.py
            state_dim: robot state dimension of the environment
            num_queries: number of object queries, ie detection slot. This is the maximal number of objects
                         DETR can detect in a single image. For COCO, we recommend 100 queries.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.camera_names = camera_names
        self.action_head = nn.Linear(1000, state_dim) # TODO add more
        if backbones is not None:
            self.backbones = nn.ModuleList(backbones)
            backbone_down_projs = []
            for backbone in backbones:
                down_proj = nn.Sequential(
                    nn.Conv2d(backbone.num_channels, 128, kernel_size=5),
                    nn.Conv2d(128, 64, kernel_size=5),
                    nn.Conv2d(64, 32, kernel_size=5)
                )
                backbone_down_projs.append(down_proj)
            self.backbone_down_projs = nn.ModuleList(backbone_down_projs)

            mlp_in_dim = 768 * len(backbones) + 14
            self.mlp = mlp(input_dim=mlp_in_dim, hidden_dim=1024, output_dim=14, hidden_depth=2)
        else:
            raise NotImplementedError

    def forward(self, qpos, image, env_state, actions=None):
        """
        qpos: batch, qpos_dim
        image: batch, num_cam, channel, height, width
        env_state: None
        actions: batch, seq, action_dim
        """
        is_training = actions is not None # train or val
        bs, _ = qpos.shape
        # Image observation features and position embeddings
        all_cam_features = []
        for cam_id, cam_name in enumerate(self.camera_names):
            features, pos = self.backbones[cam_id](image[:, cam_id])
            features = features[0] # take the last layer feature
            pos = pos[0] # not used
            all_cam_features.append(self.backbone_down_projs[cam_id](features))
        # flatten everything
        flattened_features = []
        for cam_feature in all_cam_features:
            flattened_features.append(cam_feature.reshape([bs, -1]))
        flattened_features = torch.cat(flattened_features, axis=1) # 768 each
        features = torch.cat([flattened_features, qpos], axis=1) # qpos: 14
        a_hat = self.mlp(features)
        return a_hat


def mlp(input_dim, hidden_dim, output_dim, hidden_depth):
    if hidden_depth == 0:
        mods = [nn.Linear(input_dim, output_dim)]
    else:
        mods = [nn.Linear(input_dim, hidden_dim), nn.ReLU(inplace=True)]
        for i in range(hidden_depth - 1):
            mods += [nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True)]
        mods.append(nn.Linear(hidden_dim, output_dim))
    trunk = nn.Sequential(*mods)
    return trunk


def build_encoder(args):
    d_model = args.hidden_dim # 256
    dropout = args.dropout # 0.1
    nhead = args.nheads # 8
    dim_feedforward = args.dim_feedforward # 2048
    num_encoder_layers = args.enc_layers # 4 # TODO shared with VAE decoder
    normalize_before = args.pre_norm # False
    activation = "relu"

    encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                            dropout, activation, normalize_before)
    encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
    encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

    return encoder


def build(args):
    state_dim = args.state_dim
    action_dim = args.action_dim

    #camera backbone
    backbones = []
    backbone = build_backbone(args)
    backbones.append(backbone)

    #optional tactile backbone if using tactile imgs
    tactile_image_encoding = getattr(args, 'tactile_image_encoding', 'resnet')
    if args.tactile_modality =='image' and tactile_image_encoding not in ['pca', 'pca_mlp']:
        import copy
        tactile_args = copy.copy(args)
        tactile_args.backbone = args.tactile_backbone
        tactile_args.lr_backbone = args.lr_tactile_backbone
        tactile_args.image_feature_strategy = args.tactile_image_feature_strategy
        tactile_backbone = build_backbone(tactile_args)
        backbones.append(tactile_backbone)

    transformer = build_transformer(args)

    encoder = build_encoder(args)

    model = DETRVAE(
        backbones,
        transformer,
        encoder,
        state_dim=state_dim,
        action_dim=action_dim,
        tactile_dim=args.tactile_dim,
        num_queries=args.num_queries,
        camera_names=args.camera_names,
        tactile_names=args.tactile_names,
        image_feature_strategy=args.image_feature_strategy,
        tactile_image_feature_strategy=getattr(args, 'tactile_image_feature_strategy', 'ACT_linear'),
        use_language_conditioning=args.use_language_conditioning,
        tactile_modality=args.tactile_modality,
        tactile_encoding=args.tactile_encoding,
        tactile_hidden_layers=args.tactile_hidden_layers,
        tactile_image_encoding=tactile_image_encoding,
        pca_n_components=getattr(args, 'pca_n_components', 256),
        pca_mlp_hidden_layers=getattr(args, 'pca_mlp_hidden_layers', None),
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

def build_cnnmlp(args):
    state_dim = 14 # TODO hardcode

    # From state
    # backbone = None # from state for now, no need for conv nets
    # From image
    backbones = []
    for _ in args.camera_names:
        backbone = build_backbone(args)
        backbones.append(backbone)

    model = CNNMLP(
        backbones,
        state_dim=state_dim,
        camera_names=args.camera_names,
    )

    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("number of parameters: %.2fM" % (n_parameters/1e6,))

    return model

