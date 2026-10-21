from joint_embedding_learning.cross_modal.model import (
    CrossModalVAE3, CrossModalCLR3,
    MagneticEncoder, FlexiTacEncoder,
    FlexiTacEncoderResidual, FlexiTacDecoderResidual,
)
from simclr.modules import get_resnet  # ResNet backbone for the Daimon encoder



def get_model(model_name, model_details, pretrained=False):
    if model_name == 'cross_modal_vae3':
        image_encoder = get_resnet(model_details['resnet'], pretrained=pretrained)
        n_features = image_encoder.fc.in_features

        magnetic_encoder = MagneticEncoder(
            input_dim=model_details['magnetic_input_dim'],
            hidden_dims=model_details['magnetic_hidden_dims'],
            output_dim=n_features,
        )

        flexi_h = model_details.get('flexitac_in_h', 12)
        flexi_w = model_details.get('flexitac_in_w', 16)

        # FlexiTac encoder/decoder architecture switch.
        # 'small_cnn' (default): legacy 2-conv FlexiTacEncoder + 2-convT
        #                        FlexiTacDecoder — preserves state_dict
        #                        compatibility with all prior checkpoints.
        # 'residual'           : deeper residual CNN (GN+GELU, ~3 stages,
        #                        2 downsamples). Adds capacity for FlexiTac,
        #                        which underperforms vs Daimon/eFlesh in
        #                        recon and contact-score regression. Requires
        #                        re-training (state_dict shape differs).
        flexi_arch = model_details.get('flexitac_arch', 'small_cnn')
        if flexi_arch == 'residual':
            flexi_enc_block_channels = model_details.get(
                'flexitac_encoder_block_channels', [64, 128, 256])
            flexi_enc_blocks_per_stage = model_details.get(
                'flexitac_encoder_blocks_per_stage', [2, 2, 2])
            flexi_dec_block_channels = model_details.get(
                'flexitac_decoder_block_channels', [256, 128, 64])
            flexi_dec_blocks_per_stage = model_details.get(
                'flexitac_decoder_blocks_per_stage', [2, 2, 2])
            flexi_groups = int(model_details.get('flexitac_gn_groups', 8))
            flexitac_encoder = FlexiTacEncoderResidual(
                in_h=flexi_h, in_w=flexi_w,
                block_channels=tuple(flexi_enc_block_channels),
                blocks_per_stage=tuple(flexi_enc_blocks_per_stage),
                output_dim=n_features,
                groups=flexi_groups,
            )
            flexitac_decoder = FlexiTacDecoderResidual(
                model_details['projection_dim'],
                block_channels=tuple(flexi_dec_block_channels),
                blocks_per_stage=tuple(flexi_dec_blocks_per_stage),
                out_h=flexi_h, out_w=flexi_w,
                out_channels=model_details.get('flexitac_out_channels', 1),
                groups=flexi_groups,
            )
        else:
            # === Legacy small_cnn FlexiTac encoder/decoder (default) ===
            # Kept verbatim for backward compat with existing cross_modal_vae3
            # checkpoints. Do not delete.
            flexi_enc_channels = model_details.get('flexitac_encoder_channels', [32, 64])
            # Pool to half resolution by default for the spatial bottleneck.
            flexi_pool_h = model_details.get('flexitac_pool_h', max(1, flexi_h // 2))
            flexi_pool_w = model_details.get('flexitac_pool_w', max(1, flexi_w // 2))
            flexitac_encoder = FlexiTacEncoder(
                in_h=flexi_h, in_w=flexi_w,
                channels=tuple(flexi_enc_channels),
                pool_h=flexi_pool_h, pool_w=flexi_pool_w,
                output_dim=n_features,
            )
            # None -> CrossModalVAE3 builds the legacy FlexiTacDecoder itself.
            flexitac_decoder = None

        model = CrossModalVAE3(
            image_encoder, magnetic_encoder, flexitac_encoder,
            projection_dim=model_details['projection_dim'],
            n_features=n_features,
            image_out_channels=model_details.get('image_out_channels', 3),
            image_out_size=model_details.get('image_out_size', 224),
            image_decoder_channels=model_details.get('image_decoder_channels', [256, 128, 64, 32]),
            magnetic_decoder_hidden=model_details.get('magnetic_decoder_hidden', [256, 128, 64]),
            magnetic_out_dim=model_details.get('magnetic_input_dim', 15),
            flexitac_in_h=flexi_h, flexitac_in_w=flexi_w,
            flexitac_decoder_channels=model_details.get('flexitac_decoder_channels', [64, 32]),
            flexitac_out_channels=model_details.get('flexitac_out_channels', 1),
            flexitac_decoder=flexitac_decoder,
        )
        return model

    elif model_name == 'cross_modal_clr3':
        image_encoder = get_resnet(model_details['resnet'], pretrained=pretrained)
        n_features = image_encoder.fc.in_features

        magnetic_encoder = MagneticEncoder(
            input_dim=model_details['magnetic_input_dim'],
            hidden_dims=model_details['magnetic_hidden_dims'],
            output_dim=n_features,
        )

        flexi_h = model_details.get('flexitac_in_h', 12)
        flexi_w = model_details.get('flexitac_in_w', 16)

        # Same architecture switch as cross_modal_vae3. See that branch for
        # rationale. CLR3 has no decoder so only the encoder is swapped.
        flexi_arch = model_details.get('flexitac_arch', 'small_cnn')
        if flexi_arch == 'residual':
            flexi_enc_block_channels = model_details.get(
                'flexitac_encoder_block_channels', [64, 128, 256])
            flexi_enc_blocks_per_stage = model_details.get(
                'flexitac_encoder_blocks_per_stage', [2, 2, 2])
            flexi_groups = int(model_details.get('flexitac_gn_groups', 8))
            flexitac_encoder = FlexiTacEncoderResidual(
                in_h=flexi_h, in_w=flexi_w,
                block_channels=tuple(flexi_enc_block_channels),
                blocks_per_stage=tuple(flexi_enc_blocks_per_stage),
                output_dim=n_features,
                groups=flexi_groups,
            )
        else:
            # === Legacy small_cnn FlexiTacEncoder (default) ===
            # Kept for backward compat with existing cross_modal_clr3
            # checkpoints. Do not delete.
            flexi_enc_channels = model_details.get('flexitac_encoder_channels', [32, 64])
            flexi_pool_h = model_details.get('flexitac_pool_h', max(1, flexi_h // 2))
            flexi_pool_w = model_details.get('flexitac_pool_w', max(1, flexi_w // 2))
            flexitac_encoder = FlexiTacEncoder(
                in_h=flexi_h, in_w=flexi_w,
                channels=tuple(flexi_enc_channels),
                pool_h=flexi_pool_h, pool_w=flexi_pool_w,
                output_dim=n_features,
            )

        model = CrossModalCLR3(
            image_encoder, magnetic_encoder, flexitac_encoder,
            projection_dim=model_details['projection_dim'],
            n_features=n_features,
        )
        return model

    else:
        raise ValueError(f"Model {model_name} not found")
