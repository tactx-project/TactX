import torch
import torch.nn as nn
import torch.nn.functional as F

class MagneticEncoder(nn.Module):
    """
    MLP encoder for magnetic sensor data (eFlesh).
    Maps low-dimensional magnetic readings to a high-dimensional representation space.
    
    eFlesh sensor: 5 magnets × 3 axes = 15D input
    Output dimension matches the image encoder (e.g., 512 for ResNet-18, 2048 for ResNet-50)
    """

    def __init__(self, input_dim=15, hidden_dims=[64, 128, 256], output_dim=512):
        super(MagneticEncoder, self).__init__()

        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))

        self.encoder = nn.Sequential(*layers)

    def forward(self, x):
        """
        Args:
            x: (B, input_dim) magnetic sensor readings
        Returns:
            (B, output_dim) representation vector
        """
        return self.encoder(x)

class ImageDecoder(nn.Module):
    """
    Deconvolutional decoder reconstructing a Daimon-style image from the
    VAE latent z (B, latent_dim).

    The spatial side is recovered by a stack of ConvTranspose2d blocks,
    each doubling the spatial resolution. A linear layer first maps the
    low-dim latent to a (channels[0], start_h, start_w) feature map.

    Args:
        in_dim: input latent dimension (e.g., projection_dim=16).
        channels: list of channel widths for the transposed-conv stack
                  (e.g., [256, 128, 64, 32] for 4 upsampling stages).
        out_channels: number of output image channels (3 for RGB / repeated depth).
        out_size: target square output side (e.g., 224).
    """

    def __init__(self, in_dim, channels=(256, 128, 64, 32), out_channels=3, out_size=224):
        super().__init__()
        assert len(channels) >= 1, "ImageDecoder.channels must have at least one entry"
        num_upsample = len(channels)
        assert out_size % (2 ** num_upsample) == 0, (
            f"out_size={out_size} must be divisible by 2**{num_upsample}"
        )
        start = out_size // (2 ** num_upsample)
        self.start_c = channels[0]
        self.start_h = start
        self.start_w = start

        self.fc = nn.Linear(in_dim, channels[0] * start * start)

        layers = []
        for i in range(num_upsample):
            in_c = channels[i]
            if i == num_upsample - 1:
                layers.append(nn.ConvTranspose2d(in_c, out_channels, kernel_size=4, stride=2, padding=1))
            else:
                out_c = channels[i + 1]
                layers.append(nn.ConvTranspose2d(in_c, out_c, kernel_size=4, stride=2, padding=1))
                layers.append(nn.BatchNorm2d(out_c))
                layers.append(nn.ReLU(inplace=True))
        self.deconv = nn.Sequential(*layers)

    def forward(self, z):
        """z: (B, in_dim) -> x_hat: (B, out_channels, out_size, out_size)."""
        B = z.shape[0]
        x = self.fc(z).view(B, self.start_c, self.start_h, self.start_w)
        return self.deconv(x)

class MagneticDecoder(nn.Module):
    """
    MLP decoder reconstructing a magnetic (eFlesh) vector from the
    VAE latent z (B, latent_dim).

    Args:
        in_dim: input latent dimension (e.g., projection_dim=16).
        hidden_dims: MLP hidden widths.
        out_dim: output vector dimension (typically 15 for eFlesh).
    """

    def __init__(self, in_dim, hidden_dims=(256, 128, 64), out_dim=15):
        super().__init__()
        layers = []
        prev = in_dim
        for hd in hidden_dims:
            layers.append(nn.Linear(prev, hd))
            layers.append(nn.BatchNorm1d(hd))
            layers.append(nn.ReLU(inplace=True))
            prev = hd
        layers.append(nn.Linear(prev, out_dim))
        self.decoder = nn.Sequential(*layers)

    def forward(self, z):
        return self.decoder(z)

class FlexiTacEncoder(nn.Module):
    """
    Small CNN encoder for FlexiTac sensor (12 x 16 grid).

    Input layout:
        x: (B, 12, 16) raw, OR (B, 1, 12, 16). Internally normalized to (B, 1, 12, 16).

    Architecture (default, channels=[32, 64]):
        Conv2d(1 -> 32, k3, p1) -> BN -> ReLU
        Conv2d(32 -> 64, k3, p1) -> BN -> ReLU
        AdaptiveAvgPool2d((6, 8))
        Flatten -> Linear(64*6*8 -> output_dim)

    output_dim is set to match the other encoders (e.g., 512 for ResNet-18).
    """

    def __init__(self, in_h=12, in_w=16, channels=(32, 64), pool_h=6, pool_w=8,
                 output_dim=512):
        super().__init__()
        layers = []
        prev_c = 1
        for c in channels:
            layers.append(nn.Conv2d(prev_c, c, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(c))
            layers.append(nn.ReLU(inplace=True))
            prev_c = c
        layers.append(nn.AdaptiveAvgPool2d((pool_h, pool_w)))
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Linear(prev_c * pool_h * pool_w, output_dim)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)  # (B, 1, H, W)
        h = self.conv(x)
        h = h.flatten(1)
        return self.fc(h)

class _GnReluBlock(nn.Module):
    """Pre-activation residual block (GroupNorm + GELU + 3x3 conv x2 + skip).

    Used by FlexiTacEncoderResidual / FlexiTacDecoderResidual. GroupNorm is
    chosen over BatchNorm because tactile training batches can be small and
    are sometimes batched per-pair (chunked sampler), so per-batch statistics
    are unreliable. Pre-activation (norm→act→conv) gives cleaner gradient
    flow than the post-activation residual variant.
    """

    def __init__(self, in_c, out_c, stride=1, groups=8):
        super().__init__()
        g_in = min(groups, in_c)
        g_out = min(groups, out_c)
        self.norm1 = nn.GroupNorm(g_in, in_c)
        self.act1 = nn.GELU()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.norm2 = nn.GroupNorm(g_out, out_c)
        self.act2 = nn.GELU()
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1,
                               padding=1, bias=False)
        if stride != 1 or in_c != out_c:
            self.skip = nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride,
                                  bias=False)
        else:
            self.skip = nn.Identity()

    def forward(self, x):
        y = self.act1(self.norm1(x))
        y = self.conv1(y)
        y = self.act2(self.norm2(y))
        y = self.conv2(y)
        return y + self.skip(x)

class FlexiTacEncoderResidual(nn.Module):
    """
    Deeper residual CNN encoder for FlexiTac (12 x 16 grid).

    ResNet's 7x7 stride-2 stem + 3x3 maxpool would collapse a 12x16 grid to
    nothing, so we use a 3x3 stride-1 stem and a stack of pre-activation
    residual blocks (GroupNorm + GELU) with two stride-2 downsamples. Roughly
    one third more parameters than the small_cnn FlexiTacEncoder but much more
    representational capacity per parameter via residual depth.

    Input layout:
        x: (B, 12, 16) raw, OR (B, 1, 12, 16). Normalized to (B, 1, 12, 16).

    Architecture (default block_channels=[64, 128, 256], blocks_per_stage=[2, 2, 2]):
        stem:    Conv(1 → 64, k3 s1 p1)                          (B, 64, 12, 16)
        stage 1: 2x ResBlock(64 → 64)                            (B, 64, 12, 16)
        stage 2: ResBlock(64 → 128, s=2) + ResBlock(128 → 128)   (B, 128, 6, 8)
        stage 3: ResBlock(128 → 256, s=2) + ResBlock(256 → 256)  (B, 256, 3, 4)
        head:    GN → GELU → AdaptiveAvgPool(1) → Linear(256 → output_dim)
    """

    def __init__(self, in_h=12, in_w=16, block_channels=(64, 128, 256),
                 blocks_per_stage=(2, 2, 2), output_dim=512, groups=8):
        super().__init__()
        assert len(block_channels) == len(blocks_per_stage), \
            "block_channels and blocks_per_stage must have same length"
        c0 = block_channels[0]
        self.stem = nn.Conv2d(1, c0, kernel_size=3, stride=1, padding=1, bias=False)

        stages = []
        in_c = c0
        for stage_idx, (c_out, n_blocks) in enumerate(
                zip(block_channels, blocks_per_stage)):
            for b in range(n_blocks):
                stride = 2 if (stage_idx > 0 and b == 0) else 1
                stages.append(_GnReluBlock(in_c, c_out, stride=stride, groups=groups))
                in_c = c_out
        self.body = nn.Sequential(*stages)

        self.norm = nn.GroupNorm(min(groups, in_c), in_c)
        self.act = nn.GELU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_c, output_dim)

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
        h = self.stem(x)
        h = self.body(h)
        h = self.act(self.norm(h))
        h = self.pool(h).flatten(1)
        return self.fc(h)

class FlexiTacDecoderResidual(nn.Module):
    """
    Deeper residual decoder, mirror of FlexiTacEncoderResidual.

    Architecture (default block_channels=[256, 128, 64], blocks_per_stage=[2, 2, 2]):
        Linear(z → 256 * 3 * 4) → reshape (B, 256, 3, 4)
        stage 1: 2x ResBlock(256 → 256)                          (B, 256, 3, 4)
        stage 2: ConvT(256 → 128, k4 s2 p1) + 2x ResBlock(128)   (B, 128, 6, 8)
        stage 3: ConvT(128 → 64,  k4 s2 p1) + 2x ResBlock(64)    (B, 64, 12, 16)
        head:    GN → GELU → Conv(64 → out_channels, k3 s1 p1)   (B, out, 12, 16)
    """

    def __init__(self, in_dim, block_channels=(256, 128, 64),
                 blocks_per_stage=(2, 2, 2), out_h=12, out_w=16,
                 out_channels=1, groups=8):
        super().__init__()
        assert len(block_channels) == len(blocks_per_stage), \
            "block_channels and blocks_per_stage must have same length"
        n_up = len(block_channels) - 1
        assert out_h % (2 ** n_up) == 0 and out_w % (2 ** n_up) == 0, (
            f"out_h={out_h}, out_w={out_w} must be divisible by 2**{n_up}"
        )
        self.start_h = out_h // (2 ** n_up)
        self.start_w = out_w // (2 ** n_up)
        self.start_c = block_channels[0]
        self.fc = nn.Linear(in_dim,
                            block_channels[0] * self.start_h * self.start_w)

        layers = []
        for stage_idx, (c_curr, n_blocks) in enumerate(
                zip(block_channels, blocks_per_stage)):
            if stage_idx > 0:
                c_prev = block_channels[stage_idx - 1]
                layers.append(nn.GroupNorm(min(groups, c_prev), c_prev))
                layers.append(nn.GELU())
                layers.append(nn.ConvTranspose2d(c_prev, c_curr, kernel_size=4,
                                                 stride=2, padding=1, bias=False))
            for _ in range(n_blocks):
                layers.append(_GnReluBlock(c_curr, c_curr, stride=1, groups=groups))
        c_last = block_channels[-1]
        layers.append(nn.GroupNorm(min(groups, c_last), c_last))
        layers.append(nn.GELU())
        layers.append(nn.Conv2d(c_last, out_channels, kernel_size=3,
                                stride=1, padding=1))
        self.body = nn.Sequential(*layers)

    def forward(self, z):
        B = z.shape[0]
        x = self.fc(z).view(B, self.start_c, self.start_h, self.start_w)
        return self.body(x)

class FlexiTacDecoder(nn.Module):
    """
    Small ConvTranspose2d decoder reconstructing a FlexiTac (12 x 16) grid
    from the VAE latent z (B, latent_dim).

    Architecture (default, channels=[64, 32]):
        Linear(z -> 64 * 3 * 4) -> reshape (B, 64, 3, 4)
        ConvT(64 -> 32, k4 s2 p1) -> BN -> ReLU              (3,4) -> (6,8)
        ConvT(32 -> 1,  k4 s2 p1)                            (6,8) -> (12,16)

    Output shape: (B, 1, 12, 16) by default. For loss against raw (B, 12, 16),
    callers can squeeze channel dim, but the loss helper handles both.
    """

    def __init__(self, in_dim, channels=(64, 32), out_h=12, out_w=16,
                 out_channels=1):
        super().__init__()
        assert out_h % (2 ** len(channels)) == 0 and out_w % (2 ** len(channels)) == 0, (
            f"out_h={out_h}, out_w={out_w} must be divisible by 2**{len(channels)}"
        )
        self.start_h = out_h // (2 ** len(channels))
        self.start_w = out_w // (2 ** len(channels))
        self.start_c = channels[0]
        self.fc = nn.Linear(in_dim, channels[0] * self.start_h * self.start_w)

        layers = []
        for i in range(len(channels)):
            in_c = channels[i]
            if i == len(channels) - 1:
                layers.append(nn.ConvTranspose2d(in_c, out_channels, kernel_size=4,
                                                 stride=2, padding=1))
            else:
                out_c = channels[i + 1]
                layers.append(nn.ConvTranspose2d(in_c, out_c, kernel_size=4,
                                                 stride=2, padding=1))
                layers.append(nn.BatchNorm2d(out_c))
                layers.append(nn.ReLU(inplace=True))
        self.deconv = nn.Sequential(*layers)

    def forward(self, z):
        B = z.shape[0]
        x = self.fc(z).view(B, self.start_c, self.start_h, self.start_w)
        return self.deconv(x)  # (B, out_channels, out_h, out_w)

class CrossModalVAE3(nn.Module):
    """
    Tri-modal variational cross-modal model (Daimon + eFlesh + FlexiTac) trained
    on pair-only data: ``de_pair`` (Daimon, eFlesh), ``ef_pair`` (eFlesh,
    FlexiTac), ``fd_pair`` (FlexiTac, Daimon).

    Three modalities share a single latent space regularized to the same
    standard-normal prior ``p(z) = N(0, I)``. Each pair sees only 2 of 3
    modalities at a time, but the shared encoders, shared projection-head form,
    and shared prior pull the three marginal posteriors into the same code,
    with direct supervision on all three pairs and a shared representation.

    Encoders:
        Daimon   x_D (B, 3, 224, 224) -> ResNet           -> h_D (B, n_features)
        eFlesh   x_E (B, 15)          -> MagneticEncoder  -> h_E (B, n_features)
        FlexiTac x_F (B, 12, 16)      -> FlexiTacEncoder  -> h_F (B, n_features)

    Projection heads (one per modality, output 2 * projection_dim for
    (mu, logvar)):
        proj_D, proj_E, proj_F

    Decoders (consume z, NOT h):
        Daimon   D_img: z -> (B, 3, 224, 224)
        eFlesh   D_mag: z -> (B, 15)
        FlexiTac D_flx: z -> (B, 1, 12, 16)

    Forward interface: ``forward_pair(x_a, x_b, name)`` where ``name`` is one of
    ``{'de', 'ef', 'fd'}``. Returns a generic dict with ``a/b``-suffixed keys so
    the loss can iterate over the three pairs uniformly:
        h_a, h_b, mu_a, logvar_a, mu_b, logvar_b, z_a, z_b,
        recon_aa, recon_ba, recon_bb, recon_ab,
        sensor_a, sensor_b
    """

    _PAIR_TABLE = {
        "de": ("D", "E"),
        "ef": ("E", "F"),
        "fd": ("F", "D"),
    }

    def __init__(self, image_encoder, magnetic_encoder, flexitac_encoder,
                 projection_dim, n_features,
                 image_out_channels=3, image_out_size=224,
                 image_decoder_channels=(256, 128, 64, 32),
                 magnetic_decoder_hidden=(256, 128, 64),
                 magnetic_out_dim=15,
                 flexitac_in_h=12, flexitac_in_w=16,
                 flexitac_decoder_channels=(64, 32),
                 flexitac_out_channels=1,
                 flexitac_decoder=None):
        """
        flexitac_decoder: optional pre-built decoder module (e.g.
            FlexiTacDecoderResidual). If None, the legacy small ConvTranspose
            FlexiTacDecoder is built from `flexitac_decoder_channels` etc.
            Pre-existing checkpoints stay loadable: their saved state_dict
            still matches the default-constructed FlexiTacDecoder.
        """
        super().__init__()
        self.n_features = n_features
        self.projection_dim = projection_dim

        self.image_encoder = image_encoder
        self.image_encoder.fc = nn.Identity()
        self.magnetic_encoder = magnetic_encoder
        self.flexitac_encoder = flexitac_encoder

        # Per-modality projection heads (each produces (mu, log_var)).
        self.proj_D = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            nn.ReLU(),
            nn.Linear(n_features, 2 * projection_dim, bias=False),
        )
        self.proj_E = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            nn.ReLU(),
            nn.Linear(n_features, 2 * projection_dim, bias=False),
        )
        self.proj_F = nn.Sequential(
            nn.Linear(n_features, n_features, bias=False),
            nn.ReLU(),
            nn.Linear(n_features, 2 * projection_dim, bias=False),
        )

        # Decoders consume z (the variational bottleneck).
        self.dec_D = ImageDecoder(
            projection_dim,
            channels=tuple(image_decoder_channels),
            out_channels=image_out_channels,
            out_size=image_out_size,
        )
        self.dec_E = MagneticDecoder(
            projection_dim,
            hidden_dims=tuple(magnetic_decoder_hidden),
            out_dim=magnetic_out_dim,
        )
        if flexitac_decoder is not None:
            # External decoder injected (e.g. FlexiTacDecoderResidual built in
            # models_setup.py based on flexitac_arch='residual'). The
            # flexitac_decoder_channels argument is ignored in this branch —
            # the injected module owns its own architecture knobs.
            self.dec_F = flexitac_decoder
        else:
            # === Legacy small ConvTranspose FlexiTacDecoder (default) ===
            # Kept as the default to preserve state_dict compatibility with
            # all existing cross_modal_vae3 checkpoints. Do not remove.
            self.dec_F = FlexiTacDecoder(
                projection_dim,
                channels=tuple(flexitac_decoder_channels),
                out_h=flexitac_in_h,
                out_w=flexitac_in_w,
                out_channels=flexitac_out_channels,
            )

    @staticmethod
    def reparameterize(mu, logvar, training=True):
        if training:
            std = (0.5 * logvar).exp()
            eps = torch.randn_like(std)
            return mu + std * eps
        return mu

    def _encoder_for(self, sensor):
        return {
            "D": self.image_encoder,
            "E": self.magnetic_encoder,
            "F": self.flexitac_encoder,
        }[sensor]

    def _projector_for(self, sensor):
        return {"D": self.proj_D, "E": self.proj_E, "F": self.proj_F}[sensor]

    def _decoder_for(self, sensor):
        return {"D": self.dec_D, "E": self.dec_E, "F": self.dec_F}[sensor]

    def _enc(self, x, sensor):
        h = self._encoder_for(sensor)(x)
        proj = self._projector_for(sensor)(h)
        mu, logvar = proj.chunk(2, dim=-1)
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        z = self.reparameterize(mu, logvar, training=self.training)
        return h, mu, logvar, z

    def forward_pair(self, x_a, x_b, name):
        """
        Run a pair (a, b) through their respective encoders, sample z, and
        produce a 4-branch reconstruction set decoded from z.

        Args:
            x_a: input tensor for sensor_a in the pair.
            x_b: input tensor for sensor_b in the pair.
            name: one of {'de', 'ef', 'fd'}.

        Returns dict with:
            h_a, h_b               : (B, n_features)
            mu_a, logvar_a, z_a    : (B, projection_dim)
            mu_b, logvar_b, z_b    : (B, projection_dim)
            recon_aa, recon_ba     : decoded by D_a from (z_a, z_b) -> ~ x_a
            recon_bb, recon_ab     : decoded by D_b from (z_b, z_a) -> ~ x_b
            sensor_a, sensor_b     : 'D' / 'E' / 'F'
            pair                   : name
        """
        if name not in self._PAIR_TABLE:
            raise ValueError(
                f"Unknown pair '{name}', expected one of {list(self._PAIR_TABLE)}"
            )
        sa, sb = self._PAIR_TABLE[name]

        h_a, mu_a, logvar_a, z_a = self._enc(x_a, sa)
        h_b, mu_b, logvar_b, z_b = self._enc(x_b, sb)

        dec_a = self._decoder_for(sa)
        dec_b = self._decoder_for(sb)

        recon_aa = dec_a(z_a)  # self-recon (a-side)
        recon_ba = dec_a(z_b)  # cross: b's z decodes to a
        recon_bb = dec_b(z_b)  # self-recon (b-side)
        recon_ab = dec_b(z_a)  # cross: a's z decodes to b

        return {
            "pair": name, "sensor_a": sa, "sensor_b": sb,
            "h_a": h_a, "h_b": h_b,
            "mu_a": mu_a, "logvar_a": logvar_a, "z_a": z_a,
            "mu_b": mu_b, "logvar_b": logvar_b, "z_b": z_b,
            "recon_aa": recon_aa, "recon_ba": recon_ba,
            "recon_bb": recon_bb, "recon_ab": recon_ab,
        }

    def forward_self(self, x, sensor):
        """
        Single-modality forward used by Stage 1 dynamic self-recon pretraining.

        Args:
            x: input tensor for the given sensor.
            sensor: one of {'D', 'E', 'F'}.

        Returns dict with: h, mu, logvar, z, recon, sensor.
        """
        if sensor not in ("D", "E", "F"):
            raise ValueError(f"Unknown sensor '{sensor}', expected one of D/E/F")
        h, mu, logvar, z = self._enc(x, sensor)
        recon = self._decoder_for(sensor)(z)
        return {
            "sensor": sensor,
            "h": h, "mu": mu, "logvar": logvar, "z": z,
            "recon": recon,
        }

    def forward(self, batch):
        """
        Joint forward over all three pairs.

        Args:
            batch: dict with keys 'de', 'ef', 'fd', each mapping to (x_a, x_b).
        Returns:
            dict { 'de': out_de, 'ef': out_ef, 'fd': out_fd } where each value
            is a per-pair dict from forward_pair.
        """
        out = {}
        for name, (x_a, x_b) in batch.items():
            out[name] = self.forward_pair(x_a, x_b, name)
        return out

class CrossModalCLR3(nn.Module):
    """
    Tri-modal *plain contrastive* model (Daimon + eFlesh + FlexiTac).

    The non-variational, reconstruction-free counterpart of CrossModalVAE3:
    three sensor encoders feed three projection heads that map into ONE shared
    latent space, where pair-wise NT-Xent contrastive loss is applied over the
    de / ef / fd pairs. No (mu, logvar), no reparameterization, no decoders,
    no KL — just the three encoders, three projection heads, and contrastive
    alignment in the common latent space.

    Encoders (identical to CrossModalVAE3):
        Daimon   x_D (B, 3, 224, 224) -> ResNet           -> h_D (B, n_features)
        eFlesh   x_E (B, 15)          -> MagneticEncoder  -> h_E (B, n_features)
        FlexiTac x_F (B, 12, 16)      -> FlexiTacEncoder  -> h_F (B, n_features)

    Projection heads (one per modality, output projection_dim directly):
        g(h) = W2 * ReLU(W1 * h)

    forward_pair(x_a, x_b, name) returns a dict whose key names mirror
    CrossModalVAE3.forward_pair (``z_a``/``z_b`` plus ``mu_a``/``mu_b`` aliases
    that point at the same projections) so the existing train_tri /
    tri_train_step loop and its metric helpers consume it unchanged.
    """

    _PAIR_TABLE = {
        "de": ("D", "E"),
        "ef": ("E", "F"),
        "fd": ("F", "D"),
    }

    def __init__(self, image_encoder, magnetic_encoder, flexitac_encoder,
                 projection_dim, n_features):
        super().__init__()
        self.n_features = n_features
        self.projection_dim = projection_dim

        self.image_encoder = image_encoder
        self.image_encoder.fc = nn.Identity()
        self.magnetic_encoder = magnetic_encoder
        self.flexitac_encoder = flexitac_encoder

        def _make_projector():
            # g(h) = W2 * ReLU(W1 * h) -> projection_dim (no (mu, logvar) split).
            return nn.Sequential(
                nn.Linear(n_features, n_features, bias=False),
                nn.ReLU(),
                nn.Linear(n_features, projection_dim, bias=False),
            )

        self.proj_D = _make_projector()
        self.proj_E = _make_projector()
        self.proj_F = _make_projector()

    def _encoder_for(self, sensor):
        return {
            "D": self.image_encoder,
            "E": self.magnetic_encoder,
            "F": self.flexitac_encoder,
        }[sensor]

    def _projector_for(self, sensor):
        return {"D": self.proj_D, "E": self.proj_E, "F": self.proj_F}[sensor]

    def _enc(self, x, sensor):
        h = self._encoder_for(sensor)(x)
        z = self._projector_for(sensor)(h)
        return h, z

    def forward_pair(self, x_a, x_b, name):
        """
        Run a pair (a, b) through their encoders + projection heads.

        Args:
            x_a, x_b: input tensors for sensor_a / sensor_b in the pair.
            name: one of {'de', 'ef', 'fd'}.

        Returns dict with:
            h_a, h_b               : (B, n_features)   representation space
            z_a, z_b               : (B, projection_dim) shared latent (contrastive)
            mu_a, mu_b             : aliases of z_a, z_b (shared-metric compat)
            sensor_a, sensor_b     : 'D' / 'E' / 'F'
            pair                   : name
        """
        if name not in self._PAIR_TABLE:
            raise ValueError(
                f"Unknown pair '{name}', expected one of {list(self._PAIR_TABLE)}"
            )
        sa, sb = self._PAIR_TABLE[name]

        h_a, z_a = self._enc(x_a, sa)
        h_b, z_b = self._enc(x_b, sb)

        return {
            "pair": name, "sensor_a": sa, "sensor_b": sb,
            "h_a": h_a, "h_b": h_b,
            "z_a": z_a, "z_b": z_b,
            # Aliases so _tri_log_metrics / _transitive_df_metric (which read
            # mu_a / mu_b) work without branching on model type.
            "mu_a": z_a, "mu_b": z_b,
        }

    def forward(self, batch):
        """Joint forward over all three pairs (dict {'de','ef','fd'})."""
        out = {}
        for name, (x_a, x_b) in batch.items():
            out[name] = self.forward_pair(x_a, x_b, name)
        return out
