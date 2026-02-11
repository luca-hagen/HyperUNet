from typing import Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels, num_conv=2, dropout=0.0):
        super().__init__()
        assert num_conv >= 1
        self.act = nn.LeakyReLU(inplace=True)

        layers = []
        
        for i in range(num_conv):
            conv_in = in_channels if i == 0 else out_channels
            conv = nn.Conv3d(conv_in, out_channels, kernel_size=3, padding=1, bias=False)
            bn = nn.GroupNorm(out_channels//8, num_channels=out_channels)
            block = [conv, bn]
            if dropout > 0 and i < num_conv - 1:  # Dropout nur zwischen den Convs
                block.append(nn.Dropout3d(dropout))
            if i < num_conv - 1:
                block.append(self.act)
            layers.append(nn.Sequential(*block))
        self.layers = nn.ModuleList(layers)

        self.shortcut = (
            nn.Conv3d(in_channels, out_channels, kernel_size=1, bias=False)
            if in_channels != out_channels else nn.Identity()
        )

    def forward(self, x):
        identity = self.shortcut(x)
        out = x
        for layer in self.layers:
            out = layer(out)
        out += identity
        return self.act(out)


class EncoderStage(nn.Module):
    def __init__(self, in_ch, out_ch, downsample=True, num_conv=2, dropout=0.0):
        super().__init__()
        self.resblock = ResBlock3D(in_ch, out_ch, num_conv=num_conv, dropout=dropout)
        self.downsample = downsample
        if downsample:
            # Strided Conv als Downsampling
            self.pool = nn.Conv3d(out_ch, out_ch, kernel_size=2, stride=2, bias=False)

    def forward(self, x):
        # Residual Block
        features = self.resblock(x)          # (B, out_channels, H, W, D)
        skip = features                      # Skip-Connection wird hier abgezweigt
        if self.downsample:
            x = self.pool(features)          # (B, out_channels, H/2, W/2, D/2)
            return x, skip
        else:
            # Bottleneck-Fall: kein Downsampling
            return features, skip
        

class ImageEncoder(nn.Module):
    """Simple 3D UNet encoder returning bottleneck and skips.
    Returns (bn, e4, e3, e2, e1) where bn is deepest low-res featuremap F_s.
    """
    def __init__(self, in_channels=1, base_filters=32, dropout=0.0):
        super().__init__()
        self.enc1 = EncoderStage(in_channels, base_filters, dropout=dropout)
        self.enc2 = EncoderStage(base_filters, base_filters*2, dropout=dropout)
        self.enc3 = EncoderStage(base_filters*2, base_filters*4, dropout=dropout)
        self.enc4 = EncoderStage(base_filters*4, base_filters*6, dropout=dropout)
        self.bottleneck = EncoderStage(base_filters*6, base_filters*8, downsample=False, num_conv=2, dropout=dropout)

    def forward(self, x):
        x, e1 = self.enc1(x)
        x, e2 = self.enc2(x)
        x, e3 = self.enc3(x)
        x, e4 = self.enc4(x)
        b, _ = self.bottleneck(x)
        return b, e4, e3, e2, e1
    

def MaskAveragePoolHigh(feats, mask):
    B = mask.shape[0]
    maps = []
    for feat in feats:
        feat_up = F.interpolate(feat, mask.shape[2:])
        mask_down = mask

        mask_min = mask_down.view(mask_down.size(0), -1).min(dim=1, keepdim=True).values.view(B, *([1] * (mask_down.ndim - 1)))
        mask_max = mask_down.view(mask_down.size(0), -1).max(dim=1, keepdim=True).values.view(B, *([1] * (mask_down.ndim - 1)))
        mask_down = (mask_down - mask_min) / (mask_max - mask_min + 1e-4)

        x = (mask_down*feat_up).sum(dim=(2,3,4))
        denom = mask_down.sum(dim=(2,3,4)) + 1e-4

        maps.append(x/denom)
    map = torch.cat(maps, dim=1)
    #map = F.layer_norm(map, map.shape[1:])
    return map


class HyperPatch(nn.Module):
    def __init__(self, in_channels=1, base_filters=32, dropout=0.0, num_samples=32, patch_size=3, group_size=8):
        super().__init__()
        self.base_filters = base_filters
        self.group_size = group_size
        self.num_samples = num_samples
        self.patch_size = patch_size
        self.convs_per_group = group_size
        self.num_convs = 24
        self.boundary_share = 0.7

        self.img_enc = ImageEncoder(in_channels=in_channels, base_filters=base_filters, dropout=dropout)

        # Embedding-Dimension: g * patch³ * num_samples
        self.emb_dim = group_size * (patch_size**3) * num_samples * 2

        # Output-Dimension: g*g*patch³
        self.filter_dim = self.num_convs * group_size * (patch_size**3)

        # Hypernetz 4096, 16384
        self.hypernet = nn.Sequential(
            nn.Linear(self.emb_dim + 256, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, 2048),
            nn.LayerNorm(2048),
            nn.GELU(),
            nn.Linear(2048, self.filter_dim)
        )
        
        self.up1 = nn.ConvTranspose3d(8*base_filters, 6*base_filters, kernel_size=2, stride=2)
        self.up2 = nn.ConvTranspose3d(6*base_filters, 4*base_filters, kernel_size=2, stride=2)
        self.up3 = nn.ConvTranspose3d(4*base_filters, 2*base_filters, kernel_size=2, stride=2)
        self.up4 = nn.ConvTranspose3d(2*base_filters, 1*base_filters, kernel_size=2, stride=2)

        self.up_sample = nn.ModuleList([
                            self.up1,
                            self.up2,
                            self.up3,
                            self.up4
                        ])


        self.norm3 = nn.GroupNorm(base_filters, 6*base_filters, affine=True)
        self.norm4 = nn.GroupNorm(base_filters, 6*base_filters, affine=True)
        self.norm5 = nn.GroupNorm(16, 4*base_filters, affine=True)
        self.norm6 = nn.GroupNorm(16, 4*base_filters, affine=True)
        self.norm7 = nn.GroupNorm(8, 2*base_filters, affine=True)
        self.norm8 = nn.GroupNorm(8, 2*base_filters, affine=True)
        self.norm9 = nn.GroupNorm(4, base_filters, affine=True)
        self.norm10 = nn.GroupNorm(4, base_filters, affine=True)          
        self.norm_list = nn.ModuleList([
                            self.norm3,
                            self.norm4,
                            self.norm5,
                            self.norm6,
                            self.norm7,
                            self.norm8,
                            self.norm9,
                            self.norm10
                        ])
        
        self.task_pe = nn.Sequential(
            nn.Linear(256 + base_filters*8, 512),
            nn.GELU(),
            nn.Linear(512, 256)
        )


        self.pe = nn.ParameterList([
                            0.05 * nn.Parameter(torch.randn((1, (base_filters*6 // self.group_size) * (base_filters*6 // self.num_convs), 256))),
                            0.05 * nn.Parameter(torch.randn((1, (base_filters*6 // self.group_size) * (base_filters*6 // self.num_convs), 256))),
                            0.05 * nn.Parameter(torch.randn((1, (base_filters*4 // self.group_size) * (base_filters*4 // self.num_convs), 256))),
                            0.05 * nn.Parameter(torch.randn((1, (base_filters*4 // self.group_size) * (base_filters*4 // self.num_convs), 256))),
                            0.05 * nn.Parameter(torch.randn((1, (base_filters*2 // self.group_size) * (base_filters*2 // self.num_convs), 256))),
                            0.05 * nn.Parameter(torch.randn((1, (base_filters*2 // self.group_size) * (base_filters*2 // self.num_convs), 256))),
                            0.05 * nn.Parameter(torch.randn((1, (base_filters*1 // self.group_size) * (base_filters*1 // self.num_convs), 256))),
                            0.05 * nn.Parameter(torch.randn((1, (base_filters*1 // self.group_size) * (base_filters*1 // self.num_convs), 256))),
        ])

        self.final_bias = nn.Sequential(
            nn.Linear(8*base_filters, 8*base_filters),
            nn.GELU(),
            nn.Linear(8*base_filters, 1)
        )

        self.final_weights  = nn.Sequential(
            nn.Linear(8*base_filters, 8*base_filters),
            nn.GELU(),
            nn.Linear(8*base_filters, base_filters)
        )

        self.final_scale = nn.Parameter(torch.ones((1)))

        self.scale = nn.ParameterList([
            nn.Parameter(torch.ones((1))),
            nn.Parameter(torch.ones((1))),
            nn.Parameter(torch.ones((1))),
            nn.Parameter(torch.ones((1)))
        ])


    def forward(self, x_s: torch.Tensor, y_s: torch.Tensor, x_q: torch.Tensor):
        B = x_s.shape[0]

        # Encoder nur einmal aufrufen, Listen direkt nutzen
        f_maps_s = self.img_enc(x_s)
        f_maps_q = self.img_enc(x_q)

        input_s = f_maps_s[0]
        input_q = f_maps_q[0]
        weight_matrices = []

        task_map = MaskAveragePoolHigh([input_s], y_s)
        # Device und konstante Werte vorab berechnen
        device = input_s.device
        g_size = self.group_size
        p = self.patch_size
        
        for idx, (f_map_s, f_map_q) in enumerate(zip(f_maps_s[1:], f_maps_q[1:])):
            # Upsampling und Addition
            input_s = self.up_sample[idx](input_s) + f_map_s
            input_q = self.up_sample[idx](input_q) + f_map_q

            res_s = input_s
            res_q = input_q
            
            # Y-Map Downsampling optimiert
            scale_factor = y_s.shape[2] // input_s.shape[2]
            if scale_factor > 2:
                y_map = F.adaptive_max_pool3d(F.max_pool3d(y_s.float(), kernel_size=3, stride=1, padding=1),
                                               input_s.shape[2:])
                """
                y_map = F.interpolate(
                    F.max_pool3d(y_s.float(), kernel_size=3, stride=1, padding=1),
                    size=input_s.shape[2:],
                    mode="nearest"
                )"""
            else: 
                y_map = F.adaptive_max_pool3d(y_s.float(), input_s.shape[2:])
                #y_map = F.interpolate(y_s.float(), scale_factor=1/scale_factor, mode="nearest")
            
            if y_map.ndim == 5:
                y_map = y_map[:, 0]

            
            for i in range(2):
                B, C, H, W, D = f_map_q.shape
                K = C // g_size
                O = C // self.num_convs
                J = O

                # Permutationen vorab berechnen (außerhalb der Schleife wenn möglich)
                perms = torch.stack([
                    torch.randperm(C, generator=torch.Generator().manual_seed(j)).to(device)
                    for j in range(J)
                ], dim=0)
                perms_flat = perms.reshape(-1)
                g_inp_sel = input_s.index_select(1, perms_flat)
                spatial = input_s.shape[2:]
                g_inp_all = g_inp_sel.view(B, J, C, *spatial).permute(1, 0, 2, 3, 4, 5).contiguous()
                y_map_rep = y_map.unsqueeze(0).expand(J, -1, -1, -1, -1).contiguous()

                chunk_size = 64
                
                # Weights pre-allocate
                weights = torch.zeros((B, C, C, 3, 3, 3), device=device, dtype=input_s.dtype)
                
                for j0 in range(0, J, chunk_size):
                    j1 = min(J, j0 + chunk_size)
                    Jc = j1 - j0
                    
                    # Slicing und Reshaping kombinieren
                    g_inp_chunk = g_inp_all[j0:j1].reshape(Jc * B, C, *spatial)
                    y_map_chunk = y_map_rep[j0:j1].reshape(Jc * B, *y_map.shape[1:])

                    avg = task_map.unsqueeze(1).repeat(1, Jc*K, 1).view(B*Jc, K, -1)

                    # Embeddings und Hypernet
                    g_emb = sample_neighborhoods_group_fast_vec(
                        y_map_chunk, g_inp_chunk,
                        num_samples=self.num_samples,
                        patch_size=p, 
                        group_size=g_size,
                        num_runs=3,
                        boundary_share=self.boundary_share
                    )
                    
                    pos_emb = self.pe[2*idx+i][:, j0*K:j1*K].repeat(B, 1, 1).view(B*Jc, K, -1)
                    #pos_emb = pos_emb.unsqueeze(2).repeat(1, 1, 3, 1) #ablation
                    pos_emb = self.task_pe(torch.cat([pos_emb, avg], dim=2)).unsqueeze(2).repeat(1, 1, 3, 1)
                    g_emb = torch.cat([g_emb, pos_emb], dim=3)
                    
                    # Hypernet und Reshaping optimiert
                    filters = self.hypernet(g_emb).mean(dim=2)
                    g_filter_flat = filters.view(Jc * B, K, self.num_convs, g_size, p, p, p)
                    g_filter_flat = g_filter_flat.permute(0, 2, 1, 3, 4, 5, 6).reshape(
                        Jc * B, self.num_convs, C, p, p, p
                    )

                    # Reshape für scatter
                    g_filter_concat = g_filter_flat.view(Jc, B, self.num_convs, C, p, p, p) \
                        .permute(1, 0, 2, 3, 4, 5, 6).reshape(B, Jc * self.num_convs, C, p, p, p)

                    # Index für scatter_add
                    out_chunk = self.num_convs * Jc
                    perms_chunk = perms[j0:j1]
                    w_idx = perms_chunk.repeat_interleave(self.num_convs, dim=0) \
                        .view(1, out_chunk, C, 1, 1, 1) \
                        .expand(B, out_chunk, C, p, p, p)

                    # Scatter_add
                    out_offset = j0 * self.num_convs
                    weights[:, out_offset:out_offset + out_chunk].scatter_add_(2, w_idx, g_filter_concat)
                
                # Convolution
                weight = self.scale[idx] * F.tanh(weights.view(B*C, C, p, p, p))
                
                # Beide Convolutions parallel vorbereiten
                inp_combined = torch.cat([
                    input_s.view(1, B * C, H, W, D),
                    input_q.view(1, B * C, H, W, D)
                ], dim=0)
                
                out_combined = F.conv3d(inp_combined, weight=weight, stride=1, padding=p // 2, groups=B)
                out_s, out_q = out_combined.chunk(2, dim=0)

                weight_matrices.append(weight.view(B, C, C, p, p, p))

                # Normalization und Activation
                norm_layer = self.norm_list[2*idx+i]
                s_random_maps = norm_layer(out_s.view(B, C, D, H, W))
                q_random_maps = norm_layer(out_q.view(B, C, D, H, W))

                if i < 1:
                    input_s = F.leaky_relu(s_random_maps)
                    input_q = F.leaky_relu(q_random_maps)
                else:
                    input_s = F.leaky_relu(s_random_maps + res_s)
                    input_q = F.leaky_relu(q_random_maps + res_q)

        # Final Convolution
        B, C, D, H, W = input_s.shape

        final_w = self.final_scale * F.tanh(self.final_weights(task_map).view(B, 1, C, 1, 1, 1))
        final_b = self.final_bias(task_map)

        weight_matrices.extend([final_w, final_b])

        # Final Conv für beide inputs parallel
        weight = final_w.view(B, C, 1, 1, 1)
        bias = final_b.view(1, B, 1, 1, 1)
        
        inp_combined = torch.cat([
            input_s.view(1, B * C, H, W, D),
            input_q.view(1, B * C, H, W, D)
        ], dim=0)
        
        out_combined = F.conv3d(inp_combined, weight=weight, stride=1, groups=B) + bias
        s_logits, q_logits = out_combined.chunk(2, dim=0)

        s_logits = s_logits.view(B, 1, H, W, D)
        q_logits = q_logits.view(B, 1, H, W, D)
        
        return s_logits, q_logits, weight_matrices
    


def sample_neighborhoods_group_fast_vec(mask: torch.Tensor,
                                        fmap: torch.Tensor,
                                        num_samples: int = 32,
                                        patch_size: int = 3,
                                        group_size: int = 8,
                                        num_runs: int = 1,
                                        debug: bool = False,
                                        boundary_share: float = 0.7):
    assert patch_size % 2 == 1, "patch_size muss ungerade sein"
    device = fmap.device
    B, C, D, H, W = fmap.shape
    g = group_size
    assert C % g == 0, f"Channels {C} must be divisible by group_size {g}"
    G = C // g
    p = patch_size
    r = p // 2
    p3 = p**3

    fmap_pad = F.pad(fmap, (r, r, r, r, r, r), mode='replicate')
    Dp, Hp, Wp = fmap_pad.shape[2:]
    S_un = D * H * W
    S_p = Dp * Hp * Wp

    valid_mask = torch.zeros_like(mask, dtype=torch.bool, device=device)
    valid_mask[:, r:D - r, r:H - r, r:W - r] = True
    valid_flat = valid_mask.view(B, -1)

    mask_flat = mask.view(B, -1)

    kernel = torch.ones((1, 1, p, p, p), device=device, dtype=torch.float32)
    mask_counts = F.conv3d(mask.unsqueeze(1).float(), kernel, padding=r)[:, 0]

    pos_boundary_flat = ((mask == 1) & (mask_counts < p3)).view(B, -1).float()
    neg_boundary_flat = ((mask == 0) & (mask_counts > 0)).view(B, -1).float()

    pos_flat = (mask_flat == 1).float()
    neg_flat = (mask_flat == 0).float()

    def expand_BG(mat):
        return mat.repeat_interleave(G, dim=0)

    pos_boundary_BG = expand_BG(pos_boundary_flat)
    neg_boundary_BG = expand_BG(neg_boundary_flat)
    pos_BG = expand_BG(pos_flat)
    neg_BG = expand_BG(neg_flat)
    valid_BG = expand_BG(valid_flat.float())

    # Zusätzliche Guard: valid_BG darf nicht leer sein
    if debug:
        assert (valid_BG.sum(1) > 0).all(), "valid_BG enthält leere Zeilen; valid centers fehlen (prüfe patch_size/ränder)"

    def sample_k_from_mask_BG(mask_probs_BG, k, num_runs=1, fallback_BG=None):
        N, S = mask_probs_BG.shape
        probs = mask_probs_BG.float()
        probs = torch.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = probs.clamp(min=0.0)

        sums = probs.sum(dim=1, keepdim=True)
        zero_rows = (sums.squeeze(1) == 0)

        if zero_rows.any():
            if fallback_BG is not None:
                fb = fallback_BG.float()
                fb = torch.nan_to_num(fb, nan=0.0, posinf=0.0, neginf=0.0).clamp(min=0.0)
                # Falls auch der Fallback leer ist, setze uniform
                fb_sums = fb.sum(dim=1, keepdim=True)
                fb_zero = (fb_sums.squeeze(1) == 0)
                fb[fb_zero] = 1.0  # uniform Masse
                probs[zero_rows] = fb[zero_rows]
            else:
                probs[zero_rows] = 1.0  # uniform Masse

            sums = probs.sum(dim=1, keepdim=True)

        # Normalisieren
        probs = probs / (sums + 1e-12)

        if debug:
            assert torch.isfinite(probs).all(), "Nicht-finite Wahrscheinlichkeiten nach Normalisierung"
            assert (probs >= 0).all(), "Negative Wahrscheinlichkeiten"
            assert (probs.sum(1).abs() - 1.0).abs().max() < 1e-5, "Zeilensummen ungleich 1"

        # Ziehe num_runs*k Indizes und forme in [N, num_runs, k]
        try:
            idx = torch.multinomial(probs, num_samples=max(num_runs * k, 1), replacement=True)
        except RuntimeError as e:
            if debug:
                print("multinomial failed; fallback uniform. Error:", e)
            # Uniform-Fallback sicher
            idx = torch.randint(0, S, (N, max(num_runs * k, 1)), device=device)
        return idx.view(N, num_runs, k) if k > 0 else torch.empty((N, num_runs, 0), dtype=torch.long, device=device)

    boundary_k = int(num_samples * boundary_share)
    rest_k = num_samples - boundary_k
    N = B * G

    pos_bound_idx_BG = sample_k_from_mask_BG(pos_boundary_BG, boundary_k, num_runs, fallback_BG=pos_BG)
    pos_rest_idx_BG = sample_k_from_mask_BG(pos_BG, rest_k, num_runs, fallback_BG=valid_BG)
    pos_idx_BG = torch.cat([pos_bound_idx_BG, pos_rest_idx_BG], dim=2)  # [N, num_runs, num_samples]

    neg_bound_idx_BG = sample_k_from_mask_BG(neg_boundary_BG, boundary_k, num_runs, fallback_BG=neg_BG)
    neg_rest_idx_BG = sample_k_from_mask_BG(neg_BG, rest_k, num_runs, fallback_BG=valid_BG)
    neg_idx_BG = torch.cat([neg_bound_idx_BG, neg_rest_idx_BG], dim=2)

    pos_idx = pos_idx_BG.view(B, G, num_runs, num_samples)
    neg_idx = neg_idx_BG.view(B, G, num_runs, num_samples)

    def unravel(flat_idx_BG):
        z = (flat_idx_BG // (H * W)).long()
        rem = flat_idx_BG % (H * W)
        y = (rem // W).long()
        x = (rem % W).long()
        return z, y, x

    pos_z, pos_y, pos_x = unravel(pos_idx)
    neg_z, neg_y, neg_x = unravel(neg_idx)

    pos_zp = pos_z + r
    pos_yp = pos_y + r
    pos_xp = pos_x + r
    neg_zp = neg_z + r
    neg_yp = neg_y + r
    neg_xp = neg_x + r

    stride_z = Hp * Wp
    stride_y = Wp
    dz = torch.arange(-r, r+1, device=device, dtype=torch.long) * stride_z
    dy = torch.arange(-r, r+1, device=device, dtype=torch.long) * stride_y
    dx = torch.arange(-r, r+1, device=device, dtype=torch.long)
    offs = (dz[:, None, None] + dy[None, :, None] + dx[None, None, :]).reshape(-1)

    center_pos_flat = (pos_zp * stride_z + pos_yp * stride_y + pos_xp).long()
    center_neg_flat = (neg_zp * stride_z + neg_yp * stride_y + neg_xp).long()

    # Indizes bilden: [B, G, num_runs, num_samples, p3]
    idxs_pos = center_pos_flat.unsqueeze(-1) + offs.view(1, 1, 1, 1, p3)
    idxs_neg = center_neg_flat.unsqueeze(-1) + offs.view(1, 1, 1, 1, p3)

    # Guard vor gather: alle Indizes im Bereich [0, S_p)
    if debug:
        assert idxs_pos.min() >= 0 and idxs_pos.max() < S_p, f"pos idx out of bounds (0..{S_p-1})"
        assert idxs_neg.min() >= 0 and idxs_neg.max() < S_p, f"neg idx out of bounds (0..{S_p-1})"

    M = num_samples * p3
    idxs_pos_flat = idxs_pos.view(B, G, num_runs, M)
    idxs_neg_flat = idxs_neg.view(B, G, num_runs, M)

    source_group = fmap_pad.view(B, G, g, S_p)

    # Expand für gather
    idxs_pos_exp = idxs_pos_flat.unsqueeze(3).expand(B, G, num_runs, g, M)
    idxs_neg_exp = idxs_neg_flat.unsqueeze(3).expand(B, G, num_runs, g, M)

    # Gather auf dim=4 (S_p)
    source_exp = source_group.unsqueeze(2).expand(B, G, num_runs, g, S_p)
    patches_pos_g = torch.gather(source_exp, dim=4, index=idxs_pos_exp)
    patches_neg_g = torch.gather(source_exp, dim=4, index=idxs_neg_exp)

    patches_pos_g = patches_pos_g.view(B, G, num_runs, g, num_samples, p3)
    patches_neg_g = patches_neg_g.view(B, G, num_runs, g, num_samples, p3)

    patches_pos_vec = patches_pos_g.permute(0, 1, 2, 4, 3, 5).reshape(B, G, num_runs, num_samples, g * p3)
    patches_neg_vec = patches_neg_g.permute(0, 1, 2, 4, 3, 5).reshape(B, G, num_runs, num_samples, g * p3)

    pos_flat_out = patches_pos_vec.reshape(B, G, num_runs, num_samples * g * p3)
    neg_flat_out = patches_neg_vec.reshape(B, G, num_runs, num_samples * g * p3)

    emb = torch.cat([pos_flat_out, neg_flat_out], dim=3)  # [B, G, num_runs, 2 * num_samples * g * p3]
    return emb


def sample_patches(volume, mask, N):
    B, C, D, H, W = volume.shape
    device = volume.device

    vol_padded = F.pad(volume, (1,1,1,1,1,1), mode="replicate")

    all_coords = []
    for b in range(B):
        mask_b = mask[b,0]

        pos_idx = mask_b.nonzero(as_tuple=False)  # (M,3)
        neg_idx = (mask_b == 0).nonzero(as_tuple=False)

        # --- positives ---
        if len(pos_idx) == 0:
            pos_sel = torch.stack([
                torch.randint(D, (N,), device=device),
                torch.randint(H, (N,), device=device),
                torch.randint(W, (N,), device=device)
            ], dim=1)
        elif len(pos_idx) >= N:
            perm = torch.randperm(len(pos_idx), device=device)[:N]
            pos_sel = pos_idx[perm]
        else:
            pos_sel = pos_idx[torch.randint(len(pos_idx), (N,), device=device)]

        # --- negatives ---
        if len(neg_idx) == 0:
            neg_sel = torch.stack([
                torch.randint(D, (N,), device=device),
                torch.randint(H, (N,), device=device),
                torch.randint(W, (N,), device=device)
            ], dim=1)
        elif len(neg_idx) >= N:
            perm = torch.randperm(len(neg_idx), device=device)[:N]
            neg_sel = neg_idx[perm]
        else:
            neg_sel = neg_idx[torch.randint(len(neg_idx), (N,), device=device)]

        coords = torch.cat([pos_sel, neg_sel], dim=0)  # (2N,3)
        all_coords.append(coords)

    coords = torch.stack(all_coords, dim=0)  # (B,2N,3)
    coords = coords + 1  # +1 wegen Padding

    # Normierte Koordinaten für grid_sample
    d = coords[:,:,0].float() / (D+1) * 2 - 1
    h = coords[:,:,1].float() / (H+1) * 2 - 1
    w = coords[:,:,2].float() / (W+1) * 2 - 1
    centers = torch.stack([w,h,d], dim=-1)  # (B,2N,3)

    offsets = torch.stack(torch.meshgrid(
        torch.linspace(-1,1,3,device=device),
        torch.linspace(-1,1,3,device=device),
        torch.linspace(-1,1,3,device=device),
        indexing="ij"
    ), dim=-1).view(-1,3)  # (27,3)

    grid = centers.unsqueeze(2) + offsets.view(1,1,27,3)
    grid = grid.view(B,2*N*27,1,1,3)

    patches = F.grid_sample(vol_padded, grid, align_corners=True)
    patches = patches.squeeze(-1).squeeze(-1)  # (B,C,2N*27)
    patches = patches.view(B,C,2*N,27).permute(0,2,1,3)  # (B,2N,C,27)
    patches = patches.reshape(B,2*N,-1)

    return patches
