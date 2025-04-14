import torch
import torch.nn as nn
import clip
from models.xrestormer import OCAB, TransformerBlock, Upsample, OverlapPatchEmbed, Downsample, ChannelAttention
from einops import rearrange

class Text_XRestormer(nn.Module):
    def __init__(self, model_clip, inp_A_channels=3, inp_B_channels=3, out_channels=3,
                 dim=48, num_blocks=[2,2,2,2],
                 num_refinement_blocks=4,
                 channel_heads=[1,2,4,8],
                 spatial_heads=[2,2,3,4],
                 overlap_ratio=[0.5,0.5,0.5,0.5],
                 window_size=8,
                 spatial_dim_head=16,
                 ffn_expansion_factor=2.66,
                 cross_attention_type="TSA",
                 fusion_type="1x1conv",
                 bias=False,
                 LayerNorm_type='WithBias'):
        super(Text_XRestormer, self).__init__()
        
        self.model_clip = model_clip
        self.model_clip.eval()
        self.fusion_type = fusion_type
        if fusion_type == '1x1conv':
            FusionClass = Fusion_Embed
        elif fusion_type == 'spatial_attention':
            FusionClass = AttentionFusion_Embed
        elif fusion_type == 'dynamic':
            FusionClass = DynamicFusion_Embed
        elif fusion_type == 'multiscale':
            FusionClass = MultiscaleFusion_Embed
        elif fusion_type == 'residual':
            FusionClass = ResidualFusion_Embed
        elif fusion_type == 'cross_attention':
            FusionClass = CrossFusion_Embed
        else:
            raise ValueError(f'Unknown fusion type: {fusion_type}')
        self.encoder_A = XRestormer_Encoder(
            inp_channels=inp_A_channels,
            dim=dim,
            num_blocks=num_blocks,
            channel_heads=channel_heads,
            spatial_heads=spatial_heads,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            spatial_dim_head=spatial_dim_head,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type
        )
        
        self.encoder_B = XRestormer_Encoder(
            inp_channels=inp_B_channels,
            dim=dim,
            num_blocks=num_blocks,
            channel_heads=channel_heads, 
            spatial_heads=spatial_heads,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            spatial_dim_head=spatial_dim_head,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type
        )

        self.cross_attention_type = cross_attention_type
        if cross_attention_type == "TSA":
            self.cross_attention = Cross_attentionTSA(dim * 2 ** 3)
            self.attention_spatial = OCAB(
                dim=dim * 2 ** 3,
                window_size=window_size,
                overlap_ratio=overlap_ratio[-1],
                num_heads=spatial_heads[-1],
                dim_head=spatial_dim_head,
                bias=bias
            )
        else:
            self.cross_attention = Cross_attentionSSA(
                dim=dim * 2 ** 3,
                window_size=window_size,
                overlap_ratio=overlap_ratio[-1],
                num_spatial_heads=spatial_heads[-1],
                spatial_dim_head=spatial_dim_head,
                bias=bias
            )

        # Level 4
        self.feature_fusion_4 = FusionClass(embed_dim=dim * 2 ** 3)
        self.prompt_guidance_4 = FeatureWiseAffine(in_channels=512, out_channels=dim * 2 ** 3)
        self.decoder_level4 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2 ** 3),
                window_size=window_size,
                overlap_ratio=overlap_ratio[3],
                num_channel_heads=channel_heads[3],
                num_spatial_heads=spatial_heads[3],
                spatial_dim_head=spatial_dim_head,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for i in range(num_blocks[3])
        ])

        # Level 3
        self.feature_fusion_3 = FusionClass(embed_dim=dim * 2 ** 2)
        self.prompt_guidance_3 = FeatureWiseAffine(in_channels=512, out_channels=dim * 2 ** 2)
        self.up4_3 = Upsample(int(dim * 2 ** 3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2 ** 2),
                window_size=window_size,
                overlap_ratio=overlap_ratio[2],
                num_channel_heads=channel_heads[2],
                num_spatial_heads=spatial_heads[2],
                spatial_dim_head=spatial_dim_head,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for i in range(num_blocks[2])
        ])

        # Level 2
        self.feature_fusion_2 = FusionClass(embed_dim=dim * 2)
        self.prompt_guidance_2 = FeatureWiseAffine(in_channels=512, out_channels=dim * 2)
        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2),
                window_size=window_size,
                overlap_ratio=overlap_ratio[1],
                num_channel_heads=channel_heads[1],
                num_spatial_heads=spatial_heads[1],
                spatial_dim_head=spatial_dim_head,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for i in range(num_blocks[1])
        ])

        # Level 1
        self.feature_fusion_1 = FusionClass(embed_dim=dim)
        self.prompt_guidance_1 = FeatureWiseAffine(in_channels=512, out_channels=dim)
        self.up2_1 = Upsample(int(dim * 2))
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2),
                window_size=window_size,
                overlap_ratio=overlap_ratio[0],
                num_channel_heads=channel_heads[0],
                num_spatial_heads=spatial_heads[0],
                spatial_dim_head=spatial_dim_head,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for i in range(num_blocks[0])
        ])

        # Refinement
        self.refinement = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2),
                window_size=window_size,
                overlap_ratio=overlap_ratio[0],
                num_channel_heads=channel_heads[0],
                num_spatial_heads=spatial_heads[0],
                spatial_dim_head=spatial_dim_head,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for i in range(num_refinement_blocks)
        ])

        self.output = nn.Conv2d(int(dim * 2), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img_A, inp_img_B, text, return_features=False):
        text_tokens = clip.tokenize(text).to(inp_img_A.device)
        text_features = self.get_text_feature(text_tokens).to(inp_img_A.dtype)
        out_enc_level4_A, out_enc_level3_A, out_enc_level2_A, out_enc_level1_A = self.encoder_A(inp_img_A)
        out_enc_level4_B, out_enc_level3_B, out_enc_level2_B, out_enc_level1_B = self.encoder_B(inp_img_B)
        if return_features:
            features = []
        
        # Level 4
        out_enc_level4_A, out_enc_level4_B = self.cross_attention(out_enc_level4_A, out_enc_level4_B)
        out_enc_level4 = self.feature_fusion_4(out_enc_level4_A, out_enc_level4_B)
        if self.cross_attention_type == "TSA":
            out_enc_level4 = self.attention_spatial(out_enc_level4)
        out_enc_level4 = self.prompt_guidance_4(out_enc_level4, text_features)
        out_dec_level4 = self.decoder_level4(out_enc_level4)
        
        # Level 3
        inp_dec_level3 = self.up4_3(out_dec_level4)
        inp_dec_level3 = self.prompt_guidance_3(inp_dec_level3, text_features)
        out_enc_level3 = self.feature_fusion_3(out_enc_level3_A, out_enc_level3_B)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        # Level 2
        inp_dec_level2 = self.up3_2(out_dec_level3)
        inp_dec_level2 = self.prompt_guidance_2(inp_dec_level2, text_features)
        out_enc_level2 = self.feature_fusion_2(out_enc_level2_A, out_enc_level2_B)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        # Level 1
        inp_dec_level1 = self.up2_1(out_dec_level2)
        inp_dec_level1 = self.prompt_guidance_1(inp_dec_level1, text_features)
        out_enc_level1 = self.feature_fusion_1(out_enc_level1_A, out_enc_level1_B)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        if return_features:
            features.append(out_dec_level4)
            features.append(out_dec_level3)
            features.append(out_dec_level2)
            features.append(out_dec_level1)

        # Refinement & Output
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img_A

        if return_features:
            return out_dec_level1, features
        return out_dec_level1

    @torch.no_grad()
    def get_text_feature(self, text):
        text_feature = self.model_clip.encode_text(text)
        return text_feature

# no text version
class XRestormer(nn.Module):
    def __init__(self, inp_A_channels=3, inp_B_channels=3, out_channels=3,
                 dim=48, num_blocks=[2,2,2,2],
                 num_refinement_blocks=4,
                 channel_heads=[1,2,4,8],
                 spatial_heads=[2,2,3,4],
                 overlap_ratio=[0.5,0.5,0.5,0.5],
                 window_size=8,
                 spatial_dim_head=16,
                 ffn_expansion_factor=2.66,
                 cross_attention_type="TSA",
                 fusion_type="1x1conv",
                 bias=False,
                 LayerNorm_type='WithBias'):
        super(XRestormer, self).__init__()
        
        self.fusion_type = fusion_type
        if fusion_type == '1x1conv':
            FusionClass = Fusion_Embed
        elif fusion_type == 'spatial_attention':
            FusionClass = AttentionFusion_Embed
        elif fusion_type == 'dynamic':
            FusionClass = DynamicFusion_Embed
        elif fusion_type == 'multiscale':
            FusionClass = MultiscaleFusion_Embed
        elif fusion_type == 'residual':
            FusionClass = ResidualFusion_Embed
        elif fusion_type == 'cross_attention':
            FusionClass = CrossFusion_Embed

        self.encoder_A = XRestormer_Encoder(
            inp_channels=inp_A_channels,
            dim=dim,
            num_blocks=num_blocks,
            channel_heads=channel_heads,
            spatial_heads=spatial_heads,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            spatial_dim_head=spatial_dim_head,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type
        )
        
        self.encoder_B = XRestormer_Encoder(
            inp_channels=inp_B_channels,
            dim=dim,
            num_blocks=num_blocks,
            channel_heads=channel_heads, 
            spatial_heads=spatial_heads,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            spatial_dim_head=spatial_dim_head,
            ffn_expansion_factor=ffn_expansion_factor,
            bias=bias,
            LayerNorm_type=LayerNorm_type
        )

        self.cross_attention_type = cross_attention_type
        if cross_attention_type == "TSA":
            self.cross_attention = Cross_attentionTSA(dim * 2 ** 3)
            self.attention_spatial = OCAB(
                dim=dim * 2 ** 3,
                window_size=window_size,
                overlap_ratio=overlap_ratio[-1],
                num_heads=spatial_heads[-1],
                dim_head=spatial_dim_head,
                bias=bias
            )
        else:
            self.cross_attention = Cross_attentionSSA(
                dim=dim * 2 ** 3,
                window_size=window_size,
                overlap_ratio=overlap_ratio[-1],
                num_spatial_heads=spatial_heads[-1],
                spatial_dim_head=spatial_dim_head,
                bias=bias
            )

        # Level 4
        self.feature_fusion_4 = FusionClass(embed_dim=dim * 2 ** 3)
        self.decoder_level4 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2 ** 3),
                window_size=window_size,
                overlap_ratio=overlap_ratio[3],
                num_channel_heads=channel_heads[3],
                num_spatial_heads=spatial_heads[3],
                spatial_dim_head=spatial_dim_head,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for i in range(num_blocks[3])
        ])

        # Level 3
        self.feature_fusion_3 = FusionClass(embed_dim=dim * 2 ** 2)
        self.up4_3 = Upsample(int(dim * 2 ** 3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim * 2 ** 3), int(dim * 2 ** 2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2 ** 2),
                window_size=window_size,
                overlap_ratio=overlap_ratio[2],
                num_channel_heads=channel_heads[2],
                num_spatial_heads=spatial_heads[2],
                spatial_dim_head=spatial_dim_head,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for i in range(num_blocks[2])
        ])

        # Level 2
        self.feature_fusion_2 = FusionClass(embed_dim=dim * 2)
        self.up3_2 = Upsample(int(dim * 2 ** 2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim * 2 ** 2), int(dim * 2), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2),
                window_size=window_size,
                overlap_ratio=overlap_ratio[1],
                num_channel_heads=channel_heads[1],
                num_spatial_heads=spatial_heads[1],
                spatial_dim_head=spatial_dim_head,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for i in range(num_blocks[1])
        ])

        # Level 1
        self.feature_fusion_1 = FusionClass(embed_dim=dim)
        self.up2_1 = Upsample(int(dim * 2))
        self.decoder_level1 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2),
                window_size=window_size,
                overlap_ratio=overlap_ratio[0],
                num_channel_heads=channel_heads[0],
                num_spatial_heads=spatial_heads[0],
                spatial_dim_head=spatial_dim_head,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for i in range(num_blocks[0])
        ])

        # Refinement
        self.refinement = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim * 2),
                window_size=window_size,
                overlap_ratio=overlap_ratio[0],
                num_channel_heads=channel_heads[0],
                num_spatial_heads=spatial_heads[0],
                spatial_dim_head=spatial_dim_head,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for i in range(num_refinement_blocks)
        ])

        self.output = nn.Conv2d(int(dim * 2), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, inp_img_A, inp_img_B, return_features=False):
        out_enc_level4_A, out_enc_level3_A, out_enc_level2_A, out_enc_level1_A = self.encoder_A(inp_img_A)
        out_enc_level4_B, out_enc_level3_B, out_enc_level2_B, out_enc_level1_B = self.encoder_B(inp_img_B)
        if return_features:
            features = []
        
        # Level 4
        out_enc_level4_A, out_enc_level4_B = self.cross_attention(out_enc_level4_A, out_enc_level4_B)
        out_enc_level4 = self.feature_fusion_4(out_enc_level4_A, out_enc_level4_B)
        if self.cross_attention_type == "TSA":
            out_enc_level4 = self.attention_spatial(out_enc_level4)
        out_dec_level4 = self.decoder_level4(out_enc_level4)
        
        # Level 3
        inp_dec_level3 = self.up4_3(out_dec_level4)
        out_enc_level3 = self.feature_fusion_3(out_enc_level3_A, out_enc_level3_B)
        inp_dec_level3 = torch.cat([inp_dec_level3, out_enc_level3], 1)
        inp_dec_level3 = self.reduce_chan_level3(inp_dec_level3)
        out_dec_level3 = self.decoder_level3(inp_dec_level3)

        # Level 2
        inp_dec_level2 = self.up3_2(out_dec_level3)
        out_enc_level2 = self.feature_fusion_2(out_enc_level2_A, out_enc_level2_B)
        inp_dec_level2 = torch.cat([inp_dec_level2, out_enc_level2], 1)
        inp_dec_level2 = self.reduce_chan_level2(inp_dec_level2)
        out_dec_level2 = self.decoder_level2(inp_dec_level2)

        # Level 1
        inp_dec_level1 = self.up2_1(out_dec_level2)
        out_enc_level1 = self.feature_fusion_1(out_enc_level1_A, out_enc_level1_B)
        inp_dec_level1 = torch.cat([inp_dec_level1, out_enc_level1], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        if return_features:
            features.append(out_dec_level4)
            features.append(out_dec_level3)
            features.append(out_dec_level2)
            features.append(out_dec_level1)

        # Refinement & Output
        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img_A

        if return_features:
            return out_dec_level1, features
        return out_dec_level1

class XRestormer_Encoder(nn.Module):
    def __init__(self, inp_channels=3, dim=48, num_blocks=[2,2,2,2], 
                 channel_heads=[1,2,4,8], spatial_heads=[2,2,3,4],
                 window_size=8, overlap_ratio=[0.5,0.5,0.5,0.5],
                 spatial_dim_head=16, ffn_expansion_factor=2.66,
                 bias=False, LayerNorm_type='WithBias'):
        super(XRestormer_Encoder, self).__init__()

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)

        # Level 1
        self.encoder_level1 = nn.Sequential(*[
            TransformerBlock(
                dim=dim,
                window_size=window_size,
                overlap_ratio=overlap_ratio[0],
                num_channel_heads=channel_heads[0],
                num_spatial_heads=spatial_heads[0],
                spatial_dim_head=spatial_dim_head,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for i in range(num_blocks[0])
        ])

        # Level 2
        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim*2**1),
                window_size=window_size,
                overlap_ratio=overlap_ratio[1],
                num_channel_heads=channel_heads[1],
                num_spatial_heads=spatial_heads[1],
                spatial_dim_head=spatial_dim_head,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for i in range(num_blocks[1])
        ])

        # Level 3
        self.down2_3 = Downsample(int(dim*2**1))
        self.encoder_level3 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim*2**2),
                window_size=window_size,
                overlap_ratio=overlap_ratio[2],
                num_channel_heads=channel_heads[2],
                num_spatial_heads=spatial_heads[2],
                spatial_dim_head=spatial_dim_head,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for i in range(num_blocks[2])
        ])

        # Level 4
        self.down3_4 = Downsample(int(dim*2**2))
        self.encoder_level4 = nn.Sequential(*[
            TransformerBlock(
                dim=int(dim*2**3),
                window_size=window_size,
                overlap_ratio=overlap_ratio[3],
                num_channel_heads=channel_heads[3],
                num_spatial_heads=spatial_heads[3],
                spatial_dim_head=spatial_dim_head,
                ffn_expansion_factor=ffn_expansion_factor,
                bias=bias,
                LayerNorm_type=LayerNorm_type
            ) for i in range(num_blocks[3])
        ])

    def forward(self, x):
        # Level 1
        inp_enc_level1 = self.patch_embed(x)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        # Level 2
        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        # Level 3
        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        # Level 4
        inp_enc_level4 = self.down3_4(out_enc_level3)
        out_enc_level4 = self.encoder_level4(inp_enc_level4)

        return out_enc_level4, out_enc_level3, out_enc_level2, out_enc_level1

class Cross_attentionTSA(nn.Module):
    def __init__(self, dim, num_heads=8):
        super(Cross_attentionTSA, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        
        self.qkv_A = nn.Conv2d(dim, dim*3, kernel_size=1, bias=False)
        self.qkv_B = nn.Conv2d(dim, dim*3, kernel_size=1, bias=False)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=False)

    def forward(self, x_A, x_B):
        b, c, h, w = x_A.shape

        qkv_A = self.qkv_A(x_A)
        qkv_B = self.qkv_B(x_B)
        
        q_A, k_A, v_A = qkv_A.chunk(3, dim=1)
        q_B, k_B, v_B = qkv_B.chunk(3, dim=1)

        q_A = rearrange(q_A, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_B = rearrange(k_B, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_B = rearrange(v_B, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_B = rearrange(q_B, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_A = rearrange(k_A, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_A = rearrange(v_A, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        # Cross attention A->B
        attn_A = (q_A @ k_B.transpose(-2, -1)) * self.temperature
        attn_A = attn_A.softmax(dim=-1)
        out_A = (attn_A @ v_B)

        # Cross attention B->A  
        attn_B = (q_B @ k_A.transpose(-2, -1)) * self.temperature
        attn_B = attn_B.softmax(dim=-1)
        out_B = (attn_B @ v_A)

        out_A = rearrange(out_A, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        out_B = rearrange(out_B, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)

        out_A = self.project_out(out_A)
        out_B = self.project_out(out_B)

        return out_A, out_B

class Fusion_Embed(nn.Module):
    def __init__(self, embed_dim):
        super(Fusion_Embed, self).__init__()
        self.conv = nn.Conv2d(embed_dim*2, embed_dim, 3, 1, 1)
        
    def forward(self, x_A, x_B):
        return self.conv(torch.cat([x_A, x_B], dim=1))

class AttentionFusion_Embed(nn.Module):
    def __init__(self, embed_dim):
        super(AttentionFusion_Embed, self).__init__()
        self.channel_attn = ChannelAttention(embed_dim*2, num_heads=8, bias=False)
        self.spatial_attn = OCAB(
            dim=embed_dim*2,
            window_size=8,
            overlap_ratio=0.5,
            num_heads=4,
            dim_head=32,
            bias=False
        )
        self.conv = nn.Conv2d(embed_dim*2, embed_dim, 3, 1, 1)
        
    def forward(self, x_A, x_B):
        concat = torch.cat([x_A, x_B], dim=1)
        out = self.channel_attn(concat) + self.spatial_attn(concat)
        return self.conv(out)

class DynamicFusion_Embed(nn.Module):
    def __init__(self, embed_dim):
        super(DynamicFusion_Embed, self).__init__()
        self.weight_net = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(embed_dim*2, embed_dim*2, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim*2, 2, 1),
            nn.Softmax(dim=1)
        )
        self.conv = nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        
    def forward(self, x_A, x_B):
        concat = torch.cat([x_A, x_B], dim=1)
        weights = self.weight_net(concat)
        x_fused = x_A * weights[:,0:1] + x_B * weights[:,1:2]
        return self.conv(x_fused)

class MultiscaleFusion_Embed(nn.Module):
    def __init__(self, embed_dim):
        super(MultiscaleFusion_Embed, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(embed_dim*2, embed_dim, 1),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(embed_dim*2, embed_dim, 3, 1, 1),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(embed_dim*2, embed_dim, 5, 1, 2),
            nn.ReLU(inplace=True)
        )
        self.fusion = nn.Conv2d(embed_dim*3, embed_dim, 1)
        
    def forward(self, x_A, x_B):
        concat = torch.cat([x_A, x_B], dim=1)
        out1 = self.branch1(concat)
        out2 = self.branch2(concat)
        out3 = self.branch3(concat)
        return self.fusion(torch.cat([out1, out2, out3], dim=1))
    
class ResidualFusion_Embed(nn.Module):
    def __init__(self, embed_dim):
        super(ResidualFusion_Embed, self).__init__()
        self.conv1 = nn.Conv2d(embed_dim*2, embed_dim, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 3, 1, 1)
        )
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self, x_A, x_B):
        concat = torch.cat([x_A, x_B], dim=1)
        identity = self.conv1(concat)
        out = self.conv2(identity)
        return self.relu(out + identity)

class CrossFusion_Embed(nn.Module):
    def __init__(self, embed_dim):
        super(CrossFusion_Embed, self).__init__()
        self.query_conv = nn.Conv2d(embed_dim, embed_dim, 1)
        self.key_conv = nn.Conv2d(embed_dim, embed_dim, 1)
        self.value_conv = nn.Conv2d(embed_dim, embed_dim, 1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, x_A, x_B):
        B, C, H, W = x_A.size()
        
        query = self.query_conv(x_A).view(B, -1, H*W).permute(0, 2, 1)
        key = self.key_conv(x_B).view(B, -1, H*W)
        energy = torch.bmm(query, key)
        attention = self.softmax(energy)
        
        value = self.value_conv(x_B).view(B, -1, H*W)
        out = torch.bmm(value, attention.permute(0, 2, 1))
        out = out.view(B, C, H, W)
        
        return x_A + self.gamma * out


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureWiseAffine, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, out_channels*2)
        )

    def forward(self, x, condition):
        B, _, H, W = x.shape
        style = self.fc(condition)
        gamma, beta = style.chunk(2, dim=1)
        gamma = gamma.view(B, -1, 1, 1)
        beta = beta.view(B, -1, 1, 1)
        return gamma * x + beta

class Cross_attentionSSA(nn.Module):
    def __init__(self, dim, window_size=8, overlap_ratio=0.5, 
                 num_spatial_heads=8, spatial_dim_head=16, bias=False):
        super(Cross_attentionSSA, self).__init__()
        
        self.spatial_attn_A = OCAB(
            dim=dim,
            window_size=window_size,
            overlap_ratio=overlap_ratio,
            num_heads=num_spatial_heads,
            dim_head=spatial_dim_head,
            bias=bias
        )
        
        self.spatial_attn_B = OCAB(
            dim=dim,
            window_size=window_size, 
            overlap_ratio=overlap_ratio,
            num_heads=num_spatial_heads,
            dim_head=spatial_dim_head,
            bias=bias
        )
        
        self.channel_attn_A = ChannelAttention(dim, num_heads=8, bias=bias)
        self.channel_attn_B = ChannelAttention(dim, num_heads=8, bias=bias)
        
        self.fusion_A = nn.Conv2d(dim*2, dim, 1, bias=bias)
        self.fusion_B = nn.Conv2d(dim*2, dim, 1, bias=bias)

    def forward(self, x_A, x_B):
        spatial_A = self.spatial_attn_A(x_A)
        spatial_B = self.spatial_attn_B(x_B)
        
        channel_A = self.channel_attn_A(x_A)
        channel_B = self.channel_attn_B(x_B)
        
        out_A = self.fusion_A(torch.cat([spatial_A, channel_B], dim=1))
        out_B = self.fusion_B(torch.cat([spatial_B, channel_A], dim=1))
        
        return out_A, out_B
