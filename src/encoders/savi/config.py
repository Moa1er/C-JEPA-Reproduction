"""Construction-time config for the released SlotFormer StoSAVi-on-CLEVRER
checkpoint (`stosavi_clevrer_params/model_12.pth`).

Copied verbatim from SlotFormer's `slotformer/base_slots/configs/
stosavi_clevrer_params.py`. Three values diverge from `StoSAVi.__init__`'s
own defaults — these matter because the released checkpoint was trained
with these exact settings:
    slot_dict.kernel_mlp = False   (smaller kernel_dist_layer)
    pred_dict.pred_type  = 'mlp'   (no transformer predictor)
    pred_dict.pred_rnn   = False   (no LSTM wrapper)
"""

STOSAVI_CLEVRER_CFG = dict(
    resolution=(64, 64),
    clip_len=6,
    slot_dict=dict(
        num_slots=7,
        slot_size=128,
        slot_mlp_size=256,
        num_iterations=2,
        kernel_mlp=False,
    ),
    enc_dict=dict(
        enc_channels=(3, 64, 64, 64, 64),
        enc_ks=5,
        enc_out_channels=128,
        enc_norm='',
    ),
    dec_dict=dict(
        dec_channels=(128, 64, 64, 64, 64),
        dec_resolution=(8, 8),
        dec_ks=5,
        dec_norm='',
    ),
    pred_dict=dict(
        pred_type='mlp',
        pred_rnn=False,
        pred_norm_first=True,
        pred_num_layers=2,
        pred_num_heads=4,
        pred_ffn_dim=512,
        pred_sg_every=None,
    ),
    loss_dict=dict(
        use_post_recon_loss=True,
        kld_method='var-0.01',
    ),
)
