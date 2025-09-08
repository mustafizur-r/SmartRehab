import warnings
warnings.filterwarnings("ignore")

import os
from os.path import join as pjoin

import torch
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from model.vq.rvq_model import HRVQVAE
from model.transformer.transformer import MoMaskPlus
from model.cnn_networks import GlobalRegressor
from config.load_config import load_config

from utils.fixseeds import fixseed
from utils import bvh_io
from utils.motion_process_bvh import process_bvh_motion, recover_bvh_from_rot
from utils.utils import plot_3d_motion
from utils.paramUtil import kinematic_chain
from common.skeleton import Skeleton
import collections
from common.animation import Animation
from einops import rearrange, repeat
from rest_pose_retarget import RestPoseRetargeter

import numpy as np

def inv_transform(data):
    if isinstance(data, np.ndarray):
        return data * std[:data.shape[-1]] + mean[:data.shape[-1]]
    elif isinstance(data, torch.Tensor):
        return data * torch.from_numpy(std[:data.shape[-1]]).float().to(
            data.device
        ) + torch.from_numpy(mean[:data.shape[-1]]).float().to(data.device)
    else:
        raise TypeError("Expected data to be either np.ndarray or torch.Tensor")


def forward_kinematic_func(data):
    motions = inv_transform(data)
    b, l, _ = data.shape
    # print(data.shape)
    global_quats, local_quats, r_pos = recover_bvh_from_rot(motions, cfg.data.joint_num, skeleton, keep_shape=False)
    _, global_pos = skeleton.fk_local_quat(local_quats, r_pos)
    global_pos = rearrange(global_pos, '(b l) j d -> b l j d', b = b)
    local_quats = rearrange(local_quats, '(b l) j d -> b l j d', b = b)
    r_pos = rearrange(r_pos, '(b l) d -> b l d', b = b)
    return global_pos, local_quats, r_pos


def load_vq_model(vq_cfg, device):

    vq_model = HRVQVAE(vq_cfg,
            vq_cfg.data.dim_pose,
            vq_cfg.model.down_t,
            vq_cfg.model.stride_t,
            vq_cfg.model.width,
            vq_cfg.model.depth,
            vq_cfg.model.dilation_growth_rate,
            vq_cfg.model.vq_act,
            vq_cfg.model.use_attn,
            vq_cfg.model.vq_norm)

    ckpt = torch.load(pjoin(vq_cfg.exp.root_ckpt_dir, vq_cfg.data.name, 'vq', vq_cfg.exp.name, 'model',mask_trans_cfg.vq_ckpt),
                            map_location=device, weights_only=True)
    model_key = 'vq_model' if 'vq_model' in ckpt else 'model'
    vq_model.load_state_dict(ckpt[model_key])
    print(f'Loading VQ Model {vq_cfg.exp.name} from epoch {ckpt["ep"]}')
    vq_model.to(device)
    vq_model.eval()
    return vq_model


def load_trans_model(t2m_cfg, which_model, device):
    t2m_transformer = MoMaskPlus(
        code_dim=t2m_cfg.vq.code_dim,
        latent_dim=t2m_cfg.model.latent_dim,
        ff_size=t2m_cfg.model.ff_size,
        num_layers=t2m_cfg.model.n_layers,
        num_heads=t2m_cfg.model.n_heads,
        dropout=t2m_cfg.model.dropout,
        text_dim=t2m_cfg.text_embedder.dim_embed,
        cond_drop_prob=t2m_cfg.training.cond_drop_prob,
        device=device,
        cfg=t2m_cfg,
        full_length=t2m_cfg.data.max_motion_length//4,
        scales=vq_cfg.quantizer.scales
    )
    ckpt = torch.load(pjoin(t2m_cfg.exp.root_ckpt_dir, t2m_cfg.data.name, "momask_plus", t2m_cfg.exp.name, 'model', which_model),
                      map_location=cfg.device, weights_only=True)
    if isinstance(ckpt["t2m_transformer"], collections.OrderedDict):
        t2m_transformer.load_state_dict(ckpt["t2m_transformer"])
    else:
        t2m_transformer.load_state_dict(ckpt["t2m_transformer"].state_dict())
    t2m_transformer.to(device)
    t2m_transformer.eval()
    print(f'Loading Mask Transformer {t2m_cfg.exp.name} from epoch {ckpt["ep"]}!')
    return t2m_transformer


def load_gmr_model(device):
    gmr_cfg = load_config(pjoin("checkpoint_dir/snapmogen/gmr", "gmr_d292", 'gmr.yaml'))
    gmr_cfg.exp.checkpoint_dir = pjoin(gmr_cfg.exp.root_ckpt_dir, gmr_cfg.data.name, 'gmr', gmr_cfg.exp.name)
    gmr_cfg.exp.model_dir = pjoin(gmr_cfg.exp.checkpoint_dir, 'model')
    regressor = GlobalRegressor(dim_in=gmr_cfg.data.dim_pose-2, dim_latent=512, dim_out=2)
    ckpt = torch.load(pjoin(gmr_cfg.exp.model_dir, 'best.tar'), map_location=device)
    regressor.load_state_dict(ckpt['regressor'])
    regressor.eval()
    regressor.to(device)
    return regressor

def load_texts_from_file(path, device):
    """
    Expected line format per line:
      original text # rewritten text # length
    Example:
      A person walks in a circular path. # The person moves steadily around in a smooth circular route. #268
    If length is missing, it will default to randint(7..9)*30.
    """
    texts, rewrites, lengths = [], [], []

    if not os.path.exists(path):
        raise FileNotFoundError(f"text description file not found: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = [p.strip() for p in line.split("#")]
            if len(parts) < 2:
                continue  # skip malformed lines
            orig = parts[0]
            rewrite = parts[1]
            length = int(parts[2]) if len(parts) >= 3 and parts[2].isdigit() else None
            texts.append(orig)
            rewrites.append(rewrite)
            lengths.append(length)

    if not texts:
        raise ValueError("No valid lines found in text_descriptions.txt")

    # Fill missing lengths with randint(7..9)*30
    for i, L in enumerate(lengths):
        if L is None:
            lengths[i] = int(torch.randint(7, 10, (1,)).item()) * 30

    m_lengths = torch.tensor(lengths, dtype=torch.long, device=device)
    return texts, rewrites, m_lengths


if __name__ == '__main__':

    cfg = load_config("./config/eval_momaskplus.yaml")
    fixseed(cfg.seed)
    retargeter = RestPoseRetargeter()

    if cfg.device != 'cpu':
        torch.cuda.set_device(cfg.device)
    device = torch.device(cfg.device)
    torch.autograd.set_detect_anomaly(True)

    cfg.checkpoint_dir = pjoin(cfg.root_ckpt_dir, cfg.data.name, 'momask_plus', cfg.mask_trans_name)
    cfg.model_dir = pjoin(cfg.checkpoint_dir, 'model')
    cfg.gen_dir = pjoin(cfg.checkpoint_dir, 'gen', cfg.ext)
    meta_dir = pjoin(cfg.data.root_dir, 'meta_data')

    # os.makedirs(cfg.gen_dir, exist_ok=True)
    # os.makedirs(pjoin(cfg.gen_dir, 'bvh'), exist_ok=True)
    # os.makedirs(pjoin(cfg.gen_dir, 'mp4'), exist_ok=True)
    bvh_out_dir = "./bvh_folder"
    mp4_out_dir = "./video_result"
    os.makedirs(bvh_out_dir, exist_ok=True)
    os.makedirs(mp4_out_dir, exist_ok=True)

    mask_trans_cfg = load_config(pjoin(cfg.root_ckpt_dir, cfg.data.name, 'momask_plus', cfg.mask_trans_name, 'train_momaskplus.yaml'))

    vq_cfg = load_config(pjoin(cfg.root_ckpt_dir, cfg.data.name, 'vq', mask_trans_cfg.vq_name, 'residual_vqvae.yaml'))
    mask_trans_cfg.vq = vq_cfg.quantizer
    # res_trans_cfg.vq = vq_cfg.quantizer

    vq_model = load_vq_model(vq_cfg, device)
    gmr_model = load_gmr_model(device)

    cfg.data.feat_dir = pjoin(cfg.data.root_dir, 'renamed_feats')
    meta_dir = pjoin(cfg.data.root_dir, 'meta_data')
    data_split_dir = pjoin(cfg.data.root_dir, 'data_split_info1')
    all_caption_path = pjoin(cfg.data.root_dir, 'all_caption_clean.json')

    test_mid_split_file = pjoin(data_split_dir, 'test_fnames.txt')
    test_cid_split_file = pjoin(data_split_dir, 'test_ids.txt')

    mean = np.load(pjoin(meta_dir, 'mean.npy'))
    std = np.load(pjoin(meta_dir, 'std.npy'))

    
    template_anim = bvh_io.load(pjoin(cfg.data.root_dir, 'renamed_bvhs', 'm_ep2_00086.bvh'))
    skeleton = Skeleton(template_anim.offsets, template_anim.parents, device=device)

    t2m_transformer = load_trans_model(mask_trans_cfg, cfg.which_epoch, device)

    num_results = 0

    # f = open(pjoin(cfg.gen_dir, 'text_descriptions.txt'), 'a+')
    # animate_gt = False
    #
    # texts = [
    #     "A person walks in a circular path.",
    # ]
    #
    # rewrite_texts = [
    #     "The person moves steadily around in a smooth circular route.",
    # ]
    #
    # m_lengths = torch.randint(7, 10, (len(rewrite_texts),)) * 30
    #
    # # print(len(rewrite_texts), len(m_lengths))
    # assert len(rewrite_texts) == len(m_lengths)
    #
    # m_lengths = m_lengths.to(device).long().detach()

    txt_path = "./input.txt"

    texts, rewrite_texts, m_lengths = load_texts_from_file(txt_path, device)

    mids = t2m_transformer.generate(rewrite_texts, m_lengths//4, cfg.time_steps[0], cfg.cond_scales[0], 
                                    temperature=1)
    pred_motions = vq_model.forward_decoder(mids, m_lengths)
    gen_global_pos, gen_local_quat, gen_r_pos = forward_kinematic_func(pred_motions)


    for k in range(len(texts)):
        print("--->", k , "<---")
        print("user prompt: ", texts[k])
        print("gpt prompt: ", rewrite_texts[k])

        gen_anim = Animation(gen_local_quat[k, :m_lengths[k]].detach().cpu().numpy(), 
                    repeat(gen_r_pos[k, :m_lengths[k]].detach().cpu().numpy(), 'i j -> i k j', k=len(template_anim)),
                        template_anim.orients, 
                        template_anim.offsets, 
                        template_anim.parents, 
                        template_anim.names, 
                        template_anim.frametime)
        
        feats = process_bvh_motion(None, 30, 30, 0.11, shift_one_frame=True, animation=gen_anim)

        feats = (feats - mean) / std
        # print(feats.shape)
        feats = torch.from_numpy(feats).unsqueeze(0).float().to(device)
        # print(feats.shape)
        gmr_input = torch.cat([feats[..., 0:1], feats[..., 3:cfg.data.dim_pose-4]], dim=-1)
        # print(gmr_input.shape)
        gmr_output = gmr_model(gmr_input)
        rec_feats = torch.cat([feats[..., 0:1], gmr_output, feats[..., 3:]], dim=-1)
        # rec_feats = inv_transform(rec_feats)

        single_gen_global_pos, single_gen_local_quat, single_gen_r_pos = forward_kinematic_func(rec_feats)

        new_anim = Animation(single_gen_local_quat[0].detach().cpu().numpy(), 
                    repeat(single_gen_r_pos[0].detach().cpu().numpy(), 'i j -> i k j', k=len(template_anim)),
                        template_anim.orients, 
                        template_anim.offsets, 
                        template_anim.parents, 
                        template_anim.names, 
                        template_anim.frametime)
        
        single_gen_motion = single_gen_global_pos[0].detach().cpu().numpy()
        # gen_bvh_path = pjoin(cfg.gen_dir, 'bvh', f"{num_results}_gen.bvh")
        gen_bvh_path = pjoin(bvh_out_dir, f"bvh_0_out.bvh")
        bvh_io.save(gen_bvh_path, 
                    retargeter.rest_pose_retarget(new_anim, tgt_rest='A'),
                    names=new_anim.names, frametime=new_anim.frametime, order='xyz', quater=True)
    
        # f.write(f"{num_results}: {texts[k]} # {rewrite_texts[k]} #{m_lengths[k]}\n")

        if cfg.gen.animate:
            real_anim_path = pjoin(mp4_out_dir, f"{num_results}_gt.mp4")
            gen_anim_path = pjoin(mp4_out_dir, f"{num_results}_gen.mp4")
            plot_3d_motion(gen_anim_path, kinematic_chain, single_gen_motion, title=texts[k], fps=30, radius=100)
        num_results += 1
        print("%d/%d"%(num_results, cfg.gen.num_samples))

    # f.close()


