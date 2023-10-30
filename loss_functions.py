import torch
import torch.nn.functional as F
import numpy as np
import diff_operators


def latent_loss(emb):
    return torch.mean(emb**2)


def info_nce_force_emb_reg(
    energy_function_list, N_neg, force_emb_, force_emb, a_t, DEVICE="cuda"
):

    loss = 0
    force_emb = force_emb.squeeze()
    force_emb_ = force_emb_.squeeze()

    d = len(energy_function_list)
    for d_i, e_f in enumerate(energy_function_list):

        p_ = e_f(torch.cat([force_emb_, a_t, force_emb[: d_i + 1]], axis=0))
        p_ = torch.clip(p_, -2, 2)
        p = torch.exp(-p_)

        n_samples = torch.FloatTensor(N_neg, d_i + 1).uniform_(-0.3, 0.3)
        n_samples = n_samples.to(DEVICE)

        p_weight = force_emb.repeat(N_neg).reshape(N_neg, d)

        actions = a_t.repeat(N_neg).reshape(N_neg, 7)
        ef_input = torch.cat((p_weight, actions, n_samples), dim=1)

        n = torch.clip(e_f(ef_input)[0], -5, 5)
        n = torch.sum(torch.exp(-n))

        loss = loss - torch.log(p / (n + p))[0]

    return loss


def info_nce_reg(
    energy_function_list, N_neg, emb_dic, idx_list, idx_c, idx_f, a_t, DEVICE="cuda"
):

    d = len(energy_function_list)
    loss = 0
    for d_, e_f in enumerate(energy_function_list):
        y_i = emb_dic(idx_f)[d_].reshape(-1)
        p_ = e_f(torch.cat([emb_dic(idx_c), a_t, y_i], axis=0))
        p_ = torch.clip(p_, -5, 5)
        p = torch.sum(torch.exp(-p_))

        n_samples = torch.FloatTensor(N_neg).uniform_(-0.010, 0.010)
        n_samples = n_samples.unsqueeze(-1).to(DEVICE)

        p_weight = emb_dic(idx_c).repeat(N_neg).reshape(N_neg, d)
        actions = a_t.repeat(N_neg).reshape(N_neg, 7)

        ef_input = torch.cat((p_weight, actions, n_samples), dim=1)
        ef_output = e_f(ef_input)
        n_tmp = torch.clip(ef_output, -5, 5)
        n = torch.sum(torch.exp(-n_tmp))

        loss = loss - torch.log(p / (n + p))

    return loss


def info_nce(
    energy_function, N_neg, emb_dic, idx_list, idx_c, idx_f, a_t, DEVICE="cuda"
):
    p_ = energy_function(torch.cat([emb_dic(idx_c), emb_dic(idx_f), a_t], axis=0))

    p_ = torch.clip(p_, -5, 5)
    p = torch.exp(-p_)

    idx_list = np.delete(
        idx_list, np.where(idx_list == int(idx_f) * np.ones_like(idx_list))
    )
    np.random.shuffle(idx_list)

    n = 0
    d = emb_dic.weight.size()[1]
    n_samples = torch.zeros((N_neg, d))
    for i in range(d):
        n_samples[:, i] = torch.FloatTensor(N_neg).uniform_(-0.15, 0.15)
        n_samples[:, i] += torch.FloatTensor(N_neg).normal_(0, 0.02)
    n_samples = n_samples.to(DEVICE)

    p_weight = emb_dic(idx_c).repeat(N_neg).reshape(N_neg, d)

    actions = a_t.repeat(N_neg).reshape(N_neg, 7)
    ef_input = torch.cat((p_weight, n_samples, actions), dim=1)
    n_tmp = torch.clip(energy_function(ef_input)[0], -5, 5)
    n = torch.sum(torch.exp(-n_tmp))

    loss = -torch.log(p / (n + p))

    return loss[0]


def emb_action_loss(emb1, emb2):
    loss = torch.abs(emb1 - emb2).mean()
    return loss


def function_mse(model_output, gt):
    return {"func_loss": ((model_output["model_out"] - gt["func"]) ** 2).mean()}


def gradients_mse(model_output, gt):
    # compute gradients on the model
    gradients = diff_operators.gradient(
        model_output["model_out"], model_output["model_in"]
    )
    # compare them with the ground-truth
    gradients_loss = torch.mean((gradients - gt["gradients"]).pow(2).sum(-1))
    return {"gradients_loss": gradients_loss}


def hypo_weight_loss(model_output):
    weight_sum = 0
    total_weights = 0

    for weight in model_output["hypo_params"].values():
        weight_sum += torch.sum(weight**2)
        total_weights += weight.numel()

    return weight_sum * (1 / total_weights)


def hyper_loss(model_output, gt_sdf, ks, ki, kg, gt_normals=None, kn=None, eps=0.0):
    losses = {}
    
    gt_sdf = gt_sdf.unsqueeze(0).float()

    if gt_normals is not None:
        gt_normals = gt_normals.unsqueeze(0).float()

    coords = model_output["model_in"].float()
    pred_sdf = model_output["model_out"].float()

    sdf_constraint = torch.where(
        torch.abs(gt_sdf) <= eps, torch.abs(pred_sdf), torch.zeros_like(pred_sdf)
    )
    losses['sdf'] = torch.abs(sdf_constraint).mean() * ks

    pred_sdf_c = torch.clip(pred_sdf, -0.3, 0.3).float()
    gt_sdf_c = torch.clip(gt_sdf, -0.3, 0.3).float()

    inter_constraint = torch.where(
        torch.abs(gt_sdf) <= eps, torch.zeros_like(pred_sdf), abs(gt_sdf_c - pred_sdf_c)
    )
    losses['inter'] = inter_constraint.mean() * ki

    gradient = diff_operators.gradient(pred_sdf, coords)

    if gt_normals is not None:
        norm = (1 - F.cosine_similarity(gradient, gt_normals, dim=-1))[..., None]
        normal_constraint = torch.where(
            torch.abs(gt_sdf) <= eps, norm, torch.zeros_like(gradient[..., :1])
        )
        losses['normal_constraint'] = normal_constraint.mean() * kn

    grad_constraint = abs(1 - torch.clip(torch.linalg.norm(gradient, dim=-1), 0, 1))
    losses['grad_constraint'] = grad_constraint.mean() * kg

    return losses


def hyper_loss_deform(model_output, gt, kl, fw, ks, ki, kn, kg, eps=0.0):
    gt_sdf = gt["sdf"]

    coords = model_output["model_in"]
    pred_sdf = model_output["model_out"]

    sdf_constraint = torch.where(
        torch.abs(gt_sdf) <= eps, torch.abs(pred_sdf), torch.zeros_like(pred_sdf)
    )
    pred_sdf_c = torch.clip(pred_sdf, -0.3, 0.3)
    gt_sdf_c = torch.clip(gt_sdf, -0.3, 0.3)
    inter_constraint = torch.where(
        gt_sdf == 0, torch.zeros_like(pred_sdf), abs(gt_sdf_c - pred_sdf_c)
    )

    if kn != 0:
        gradient = diff_operators.gradient(pred_sdf, coords)
        gt_normals = gt["normals"]
        norm = (1 - F.cosine_similarity(gradient, gt_normals, dim=-1))[..., None]

        normal_constraint = torch.where(
            torch.abs(gt_sdf) <= eps, norm, torch.zeros_like(gradient[..., :1])
        )
        grad_constraint = abs(1 - torch.linalg.norm(gradient, dim=-1))
    else:
        normal_constraint = torch.tensor([0]).float()
        grad_constraint = torch.tensor([0]).float()

    return {
        "latent_loss": kl * latent_loss(model_output),
        "hypo_weight_loss": fw * hypo_weight_loss(model_output),
        "sdf": torch.abs(sdf_constraint).mean() * ks,
        "inter": inter_constraint.mean() * ki,
        "normal_constraint": normal_constraint.mean() * kn,
        "grad_constraint": grad_constraint.mean() * kg,
    }
