import torch
import torch.nn.functional as F
import numpy as np
import diff_operators
import modules


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

    print(loss)
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
        #         n_samples += torch.FloatTensor(N_neg).normal_(0 , 0.02)
        n_samples = n_samples.unsqueeze(-1).to(DEVICE)

        p_weight = emb_dic(idx_c).repeat(N_neg).reshape(N_neg, d)
        actions = a_t.repeat(N_neg).reshape(N_neg, 7)

        ef_input = torch.cat((p_weight, actions, n_samples), dim=1)
        ef_output = e_f(ef_input)
        n_tmp = torch.clip(ef_output, -5, 5)
        n = torch.sum(torch.exp(-n_tmp))

        loss = loss - torch.log(p / (n + p))

    print(loss)
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
    n_idx = idx_list[:N_neg]

    n = 0
    mean = torch.mean(emb_dic.weight)
    std = torch.std(emb_dic.weight)

    d = emb_dic.weight.size()[1]
    n_samples = torch.zeros((N_neg, d))
    for i in range(d):
        n_samples[:, i] = torch.FloatTensor(N_neg).uniform_(-0.15, 0.15)
        n_samples[:, i] += torch.FloatTensor(N_neg).normal_(0, 0.02)
    n_samples = n_samples.to(DEVICE)

    p_weight = emb_dic(idx_c).repeat(N_neg).reshape(N_neg, d)
    #     n_samples = torch.FloatTensor(N_neg, d).uniform_(-0.15 , 0.15).to(DEVICE)

    #     p_weight = emb_dic(idx_c).repeat(N_neg).reshape(N_neg,d)
    #     n_samples = torch.FloatTensor(N_neg, 1).uniform_(-0.1 , 0.1)

    #     n_samples += p_weight

    actions = a_t.repeat(N_neg).reshape(N_neg, 7)
    ef_input = torch.cat((p_weight, n_samples, actions), dim=1)
    n_tmp = torch.clip(energy_function(ef_input)[0], -5, 5)
    n = torch.sum(torch.exp(-n_tmp))

    # #     print(idx_c, idx_f,n_idx)
    #     for idx, cnt_idx in enumerate(n_idx):
    # #         n_tmp = energy_function(torch.cat([ emb_dic(idx_c) , emb_dic( torch.tensor(cnt_idx).to(DEVICE)), a_t], axis=0))
    #         n_tmp = energy_function(torch.cat([ emb_dic(idx_c) , n_samples[idx,:].to(DEVICE), a_t], axis=0))
    #         n += torch.exp(-n_tmp)

    #     p = torch.clip(p, 1e-4, 1e4)
    #     n = torch.clip(n, 1e-4, 1e4)

    loss = -torch.log(p / (n + p))

    #     loss =  (n + p) / p
    print(n, p, n_tmp, p_, loss)
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


def hyper_loss(model_output, gt_sdf, gt_normals, ks, ki, kn, kg):
    gt_sdf = gt_sdf.unsqueeze(0).float()
    gt_normals = gt_normals.unsqueeze(0).float()

    coords = model_output["model_in"].float()
    pred_sdf = model_output["model_out"].float()

    sdf_constraint = torch.where(
        gt_sdf == 0, torch.abs(pred_sdf), torch.zeros_like(pred_sdf)
    )

    pred_sdf_c = torch.clip(pred_sdf, -0.3, 0.3).float()
    gt_sdf_c = torch.clip(gt_sdf, -0.3, 0.3).float()

    inter_constraint = torch.where(
        gt_sdf == 0, torch.zeros_like(pred_sdf), abs(gt_sdf_c - pred_sdf_c)
    )
    gradient = diff_operators.gradient(pred_sdf, coords)

    norm = (1 - F.cosine_similarity(gradient, gt_normals, dim=-1))[..., None]
    normal_constraint = torch.where(
        gt_sdf == 0, norm, torch.zeros_like(gradient[..., :1])
    )

    grad_constraint = abs(1 - torch.clip(torch.linalg.norm(gradient, dim=-1), 0, 1))

    return {
        "sdf": torch.abs(sdf_constraint).mean() * ks,  # 1e5 # 1e4      # 3e3
        "inter": inter_constraint.mean() * ki,  # 1e2                   # 1e3
        "normal_constraint": normal_constraint.mean() * kn,  # 1e2 ####YS
        "grad_constraint": grad_constraint.mean() * kg,
    }  # 1e1      # 5e1


def hyper_loss_deform(model_output, gt, kl, fw, ks, ki, kn, kg):
    gt_sdf = gt["sdf"]

    coords = model_output["model_in"]
    pred_sdf = model_output["model_out"]

    sdf_constraint = torch.where(
        gt_sdf == 0, torch.abs(pred_sdf), torch.zeros_like(pred_sdf)
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
            gt_sdf == 0, norm, torch.zeros_like(gradient[..., :1])
        )
        grad_constraint = abs(1 - torch.linalg.norm(gradient, dim=-1))
    else:
        normal_constraint = torch.tensor([0]).float()
        grad_constraint = torch.tensor([0]).float()
    # print(grad_constraint)

    # Exp      # Lapl
    # -----------------
    return {
        "latent_loss": kl * latent_loss(model_output),
        "hypo_weight_loss": fw * hypo_weight_loss(model_output),
        "sdf": torch.abs(sdf_constraint).mean() * ks,  # 1e5 # 1e4      # 3e3
        "inter": inter_constraint.mean() * ki,  # 1e2                   # 1e3
        "normal_constraint": normal_constraint.mean() * kn,  # 1e2 ####YS
        "grad_constraint": grad_constraint.mean() * kg,
    }  # 1e1      # 5e1
