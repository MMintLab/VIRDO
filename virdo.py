import os

from tqdm.autonotebook import tqdm
from tqdm import trange
import torch
<<<<<<< HEAD
from pytorch3d.loss import chamfer_distance
=======
>>>>>>> no lfs

from utilities.train_util import validation_3d, make_dir
import loss_functions, modules
from utilities.sdf_meshing import create_mesh
import modules, diff_operators, meta_modules

## Training Parameters
lr = 3e-4
epochs_til_ckpt = 5000


class ObjectDecoder(torch.nn.Module):
    def __init__(self, shape_model, shape_embedding, device):
        super().__init__()

        self.shape_embedding = shape_embedding.to(device)
        self.shape_model = shape_model

    def forward(self, coords):
        shape_input = {"coords": coords, "embedding": self.shape_embedding}
        pred_sdf = self.shape_model(shape_input)
        return pred_sdf


class VirdoModule:
    def __init__(self, data, DEVICE="cuda"):
        self.num_shape = len(data)
        self.device = DEVICE

        ## Params
        self.data = data
        self.object_code_size = 6
        self.contact_emb_size = 4
        self.deformation_code_size = 12

        self.object_code = (
            torch.nn.Embedding(self.num_shape, self.object_code_size)
            .requires_grad_(True)
            .to(self.device)
        )
        torch.nn.init.normal_(self.object_code.weight, mean=0.0, std=0.1)

        self.object_model = meta_modules.virdo_hypernet(
            in_features=3, out_features=1, hyper_in_features=self.object_code_size, hl=2
        )
        self.object_model.to(DEVICE)
        self.object_model.float()

<<<<<<< HEAD

=======
        ## Deformation Module form pretrained

        self.deformation_module = meta_modules.virdo_hypernet(
            in_features=3,
            out_features=3,
            hyper_in_features=self.deformation_code_size + self.object_code_size,
            hl=1,
        )

        self.classifier_model = modules.PointNetCls(
            d_cnt_code=self.contact_emb_size, d_force_emb=self.deformation_code_size
        ).to(DEVICE)
>>>>>>> no lfs

    def pretraining_result(self, pretrained_path, save_dir):
        self.from_pretraining(pretrained_path)
        make_dir(save_dir)

        self.object_model.eval()
        object_code_weight = self.object_code.weight
        for i in range(len(object_code_weight)):
            obj_code_i = object_code_weight[i]
            decoder = ObjectDecoder(
                self.object_model,
                obj_code_i,
                self.device,
            )
            create_mesh(
                decoder,
                os.path.join(save_dir, f"nominal_dense_{i}"),
                N=400,
                verbose=False,
            )

    def from_pretraining(self, pretrained_path):

        object_module_log = torch.load(
            pretrained_path, map_location=torch.device("cpu")
        )
        object_code_weight = (
            object_module_log["shape_embedding"].float().to(self.device)
        )
        object_module_weight = object_module_log["shape_model"]

        self.object_code = torch.nn.Embedding.from_pretrained(object_code_weight)

        try:
            self.optims.load_state_dict(object_module_log["optimizer_state_dict"])
        except:
            pass

        self.object_model.load_state_dict(object_module_weight)
        self.object_model.to(self.device)
        self.epoch = object_module_log["epoch"]

    def pretraining(self, checkpoints_dir, end_epoch=10001):
        make_dir(checkpoints_dir)
        start_epoch = 0
        self.optims = torch.optim.Adam(
            [
                {"params": self.object_code.parameters(), "lr": 1e-4},
                {"params": self.object_model.parameters(), "lr": 1e-4},
            ]
        )
        ## If checkpoint directory exists,
        if os.path.exists(os.path.join(checkpoints_dir, "shape_latest.pth")):
            self.from_pretraining(os.path.join(checkpoints_dir, "shape_latest.pth"))
            start_epoch = self.epoch
            print("log loaded from epoch ", start_epoch)

        for epoch in trange(start_epoch, end_epoch):
            tot_loss = 0
            cd_tot = 0
            for shape_idx, batch in self.data.items():
                data_nom = batch["nominal"]
                shape_idx_ = torch.tensor(shape_idx).to(self.device)

                # Train nominal Shape
                shape_input_nom = {
                    "coords": data_nom["coords"].to(self.device).float().unsqueeze(0),
                    "embedding": self.object_code(shape_idx_).float().unsqueeze(0),
                }

                shape_nom_output = self.object_model(shape_input_nom)
                nom_sdf_loss = loss_functions.hyper_loss(
                    model_output=shape_nom_output,
                    gt_sdf=data_nom["gt"].to(self.device),
                    gt_normals=data_nom["normals"].to(self.device),
                    ks=6e6,
                    ki=3e4,
                    kn=5e1,
                    kg=0,
                )

                nom_sdf_losses = 0
                for loss_name, loss in nom_sdf_loss.items():
                    single_loss = loss.mean()
                    nom_sdf_losses += single_loss

                hypo_losses = loss_functions.hypo_weight_loss(shape_nom_output)
                emb_losses = loss_functions.latent_loss(self.object_code(shape_idx_))

                train_loss = nom_sdf_losses + 1e2 * hypo_losses + 1e2 * emb_losses

                ## Update
                self.optims.zero_grad()
                train_loss.backward()
                self.optims.step()
                tot_loss += train_loss.detach()

                ## Save the model when regression succeed
                if epoch % 100 == 0:
                    decoder = ObjectDecoder(
                        self.object_model,
                        shape_input_nom["embedding"],
                        self.device,
                    )
                    cd = validation_3d(data_nom, decoder)
                    print(cd)
                    if cd == "nan":
                        cd = 1
                    cd_tot += cd

            if cd_tot != 0 and cd_tot < 0.005:
                print("cd tot :", cd_tot)
                torch.save(
                    {
                        "epoch": epoch,
                        "shape_model": self.object_model.state_dict(),
                        "shape_embedding": self.object_code.weight.data,
                        "optimizer_state_dict": self.optims.state_dict(),
                    },
                    os.path.join(checkpoints_dir, f"shape_{epoch}.pth"),
                )
                return

            if not epoch % 100:
                tqdm.write(
                    "Epoch %d, tot_loss %0.6f, train loss %0.6f"
                    % (epoch, tot_loss, train_loss)
                )
                torch.save(
                    {
                        "epoch": epoch,
                        "shape_model": self.object_model.state_dict(),
                        "shape_embedding": self.object_code.weight.data,
                        "optimizer_state_dict": self.optims.state_dict(),
                    },
                    os.path.join(checkpoints_dir, f"shape_latest.pth"),
                )
<<<<<<< HEAD


    def from_maintraining(self, main_trained_dir):

        object_module_log = torch.load( os.path.join(main_trained_dir, 'shape_latest.pth'), map_location=torch.device("cpu"))
        main_module_log = torch.load( os.path.join(main_trained_dir, 'main_latest.pth'), map_location=torch.device("cpu"))

        object_code_weight = (
            object_module_log["shape_embedding"].float().to(self.device)
        )
        object_module_weight = object_module_log["shape_model"]

        self.object_code = torch.nn.Embedding.from_pretrained(object_code_weight)


        self.object_model.load_state_dict(object_module_weight)
        self.object_model.to(self.device)


        self.epoch = main_module_log["epoch"]

        ## Load deformation module
        deform_model_weight = main_module_log["deform_model"]
        self.deformation_model.load_state_dict(deform_model_weight)

        ## Load force module
        force_model_weight = main_module_log["force_model"]
        self.force_model.load_state_dict(force_model_weight)

        ## Load Optimizer
        self.optims.load_state_dict(main_module_log["optimizer_state_dict"])

    def maintraining(self, pretrained_path, checkpoints_dir, end_epoch = 6001):

        ## Initialize Modules & optimizers
        self.from_pretraining(pretrained_path)
        self.force_model = modules.PointNetCls(
                                                d_cnt_code=self.contact_emb_size,
                                                d_force_emb=self.deformation_code_size).to(self.device)
        self.deformation_model = meta_modules.virdo_hypernet(
                                                in_features=3,
                                                out_features=3,
                                                hyper_in_features=self.deformation_code_size + self.object_code_size,
                                                hl=1,
                                                ).to(self.device)

        self.optims = torch.optim.Adam([
            {'params': self.force_model.parameters(), 'lr': 5e-5},
            {'params': self.deformation_model.parameters(), 'lr': 5e-5}
        ])


        ## Directories
        start_epoch = 0
        checkpoints_dir = os.path.join(checkpoints_dir)
        make_dir(checkpoints_dir)
        if os.path.exists(os.path.join(checkpoints_dir, "main_latest.pth")):
            self.from_maintraining(checkpoints_dir)
            start_epoch = self.epoch
            print("log loaded from epoch ", start_epoch)


        ## losses
        loss_sdf = loss_functions.hyper_loss
        loss_hypo_weight = loss_functions.hypo_weight_loss
        loss_emb = loss_functions.latent_loss


        for epoch in trange(start_epoch, end_epoch):
            tot_loss = 0
            feats = {'f_emb': {}, 'cnt_ft' : {}}
            cd_hist = {}

            for shape_idx in range ( len(self.data) ):
                shape_idx_ = torch.tensor(shape_idx).to(self.device)
                cd_hist[shape_idx] = []
                feats['f_emb'][shape_idx] = {}
                feats['cnt_ft'][shape_idx] = {}

                for deform_idx in self.data[shape_idx].keys():
                    if deform_idx == "nominal":
                        continue

                    data_def = self.data[shape_idx][deform_idx]
                    data_nom = self.data[shape_idx]['nominal']
                    shape_emb = self.object_code(shape_idx_).unsqueeze(0)


                    reaction_force = data_def['reaction'][:,:3].float().to(self.device)
                    contact_points = data_def['contact'].to(self.device)

                    feat = self.force_model(contact_points.transpose(2, 1), reaction_force, epoch) # batchsize x points
                    deform_input = {'coords': data_def['coords'].to(self.device),
                                  'embedding':torch.cat([
                                      feat, shape_emb
                                  ], dim = -1)}


                    deform_output = self.deformation_model(deform_input)

                    shape_input_def = {'coords':deform_output['model_in'],
                    'model_out' : deform_output['model_in'] + deform_output['model_out'],
                    'embedding' : shape_emb}


                    # Save Features
                    feats['cnt_ft'][shape_idx][deform_idx] = self.force_model.cnt_ft.detach().cpu()
                    feats['f_emb'][shape_idx][deform_idx] = feat.detach().cpu()
                    shape_def_output = self.object_model(shape_input_def)


                    ## Loss 1 : signed distance loss
                    deform_sdf_loss = loss_sdf(model_output= shape_def_output,
                                           gt_sdf= data_def['gt'].to(self.device),
                                           gt_normals= data_def['normals'].to(self.device),
                                           ks=1e6, ki=3e4, kn=5e1, kg=0)
                    deform_sdf_losses = 0
                    for loss_name, loss in deform_sdf_loss.items():
                        single_loss = loss.mean()
                        deform_sdf_losses += single_loss


                    ## Loss 2 : correspondence loss
                    def_on_surf_idx = torch.where(data_def['gt'] == 0)[1]
                    nom_on_surf_idx = torch.where(data_nom['gt'] == 0)[1]
                    cd_loss = chamfer_distance(shape_input_def['model_out'][:,def_on_surf_idx ,:],
                                                     data_nom['coords'][:,nom_on_surf_idx ,:].to(self.device))[0]
                    deform_loss = torch.norm(deform_output['model_out'])
                    cor_loss =  5e4*cd_loss + deform_loss


                    ## Loss 3 : Regularization loss
                    hypo_losses = loss_hypo_weight(deform_output) + loss_hypo_weight(shape_def_output)
                    emb_losses = loss_emb(feat) + loss_emb(shape_emb) + loss_emb(self.force_model.cnt_ft)
                    reg_loss =  1e2*hypo_losses + 1e2*emb_losses

                    ## Update
                    train_loss =  deform_sdf_losses + cor_loss + reg_loss
                    self.optims.zero_grad()
                    train_loss.backward()
                    self.optims.step()
                    tot_loss += train_loss.detach()


            if not epoch % 10:
                tqdm.write("Epoch %d, tot_loss %0.6f" % (epoch, tot_loss))

            if epoch % 1000 == 0 and epoch > 2000:
                torch.save({
                    'epoch': epoch,
                    'shape_model': self.object_model.state_dict(),
                    'shape_embedding': self.object_code.weight.data},
                    os.path.join(checkpoints_dir, f'shape_latest.pth'))

                torch.save({'epoch': epoch,
                            'deform_model': self.deformation_model.state_dict(),
                            'feats' : feats,
                            'force_model':  self.force_model.state_dict(),
                            "optimizer_state_dict": self.optims.state_dict()},
                            os.path.join(checkpoints_dir, f'main_latest.pth'))

=======
>>>>>>> no lfs
