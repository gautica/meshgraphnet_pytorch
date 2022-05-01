import wandb
import torch
from meshgraphnet.utils.common import NodeType
from torch_geometric.data import Dataset
from pytorch_lightning.callbacks import Callback
from numpy.random import randint
from meshgraphnet.visualization.viz import generate_images


class PlotsCallBack(Callback):
    def __init__(self,
                 dataset: Dataset,
                 field: str,
                 mode: str,
                 every_n_epoch: int):
        super(PlotsCallBack, self).__init__()
        self.dataset = dataset
        self.field = field  # velocity for cfd model or world_pos for cloth model
        self.target_field = "target_" + field
        self.prev_field = "prev_" + field
        self.mode = mode  # cfd|cloth
        self.every_n_epoch = every_n_epoch
        self.rollout_steps = 30
        self.ready = True

    def on_sanity_check_start(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.ready = False

    def on_sanity_check_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        self.ready = True

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.every_n_epoch == 0 or \
           trainer.current_epoch % self.every_n_epoch != 0:
            return

        pl_module.eval()
        idx = randint(len(self.dataset))
        graph = self.dataset[idx]
        gt = graph[self.target_field].numpy()[:, :self.rollout_steps]
        graph[self.field] = graph[self.field][:, :1]           # initial state [#nodes, 1, feature_dim]
        graph[self.target_field] = graph[self.target_field][:, :1]
        if self.mode == "cloth":
            graph[self.prev_field] = graph[self.prev_field][:, :1]
        
        predict = self._roll_out(pl_module, idx, graph, self.rollout_steps).numpy()
        graph = graph.to('cpu')
        predicted_images = generate_images(graph.mesh_pos, graph.cells, field=predict, mode=self.mode, every_k_step=1)
        gt_images = generate_images(graph.mesh_pos, graph.cells, field=gt, mode=self.mode, every_k_step=1)

        trainer.logger.experiment.log({'video/predict': wandb.Video(predicted_images, fps=5)})
        trainer.logger.experiment.log({'video/gt': wandb.Video(gt_images, fps=5)})

    def _roll_out(self, pl_module, idx, initial_state, num_steps=30):
        """ Rolls out a model trajectory. """
        field = initial_state[self.field]
        node_type_all = initial_state.node_type
        current_state = initial_state
        traj = []
        for i in range(num_steps):
            if self.mode == "cloth":
                node_type = node_type_all[:, i].unsqueeze(dim=1)
                mask = torch.eq(node_type, NodeType.NORMAL)
                current_state.node_type = node_type                 # because node_type is dynamic feature for cloth model
            else:
                node_type = node_type_all.unsqueeze(dim=1)
                mask = torch.logical_or(torch.eq(node_type, NodeType.NORMAL),
                                        torch.eq(node_type, NodeType.OUTFLOW))
      
            predict_field = pl_module.predict_step(current_state.to(pl_module.device), idx)
            # don't update boundary nodes
            next_ = torch.where(mask, predict_field, field)
            # update current state
            if self.mode == 'cloth':
                current_state[self.prev_field] = current_state[self.field]
            current_state[self.field] = next_
            traj.append(next_)
        traj = torch.cat(traj, dim=1)
        return traj
