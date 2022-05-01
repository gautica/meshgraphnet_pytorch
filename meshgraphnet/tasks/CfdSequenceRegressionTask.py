import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from meshgraphnet.model.mgn import MGN
from meshgraphnet.utils.normalization import Normalizer
from meshgraphnet.utils.common import NodeType
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import math
from tqdm import tqdm
from einops import rearrange, reduce, repeat


class CfdSequenceRegression(pl.LightningModule):
    def __init__(self, config, field: str ='velocity'):
        super(CfdSequenceRegression, self).__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.field = field
        self.config = config
        self.target_field = 'target_' + field
        self.model = MGN(self.config)
        self.node_normalizer = Normalizer(size=self.config.node_feat_size + NodeType.SIZE, name='node_normalizer')
        self.edge_normalizer = Normalizer(size=self.config.edge_feat_size, name='edge_normalizer')
        self.output_normalizer = Normalizer(size=self.config.output_feat_size, name='output_normalizer')

    def _extract_features(self, graph, is_training=False, add_noise=False):
        if add_noise:
            # add noise like in original implementation
            self._add_noise(graph)

        # build feature vectors for each node and edge
        length_trajectory = graph[self.field].shape[1]  # should be 598
        node_type = F.one_hot(graph.node_type[:, 0].to(torch.int64), NodeType.SIZE)
        # n: number of node; f: feature dims; l: trajectory length
        node_type = repeat(node_type, 'n f -> n l f', l=length_trajectory)
        node_features = torch.cat([graph[self.field], node_type], dim=-1)  # (num_nodes, length_traj, feat_dim)

        senders, receivers = graph.edge_index
        relative_mesh_pos = graph.mesh_pos[senders] - graph.mesh_pos[receivers]
        edge_features = torch.cat([relative_mesh_pos,
                                   torch.norm(relative_mesh_pos, dim=-1, keepdim=True)],
                                  dim=-1)
        # (num_edges, length_traj, feat_dim)
        edge_features = repeat(edge_features, 'n f -> n l f', l=length_trajectory)

        # normalization
        node_features = self.node_normalizer(node_features, is_training)
        edge_features = self.edge_normalizer(edge_features, is_training)

        return node_features, edge_features

    def _add_noise(self, graph):
        length_trajectory = graph[self.field].shape[1]
        mask = torch.eq(graph.node_type, NodeType.NORMAL)
        mask = repeat(mask, 'n f -> n l f', l=length_trajectory)

        noise = torch.normal(mean=0.0, std=self.config.noise_scale,
                             size=graph[self.field].shape, dtype=torch.float32).to(mask.device)

        noise = torch.where(mask, noise, torch.zeros_like(noise).to(mask.device))
        graph[self.field] += noise
        graph[self.target_field] += (1.0 - self.config.noise_gamma) * noise

    def l2_loss(self, prediction, graph, start, end, is_training=True):
        # build target velocity change
        cur_field = graph[self.field][:, start:end]
        target_field = graph[self.target_field][:, start:end]
        field_change = target_field - cur_field
        target_normalized = self.output_normalizer(field_change, accumulate=is_training)

        # build loss
        node_type = rearrange(graph.node_type, 'n 1 -> n')
        loss_mask = torch.logical_or(torch.eq(node_type, NodeType.NORMAL),
                                     torch.eq(node_type, NodeType.OUTFLOW))
        error = reduce((target_normalized - prediction) ** 2, 'n l f -> n l', 'sum')
        loss = torch.mean(error[loss_mask])
        return loss

    def training_step(self, graph_batch, batch_idx):
        # extract features
        node_features, edge_features = self._extract_features(graph_batch, is_training=True, add_noise=True)

        traj_length = graph_batch[self.field].shape[1]
        small_step = self.config.accumulate_step_size
        num_steps = int(math.ceil(traj_length / small_step))
        accumulate_loss = 0.0
        optimizer = self.optimizers()
        for i in range(0, num_steps):                   # solve the issue of out of memory
            edge_index = graph_batch.edge_index
            start = i * small_step
            end = (i + 1) * small_step
            prediction = self.model(edge_index,
                                    node_features[:, start:end],
                                    edge_features[:, start:end]
                                    )
            loss = self.l2_loss(prediction, graph_batch, start, end, is_training=True)
            if self.current_epoch == 0:  # first epoch to accumulate data for normalization term
                continue
            self.manual_backward(loss)
            accumulate_loss += loss.detach()
        if self.current_epoch == 0:  # first epoch to accumulate data for normalization term
            return
        optimizer.step()
        optimizer.zero_grad()
        self.log('train/loss', accumulate_loss / traj_length, on_step=True, on_epoch=True, logger=True)
        return loss

    def on_train_epoch_end(self):
        scheduler = self.lr_schedulers()
        scheduler.step()

    def validation_step(self, graph_batch, batch_idx):
        # extract features
        node_features, edge_features = self._extract_features(graph_batch, is_training=False, add_noise=False)

        traj_length = graph_batch[self.field].shape[1]
        small_step = self.config.accumulate_step_size
        num_steps = int(math.ceil(traj_length / small_step))
        accumulate_loss = 0.0
        for i in range(0, num_steps):  # solve the issue of out of memory
            edge_index = graph_batch.edge_index
            start = i * small_step
            end = (i + 1) * small_step
            prediction = self.model(edge_index,
                                    node_features[:, start:end],
                                    edge_features[:, start:end]
                                    )
            loss = self.l2_loss(prediction, graph_batch, start, end, is_training=False)
            accumulate_loss += loss.detach()
        self.log('valid/loss', accumulate_loss / traj_length, on_step=True, on_epoch=True, logger=True)

    def predict_step(self, graph_batch, batch_idx: int, dataloader_idx=None):
        """Integrate model outputs."""
        # extract features
        node_features, edge_features = self._extract_features(graph_batch, is_training=False, add_noise=False)

        traj_length = graph_batch[self.field].shape[1]
        small_step = self.config.accumulate_step_size
        num_steps = int(math.ceil(traj_length / small_step))
        predictions = []
        for i in range(0, num_steps):  # solve the issue of out of memory
            edge_index = graph_batch.edge_index
            start = i * small_step
            end = (i + 1) * small_step
            prediction = self.model(edge_index,
                                    node_features[:, start:end],
                                    edge_features[:, start:end]
                                    )

            field_update = self.output_normalizer.inverse(prediction)
            predict = graph_batch[self.field] + field_update
            predictions.append(predict)

        predictions = rearrange(predictions, 'b n l f -> n (b l) f')
        return predictions.cpu()

    def configure_optimizers(self):
        optimizer = Adam(self.parameters(), lr=self.config.lr)
        lr_scheduler = ExponentialLR(optimizer=optimizer, gamma=self.config.decayRate)
        return [optimizer], [lr_scheduler]

    def predict_trajectory(self, dataloader, step):
        trajectories = []

        for batch_graph in tqdm(dataloader):
            traj = self._rollout(batch_graph, step)

            for i in range(batch_graph.num_graphs):
                idx = (batch_graph.batch == i).nonzero().squeeze()
                trajectories.append(traj[idx])

        return trajectories

    def _rollout(self, graph, step):
        traj = []
        num_steps = min(graph[self.target_field].shape[1], step)
        init_field = graph[self.field][:, :1]  # initial state [#nodes, 1, feature_dim]
        node_type = rearrange(graph.node_type, 'n 1 -> n 1 1')
        mask = torch.logical_or(torch.eq(node_type, NodeType.NORMAL),
                                torch.eq(node_type, NodeType.OUTFLOW))
        current_state = graph
        current_state[self.field] = init_field
        for _ in tqdm(range(num_steps)):
            with torch.no_grad():
                predict_field = self.predict_step(current_state, batch_idx=None).detach().cpu()
            next_field = torch.where(mask, predict_field, init_field)
            # update current state
            current_state[self.field] = next_field
            traj.append(next_field)
        traj = rearrange(traj, 'l n 1 f -> n l f')
        return traj
