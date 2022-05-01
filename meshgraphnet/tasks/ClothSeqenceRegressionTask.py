import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from meshgraphnet.model.mgn import MGN
from meshgraphnet.utils.common import NodeType
from meshgraphnet.utils.normalization import Normalizer
from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
import math
from tqdm import tqdm
from einops import rearrange, reduce, repeat


class ClothSequenceRegression(pl.LightningModule):
    def __init__(self, config, field: str = 'world_pos'):
        super(ClothSequenceRegression, self).__init__()
        self.save_hyperparameters()
        self.automatic_optimization = False

        self.field = field
        self.config = config
        self.prev_field = 'prev_' + field
        self.target_field = 'target_' + field
        self.model = MGN(self.config)
        self.node_normalizer = Normalizer(size=self.config.node_feat_size + NodeType.SIZE, name='node_normalizer')
        self.edge_normalizer = Normalizer(size=self.config.edge_feat_size, name='edge_normalizer')
        self.output_normalizer = Normalizer(size=self.config.output_feat_size, name='output_normalizer')

    def _extract_features(self, graph, is_training=False, add_noise=False):
        assert graph[self.field].shape == graph[self.target_field].shape and \
               graph[self.field].shape == graph[self.prev_field].shape

        if add_noise:
            # add noise like in original implementation
            self._add_noise(graph)
        # build feature vectors for each node and edge
        length_trajectory = graph[self.field].shape[1]  # should be 398
        node_type = F.one_hot(graph.node_type[..., 0].to(torch.int64), NodeType.SIZE)  # node_type is dynamic feature

        field_change = graph[self.field] - graph[self.prev_field]
        node_features = torch.cat([field_change, node_type], dim=-1)  # (num_nodes, length_traj, feat_dim)

        senders, receivers = graph.edge_index
        relative_field = graph[self.field][senders] - graph[self.field][receivers]  # world_pos is dynamic feature
        relative_mesh_pos = graph.mesh_pos[senders] - graph.mesh_pos[receivers]  # mesh_pos is static feature
        relative_mesh_pos = repeat(relative_mesh_pos, 'n f -> n l f', l=length_trajectory)
        edge_features = torch.cat([relative_field,
                                   torch.norm(relative_field, dim=-1, keepdim=True),
                                   relative_mesh_pos,
                                   torch.norm(relative_mesh_pos, dim=-1, keepdim=True)],
                                  dim=-1)

        # normalization
        node_features = self.node_normalizer(node_features, accumulate=is_training)
        edge_features = self.edge_normalizer(edge_features, accumulate=is_training)

        return node_features, edge_features

    def _add_noise(self, graph):
        mask = torch.eq(graph.node_type, NodeType.NORMAL)
        noise = torch.normal(mean=0.0, std=self.config.noise_scale,
                             size=graph[self.field].shape, dtype=torch.float32, device=mask.device)

        noise = torch.where(mask, noise, torch.zeros_like(noise).to(mask.device))
        graph[self.field] += noise
        graph[self.target_field] += (1.0 - self.config.noise_gamma) * noise

    def l2_loss(self, prediction, graph, start, end, is_training=True):
        # build target acceleration
        cur = graph[self.field][:, start:end]
        prev = graph[self.prev_field][:, start:end]
        target = graph[self.target_field][:, start:end]
        target_acceleration = target - 2 * cur + prev
        target_normalized = self.output_normalizer(target_acceleration, accumulate=is_training)

        # build loss
        node_type = graph.node_type[..., 0][:, start:end]
        loss_mask = torch.eq(node_type, NodeType.NORMAL)
        error = reduce((target_normalized - prediction) ** 2, 'n l f -> n l', reduction='sum')
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

            acceleration = self.output_normalizer.inverse(prediction)
            # integrate forward
            current = graph_batch[self.field][:, start:end]
            prev = graph_batch[self.prev_field][:, start:end]
            predict_field = 2 * current + acceleration - prev
            predictions.append(predict_field)
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
        node_type = graph.node_type
        init_field = graph[self.field][:, :1]  # initial state [#nodes, 1, feature_dim]
        current_state = graph
        current_state[self.field] = init_field
        current_state[self.prev_field] = graph[self.prev_field][:, :1]
        current_state[self.target_field] = graph[self.target_field][:, :1]
        for i in tqdm(range(num_steps)):
            n_t = node_type[:, i:i+1]
            mask = torch.eq(n_t, NodeType.NORMAL)
            current_state.node_type = n_t
            with torch.no_grad():
                predict_field = self.predict_step(current_state, batch_idx=None).detach().cpu()
            # don't update boundary nodes
            next_ = torch.where(mask, predict_field, init_field)
            # update current state
            current_state[self.prev_field] = current_state[self.field]
            current_state[self.field] = next_

            traj.append(next_)
        traj = torch.cat(traj, dim=1)           # [#nodes, len_traj, feat]
        return traj