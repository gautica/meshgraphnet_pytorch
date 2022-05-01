from torch import nn
from meshgraphnet.model.networks import EncodeProcessDecode
from meshgraphnet.utils.common import NodeType


class MGN(nn.Module):
    def __init__(self, config):
        super(MGN, self).__init__()
        self.config = config
        self.learned_model = EncodeProcessDecode(input_dim_node=self.config.node_feat_size + NodeType.SIZE,
                                                 input_dim_edge=self.config.edge_feat_size,
                                                 hidden_dim=config.latent_size,
                                                 output_dim=self.config.output_feat_size,
                                                 num_layers=config.num_layers,
                                                 message_passing_steps=config.message_passing_steps)

    def forward(self, edge_index, node_features, edge_features):
        prediction = self.learned_model(edge_index, node_features, edge_features)
        return prediction
