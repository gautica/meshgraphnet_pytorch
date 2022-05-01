import torch
from torch import nn
from torch_scatter import scatter_add


class MLP(nn.Module):
    def __init__(self, 
                 input_dim, 
                 hidden_dim, 
                 output_dim, 
                 num_layers, 
                 layer_norm=True, 
                 activation=nn.ReLU(), 
                 activate_final=False):
        """
        Add Docs
        """
        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, hidden_dim), activation]
        for i in range(num_layers-1):
            layers += [nn.Linear(hidden_dim, hidden_dim), activation]
            
        if activate_final:
            layers += [nn.Linear(hidden_dim, output_dim), activation]
        else:
            layers += [nn.Linear(hidden_dim, output_dim)]
            
        if layer_norm:
            layers += [nn.LayerNorm(output_dim)]
            
        self.net = nn.Sequential(*layers)
        
    def forward(self, input):
        output = self.net(input)
        
        return output

    
class GraphNetBlock(nn.Module):
    def __init__(self, hidden_dim, num_layers):
        super(GraphNetBlock, self).__init__()
        self.mlp_node = MLP(input_dim=2*hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, num_layers=num_layers)  #3*hidden_dim: [nodes, accumulated_edges]
        self.mlp_edge = MLP(input_dim=3*hidden_dim, hidden_dim=hidden_dim, output_dim=hidden_dim, num_layers=num_layers)  #3*hidden_dim: [sender, edge, receiver]
    
    def _update_edges(self, edge_idx, node_features, edge_features):
        senders, receivers = edge_idx
        sender_features = node_features[senders]
        receiver_features = node_features[receivers]
        features = torch.cat([sender_features, receiver_features, edge_features], dim=-1)
        
        return self.mlp_edge(features)
    
    def _update_nodes(self, edge_idx, node_features, edge_features):
        _, receivers = edge_idx
        accumulate_edges = scatter_add(edge_features, receivers, dim=0)   # ~ tf.math.unsorted_segment_sum
        features = torch.cat([node_features, accumulate_edges], dim=-1)
        return self.mlp_node(features)
        
    def forward(self, edge_idx, node_features, edge_features):
        """
        TODO: Docs
        """
        
        new_edge_features = self._update_edges(edge_idx, node_features, edge_features)
        new_node_features = self._update_nodes(edge_idx, node_features, new_edge_features)
        
        #add residual connections
        new_node_features += node_features
        new_edge_features += edge_features
        
        return new_node_features, new_edge_features
    
    
class Encoder(nn.Module):
    def __init__(self, input_dim_node, input_dim_edge, hidden_dim, num_layers):
        super(Encoder, self).__init__()
        self.node_mlp = MLP(input_dim=input_dim_node, hidden_dim=hidden_dim, output_dim=hidden_dim, num_layers=num_layers, activate_final=False)
        self.edge_mlp = MLP(input_dim=input_dim_edge, hidden_dim=hidden_dim, output_dim=hidden_dim, num_layers=num_layers, activate_final=False)
        
    def forward(self, node_features, edge_features):
        node_latents = self.node_mlp(node_features)
        edge_latents = self.edge_mlp(edge_features)
        
        return node_latents, edge_latents


class Process(nn.Module):
    def __init__(self, hidden_dim, num_layers, message_passing_steps):
        super(Process, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(message_passing_steps):
            self.blocks.append(GraphNetBlock(hidden_dim, num_layers))
            
    def forward(self, edge_idx, node_features, edge_features):
        for graphnetblock in self.blocks:
            node_features, edge_features = graphnetblock(edge_idx, node_features, edge_features)
            
        return node_features, edge_features
    
    
class Decoder(nn.Module):
    def __init__(self, hidden_dim, output_dim, num_layers):
        super(Decoder, self).__init__()
        self.mlp = MLP(input_dim=hidden_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers, layer_norm=False, activate_final=False)
    
    def forward(self, node_features):
        return self.mlp(node_features)
        
        
class EncodeProcessDecode(nn.Module):
    """Encoder-Process-Decoder GraphNet model."""
    def __init__(self,
                 input_dim_node,
                 input_dim_edge,
                 hidden_dim,
                 output_dim,
                 num_layers,
                 message_passing_steps):
        super(EncodeProcessDecode, self).__init__()
        
        self.encoder = Encoder(input_dim_node, input_dim_edge, hidden_dim, num_layers)
        self.process = Process(hidden_dim, num_layers, message_passing_steps)
        self.decoder = Decoder(hidden_dim, output_dim, num_layers)
        
    def forward(self, edge_idx, node_features, edge_features):
        # Encode node/edge feature to latent space
        node_features, edge_features = self.encoder(node_features, edge_features)
        # Process message passing
        node_features, edge_features = self.process(edge_idx, node_features, edge_features)
        # Decode to output space
        predict = self.decoder(node_features)
        return predict
        
        
        