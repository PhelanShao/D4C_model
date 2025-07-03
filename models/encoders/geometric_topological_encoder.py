
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np
import math


class SE3EquivariantLayer(nn.Module):
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim  
        self.output_dim = output_dim
        
        self.scalar_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.vector_net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.attention_net = nn.Sequential(
            nn.Linear(input_dim + 3, hidden_dim),  # +3 for distance features
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def compute_distances(self, coords: torch.Tensor) -> torch.Tensor:
        # coords: [batch_size, num_atoms, 3]
        batch_size, num_atoms, _ = coords.shape
        
        coords_i = coords.unsqueeze(2)  # [batch_size, num_atoms, 1, 3]
        coords_j = coords.unsqueeze(1)  # [batch_size, 1, num_atoms, 3]
        
        diff = coords_i - coords_j  # [batch_size, num_atoms, num_atoms, 3]
        distances = torch.norm(diff, dim=-1, p=2)  # [batch_size, num_atoms, num_atoms]
        
        return distances, diff
    
    def forward(self, 
                node_features: torch.Tensor,
                coordinates: torch.Tensor,
                edge_mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, num_atoms, _ = node_features.shape
        
        distances, directions = self.compute_distances(coordinates)
        
        scalar_features = self.scalar_net(node_features)
        
        dist_features = torch.stack([
            distances,
            1.0 / (distances + 1e-6),  
            torch.exp(-distances)      
        ], dim=-1)  # [batch_size, num_atoms, num_atoms, 3]
        
        attention_input = torch.cat([
            node_features.unsqueeze(2).expand(-1, -1, num_atoms, -1),
            dist_features
        ], dim=-1)
        
        attention_weights = self.attention_net(attention_input).squeeze(-1)
        attention_weights = attention_weights.masked_fill(~edge_mask.bool(), float('-inf'))
        attention_weights = F.softmax(attention_weights, dim=-1)
        
        vector_messages = directions * attention_weights.unsqueeze(-1)
        vector_aggregated = vector_messages.sum(dim=2)  # [batch_size, num_atoms, 3]
        
        vector_norms = torch.norm(vector_aggregated, dim=-1, keepdim=True)
        vector_features = self.vector_net(node_features)  # [batch_size, num_atoms, output_dim]
        
        vector_out = vector_aggregated * vector_features[..., :3]  
        
        return scalar_features, vector_out


class GraphAttentionLayer(nn.Module):
    
    def __init__(self, input_dim: int, output_dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.head_dim = output_dim // num_heads
        
        assert output_dim % num_heads == 0
        
        self.q_linear = nn.Linear(input_dim, output_dim)
        self.k_linear = nn.Linear(input_dim, output_dim) 
        self.v_linear = nn.Linear(input_dim, output_dim)
        self.out_linear = nn.Linear(output_dim, output_dim)
        
        self.edge_encoder = nn.Sequential(
            nn.Linear(4, output_dim),  
            nn.ReLU(),
            nn.Linear(output_dim, output_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(output_dim)
        
    def forward(self,
                node_features: torch.Tensor,
                adjacency: torch.Tensor,
                edge_features: Optional[torch.Tensor] = None) -> torch.Tensor:

        batch_size, num_atoms, _ = node_features.shape
        
        Q = self.q_linear(node_features)  # [batch_size, num_atoms, output_dim]
        K = self.k_linear(node_features)
        V = self.v_linear(node_features)
        
        Q = Q.view(batch_size, num_atoms, self.num_heads, self.head_dim)
        K = K.view(batch_size, num_atoms, self.num_heads, self.head_dim)
        V = V.view(batch_size, num_atoms, self.num_heads, self.head_dim)
        
        scores = torch.einsum('bihd,bjhd->bhij', Q, K) / math.sqrt(self.head_dim)
        
        if edge_features is not None:
            edge_embeds = self.edge_encoder(edge_features)  # [batch_size, num_atoms, num_atoms, output_dim]
            edge_embeds = edge_embeds.view(batch_size, num_atoms, num_atoms, self.num_heads, self.head_dim)
            edge_bias = torch.einsum('bihd,bijhd->bhij', Q, edge_embeds)
            scores = scores + edge_bias
        
        adjacency_mask = adjacency.unsqueeze(1).expand(-1, self.num_heads, -1, -1)
        scores = scores.masked_fill(~adjacency_mask.bool(), float('-inf'))
        
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        out = torch.einsum('bhij,bjhd->bihd', attention_weights, V)
        out = out.contiguous().view(batch_size, num_atoms, self.output_dim)
        
        if self.input_dim == self.output_dim:
            out = out + node_features
        out = self.norm(out)
        
        return out


class GeometricTopologicalEncoder(nn.Module):
    
    def __init__(self, 
                 input_dim: int = 256,
                 hidden_dim: int = 512,
                 output_dim: int = 512,
                 num_layers: int = 4,
                 num_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        
        self.input_encoder = nn.Linear(input_dim, hidden_dim)
        
        self.geometric_layers = nn.ModuleList([
            SE3EquivariantLayer(hidden_dim, hidden_dim, hidden_dim)
            for _ in range(num_layers)
        ])
        
        self.topological_layers = nn.ModuleList([
            GraphAttentionLayer(hidden_dim, hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.geo_topo_fusion = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(hidden_dim) for _ in range(num_layers)
        ])
        
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        self.global_attention = nn.MultiheadAttention(
            embed_dim=output_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
    def create_edge_features(self, 
                           coordinates: torch.Tensor, 
                           bond_types: torch.Tensor,
                           distances: torch.Tensor) -> torch.Tensor:
        batch_size, num_atoms, _ = coordinates.shape

        edge_features = torch.zeros(batch_size, num_atoms, num_atoms, 4, 
                                  device=coordinates.device)
        
        edge_features[..., 0] = bond_types
        
        edge_features[..., 1] = distances / (distances.max() + 1e-6)
        
        edge_features[..., 2] = torch.cos(distances * math.pi)
        edge_features[..., 3] = torch.sin(distances * math.pi)
        
        return edge_features
    
    def forward(self,
                node_features: torch.Tensor,
                coordinates: torch.Tensor,
                adjacency: torch.Tensor,
                bond_types: torch.Tensor,
                atom_mask: torch.Tensor) -> torch.Tensor:

        batch_size, num_atoms, _ = node_features.shape
        
        h = self.input_encoder(node_features)  # [batch_size, num_atoms, hidden_dim]
        
        coords_i = coordinates.unsqueeze(2)
        coords_j = coordinates.unsqueeze(1)
        distances = torch.norm(coords_i - coords_j, dim=-1, p=2)
        
        edge_features = self.create_edge_features(coordinates, bond_types, distances)
        
        for i in range(self.num_layers):
            geometric_scalar, geometric_vector = self.geometric_layers[i](
                h, coordinates, adjacency
            )

            topological_features = self.topological_layers[i](
                h, adjacency, edge_features
            )

            vector_magnitude = torch.norm(geometric_vector, dim=-1, keepdim=True)
            enhanced_geometric = torch.cat([geometric_scalar, vector_magnitude], dim=-1)
            
            if enhanced_geometric.shape[-1] != topological_features.shape[-1]:
                enhanced_geometric = enhanced_geometric[..., :topological_features.shape[-1]]
            
            fused_features, _ = self.geo_topo_fusion[i](
                enhanced_geometric, topological_features, topological_features
            )

            h = h + fused_features
            h = self.layer_norms[i](h)
        
        h = self.output_projection(h)  # [batch_size, num_atoms, output_dim]
        
        global_repr, attention_weights = self.global_attention(h, h, h)
        
        mask_expanded = atom_mask.unsqueeze(-1).float()
        masked_repr = global_repr * mask_expanded
        atom_counts = atom_mask.sum(dim=1, keepdim=True).float()
        final_repr = masked_repr.sum(dim=1) / (atom_counts + 1e-8)
        
        return final_repr


class GeometricTopologicalFusion(nn.Module):

    def __init__(self, feature_dim: int, num_heads: int = 8):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_heads = num_heads

        self.geo_encoder = nn.Sequential(
            nn.Linear(feature_dim + 3, feature_dim),  # +3 for vector features
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        self.topo_encoder = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.ReLU(),
            nn.Linear(feature_dim, feature_dim)
        )

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=feature_dim,
            num_heads=num_heads,
            dropout=0.1,
            batch_first=True
        )

        self.gate = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.Sigmoid()
        )
        
    def forward(self, 
                geometric_features: torch.Tensor,
                geometric_vectors: torch.Tensor,
                topological_features: torch.Tensor) -> torch.Tensor:

        geo_input = torch.cat([geometric_features, geometric_vectors], dim=-1)
        geo_encoded = self.geo_encoder(geo_input)        

        topo_encoded = self.topo_encoder(topological_features)
        
        fused_geo, _ = self.cross_attention(geo_encoded, topo_encoded, topo_encoded)
        fused_topo, _ = self.cross_attention(topo_encoded, geo_encoded, geo_encoded)
        
        combined = torch.cat([fused_geo, fused_topo], dim=-1)
        gate_weights = self.gate(combined)
        
        final_features = gate_weights * fused_geo + (1 - gate_weights) * fused_topo
        
        return final_features


if __name__ == "__main__":
    encoder = GeometricTopologicalEncoder(
        input_dim=256,
        hidden_dim=512,
        output_dim=512,
        num_layers=4
    )
    
    batch_size = 4
    num_atoms = 64
    
    node_features = torch.randn(batch_size, num_atoms, 256)
    coordinates = torch.randn(batch_size, num_atoms, 3)
    adjacency = torch.randint(0, 2, (batch_size, num_atoms, num_atoms)).float()
    bond_types = torch.randint(0, 4, (batch_size, num_atoms, num_atoms)).float()
    atom_mask = torch.ones(batch_size, num_atoms)

    try:
        output = encoder(node_features, coordinates, adjacency, bond_types, atom_mask)
        print(f"输出形状: {output.shape}")
        print(f"几何-拓扑双重编码器创建成功!")
        
        # 输出一些统计信息
        print(f"输出均值: {output.mean().item():.4f}")
        print(f"输出标准差: {output.std().item():.4f}")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc() 
