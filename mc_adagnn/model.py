import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphConstructionLayer(nn.Module):
    def __init__(self, num_nodes, seq_len, embed_dim, alpha=1.0):
        super(GraphConstructionLayer, self).__init__()
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.alpha = alpha
        
        # 1. Skeleton-Foundation (SF) Channel
        self.E1 = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.E2 = nn.Parameter(torch.randn(num_nodes, embed_dim))
        self.Psi1 = nn.Parameter(torch.randn(embed_dim, embed_dim))
        self.Psi2 = nn.Parameter(torch.randn(embed_dim, embed_dim))
        
        # 2. Core-Stability (CS) Channel
        p = 2 # pooling factor
        self.pool = nn.AvgPool1d(kernel_size=p)
        pooled_len = seq_len // p
        self.conv_csc1 = nn.Conv1d(1, embed_dim, kernel_size=pooled_len)
        self.conv_csc2 = nn.Conv1d(1, embed_dim, kernel_size=pooled_len)
        
        # 3. Dynamic-Interaction (DI) Channel
        self.conv_dic1 = nn.Conv1d(1, embed_dim, kernel_size=seq_len)
        self.conv_dic2 = nn.Conv1d(1, embed_dim, kernel_size=seq_len)
        
        # Multi-channel fusion weights
        self.alpha_sfc = nn.Parameter(torch.tensor(0.33))
        self.alpha_csc = nn.Parameter(torch.tensor(0.33))
        self.alpha_dic = nn.Parameter(torch.tensor(0.34))

    def forward(self, x):
        # x shape: (batch, num_nodes, seq_len)
        batch_size = x.size(0)
        
        # SF Channel (Static, based on node embeddings)
        L1 = torch.tanh(self.alpha * torch.matmul(self.E1, self.Psi1))
        L2 = torch.tanh(self.alpha * torch.matmul(self.E2, self.Psi2))
        A_SFC = F.relu(torch.tanh(self.alpha * (torch.matmul(L1, L2.transpose(0, 1)) - torch.matmul(L2, L1.transpose(0, 1)))))
        
        # CS Channel (Stable, based on pooled data)
        # Reshape for conv1d: (batch * num_nodes, 1, seq_len)
        x_reshaped = x.view(batch_size * self.num_nodes, 1, self.seq_len)
        x_pooled = self.pool(x_reshaped)
        Gc1 = self.conv_csc1(x_pooled).view(batch_size, self.num_nodes, -1)
        Gc2 = self.conv_csc2(x_pooled).view(batch_size, self.num_nodes, -1)
        A_CSC = torch.tanh(torch.bmm(Gc1, Gc2.transpose(1, 2)))
        
        # DI Channel (Dynamic, based on full recent data)
        Tc1 = torch.tanh(self.conv_dic1(x_reshaped)).view(batch_size, self.num_nodes, -1)
        Tc2 = torch.tanh(self.conv_dic2(x_reshaped)).view(batch_size, self.num_nodes, -1)
        A_DIC = torch.tanh(torch.bmm(Tc1, Tc2.transpose(1, 2)))
        
        # Fusion
        # Softmax normalize weights to ensure they sum to 1
        weights = F.softmax(torch.stack([self.alpha_sfc, self.alpha_csc, self.alpha_dic]), dim=0)
        
        # Broadcast A_SFC to match batch dimension
        A_SFC_batch = A_SFC.unsqueeze(0).expand(batch_size, -1, -1)
        
        A = weights[0] * A_SFC_batch + weights[1] * A_CSC + weights[2] * A_DIC
        
        return A, A_CSC

class MixHopGraphConv(nn.Module):
    def __init__(self, in_channels, out_channels, K=2, beta=0.05):
        super(MixHopGraphConv, self).__init__()
        self.K = K
        self.beta = beta
        self.weights = nn.ParameterList([nn.Parameter(torch.randn(in_channels, out_channels)) for _ in range(K+1)])
        
    def forward(self, H_in, A):
        H_k = H_in
        H_out = torch.zeros(H_in.size(0), H_in.size(1), self.weights[0].size(1)).to(H_in.device)
        
        for k in range(self.K + 1):
            H_out += torch.matmul(H_k, self.weights[k])
            if k < self.K:
                H_k = self.beta * H_in + (1 - self.beta) * torch.bmm(A, H_k)
                
        return H_out

class InceptionDilatedConv(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(InceptionDilatedConv, self).__init__()
        # 1D conv over time dimension with different kernel sizes
        self.conv1x2 = nn.Conv1d(in_channels, out_channels//4, kernel_size=2, dilation=dilation, padding='same')
        self.conv1x3 = nn.Conv1d(in_channels, out_channels//4, kernel_size=3, dilation=dilation, padding='same')
        self.conv1x6 = nn.Conv1d(in_channels, out_channels//4, kernel_size=6, dilation=dilation, padding='same')
        self.conv1x9 = nn.Conv1d(in_channels, out_channels//4, kernel_size=9, dilation=dilation, padding='same')
        
    def forward(self, x):
        z2 = self.conv1x2(x)
        z3 = self.conv1x3(x)
        z6 = self.conv1x6(x)
        z9 = self.conv1x9(x)
        z = torch.cat([z2, z3, z6, z9], dim=1)
        return z

class TemporalConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, dilation=1):
        super(TemporalConvModule, self).__init__()
        self.filter_conv = InceptionDilatedConv(in_channels, out_channels, dilation)
        self.gate_conv = InceptionDilatedConv(in_channels, out_channels, dilation)
        
    def forward(self, x):
        filter_out = torch.tanh(self.filter_conv(x))
        gate_out = torch.sigmoid(self.gate_conv(x))
        return filter_out * gate_out

class SpatioTemporalLayer(nn.Module):
    def __init__(self, num_nodes, in_channels, out_channels, seq_len, dilation=1):
        super(SpatioTemporalLayer, self).__init__()
        self.num_nodes = num_nodes
        self.tconv1 = TemporalConvModule(in_channels, out_channels, dilation)
        self.gconv = MixHopGraphConv(out_channels, out_channels)
        self.tconv2 = TemporalConvModule(out_channels, out_channels, dilation)
        self.norm = nn.LayerNorm(out_channels)
        
    def forward(self, x, A):
        # x: (batch, num_nodes, in_channels, seq_len)
        batch_size, num_nodes, in_channels, seq_len = x.size()
        
        # 1. Temporal Convolution 1
        x_reshaped = x.reshape(batch_size * num_nodes, in_channels, seq_len)
        t_out1 = self.tconv1(x_reshaped) 
        
        # 2. Graph Convolution
        t_out1 = t_out1.reshape(batch_size, num_nodes, -1, seq_len)
        g_in = t_out1.mean(dim=-1) # Mean over time axis to form graph features
        g_out = self.gconv(g_in, A) 
        
        # Broadcast g_out to seq_len and add residual
        g_out = g_out.unsqueeze(-1).expand(-1, -1, -1, seq_len) 
        g_out = g_out + t_out1
        
        # 3. Temporal Convolution 2
        g_out_reshaped = g_out.reshape(batch_size * num_nodes, -1, seq_len)
        t_out2 = self.tconv2(g_out_reshaped) 
        t_out2 = t_out2.reshape(batch_size, num_nodes, -1, seq_len)
        
        # 4. Residual Connection & Layer Norm
        out = t_out2 + (x if in_channels == t_out2.size(2) else 0)
        out = out.permute(0, 1, 3, 2) # (batch, num_nodes, seq_len, channels)
        out = self.norm(out)
        out = out.permute(0, 1, 3, 2)
        
        return out

class MC_ADAGNN(nn.Module):
    def __init__(self, num_nodes, in_seq_len, out_seq_len, embed_dim, in_channels, hidden_channels, num_layers):
        super(MC_ADAGNN, self).__init__()
        self.graph_construction = GraphConstructionLayer(num_nodes, in_seq_len, embed_dim)
        
        self.start_conv = nn.Conv2d(in_channels, hidden_channels, kernel_size=(1, 1))
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            dilation = 2 ** i
            self.layers.append(SpatioTemporalLayer(num_nodes, hidden_channels, hidden_channels, in_seq_len, dilation))
            
        self.end_conv = nn.Conv2d(hidden_channels, out_seq_len, kernel_size=(1, in_seq_len))

    def forward(self, x):
        # x: (batch, num_nodes, seq_len)
        A, A_CSC = self.graph_construction(x)
        
        # Add channel dimension
        x = x.unsqueeze(2) # (batch, num_nodes, 1, seq_len)
        
        # Permute for start_conv: (batch, channels, num_nodes, seq_len)
        x = x.permute(0, 2, 1, 3)
        x = self.start_conv(x)
        
        # Permute back: (batch, num_nodes, channels, seq_len)
        x = x.permute(0, 2, 1, 3)
        
        for layer in self.layers:
            x = layer(x, A)
            
        # Output mapping module
        # Permute for end_conv: (batch, hidden_channels, num_nodes, seq_len)
        x = x.permute(0, 2, 1, 3)
        out = self.end_conv(x) # (batch, out_seq_len, num_nodes, 1)
        out = out.squeeze(-1) # (batch, out_seq_len, num_nodes)
        
        return out, A_CSC

def mc_adagnn_loss(pred, target, A_CSC, gamma=0.1):
    """
    Combined loss function from the paper.
    L_p: prediction loss (L1/MAE)
    L_s: stability loss (Variance of CS adjacency)
    """
    L_p = F.l1_loss(pred, target)
    
    A_csc_mean = A_CSC.mean()
    L_s = torch.sqrt(torch.mean((A_CSC - A_csc_mean) ** 2))
    
    L = (1 - gamma) * L_p + gamma * L_s
    return L
