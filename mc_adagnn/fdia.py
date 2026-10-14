import torch
import numpy as np

def apply_fdia_to_batch(x, attack_type='scaling', attack_strength=0.10, attack_ratio=1.0, 
                        attacked_nodes=None, attacked_timestamps=None):
    """
    Applies False Data Injection Attack (FDIA) to a batch of input measurements.
    
    Args:
        x (torch.Tensor): Input tensor of shape (batch, num_nodes, seq_len)
        attack_type (str): 'scaling' or 'additive'
        attack_strength (float): The strength of the attack (alpha or delta)
        attack_ratio (float): Percentage of measurements to attack (0.0 to 1.0)
        attacked_nodes (list): List of node indices to attack. If None, all nodes are candidates.
        attacked_timestamps (list): List of timestamp indices to attack. If None, all are candidates.
        
    Returns:
        torch.Tensor: Attacked input tensor of the same shape as x.
    """
    if attack_type == 'none' or attack_strength == 0.0 or attack_ratio == 0.0:
        return x
        
    x_attacked = x.clone()
    batch_size, num_nodes, seq_len = x.shape
    
    # Create a mask of shape (batch_size, num_nodes, seq_len)
    mask = torch.ones_like(x, dtype=torch.bool)
    
    # Node-specific attack
    if attacked_nodes is not None:
        node_mask = torch.zeros(num_nodes, dtype=torch.bool, device=x.device)
        node_mask[attacked_nodes] = True
        mask = mask & node_mask.view(1, num_nodes, 1)
        
    # Time-window attack
    if attacked_timestamps is not None:
        time_mask = torch.zeros(seq_len, dtype=torch.bool, device=x.device)
        time_mask[attacked_timestamps] = True
        mask = mask & time_mask.view(1, 1, seq_len)
        
    # Attack masking (ratio)
    if attack_ratio < 1.0:
        rand_mask = torch.rand(x.shape, device=x.device) < attack_ratio
        mask = mask & rand_mask
        
    if attack_type == 'scaling':
        # x_attacked = x * (1 + alpha)
        # We only apply it where mask is True
        perturbation = x * attack_strength
        x_attacked[mask] = x_attacked[mask] + perturbation[mask]
        
    elif attack_type == 'additive':
        # x_attacked = x + delta
        x_attacked[mask] = x_attacked[mask] + attack_strength
        
    else:
        raise ValueError(f"Unknown attack type: {attack_type}")
        
    return x_attacked
