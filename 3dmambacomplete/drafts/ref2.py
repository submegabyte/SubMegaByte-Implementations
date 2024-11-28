import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np

# Seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        
        # Projection layers
        self.in_proj = nn.Linear(d_model, expand * d_model)
        self.out_proj = nn.Linear(expand * d_model, d_model)
        
        # Convolution layer
        self.conv1d = nn.Conv1d(
            in_channels=expand * d_model, 
            out_channels=expand * d_model, 
            kernel_size=d_conv, 
            groups=expand * d_model,
            padding=d_conv - 1
        )
        
        # Selective scan parameters
        self.x_proj = nn.Linear(expand * d_model, d_state + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(d_state, expand * d_model)
        
        # Initialize A and D
        self.A = -torch.exp(torch.linspace(0, -5, d_state)).unsqueeze(0)
        self.D = nn.Parameter(torch.ones(expand * d_model))

        self.softplus = nn.Softplus()
        
    def forward(self, x):
        # Initial projection
        x_proj = self.in_proj(x)
        
        # Prepare for convolution
        x_conv = self.conv1d(x_proj.transpose(1, 2)).transpose(1, 2)
        
        # Selective scanning components
        x_dbl = self.x_proj(x_conv)
        
        # Split into components
        delta, B, C = torch.split(x_dbl, self.A.shape[1], dim=-1)
        
        # Prepare time scales
        # delta = F.softplus(self.dt_proj(delta))
        delta = self.softplus(self.dt_proj(delta))
        
        # Selective state space recurrence
        y = self.selective_scan(x_conv, delta, self.A, B, C)
        
        # Output projection
        return self.out_proj(y)
    
    def selective_scan(self, x, delta, A, B, C):
        batch, length, dim = x.shape
        
        # Move A to the same device as x
        A = A.to(x.device)
        
        # Initialize hidden state
        h = torch.zeros(batch, dim, device=x.device)
        outputs = []
        
        # Basic selective scan implementation
        for t in range(length):
            # Compute decay and input-dependent state update
            decay = torch.exp(delta[:, t] * A)
            h = h * decay + x[:, t] * B[:, t]
            output = h * C[:, t]
            outputs.append(output)
        
        return torch.stack(outputs, dim=1)

class MambaModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, n_layers=4):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        # Stack of Mamba blocks
        self.layers = nn.ModuleList([
            MambaBlock(d_model) for _ in range(n_layers)
        ])
        
        # Final layer norm and output projection
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, x):
        # Embed input
        x = self.embedding(x)
        
        # Pass through Mamba blocks
        for layer in self.layers:
            x = layer(x)
        
        # Final layer norm and output projection
        x = self.ln_f(x)
        return self.lm_head(x)

def generate_synthetic_data(vocab_size, batch_size, sequence_length):
    """
    Generate synthetic training data for language modeling
    """
    # Random input sequences
    input_sequences = torch.randint(0, vocab_size, (batch_size, sequence_length))
    
    # Shift sequences to create target (next token prediction)
    target_sequences = torch.roll(input_sequences, -1, dims=1)
    target_sequences[:, -1] = torch.randint(0, vocab_size, (batch_size,))
    
    return input_sequences, target_sequences

def train_mamba_model():
    # Hyperparameters
    vocab_size = 1000  # Size of vocabulary
    batch_size = 32    # Number of sequences per batch
    sequence_length = 50  # Length of each sequence
    d_model = 128      # Model dimension
    n_layers = 4       # Number of Mamba blocks
    learning_rate = 0.001
    num_epochs = 50

    # Initialize model, loss, and optimizer
    model = MambaModel(vocab_size, d_model=d_model, n_layers=n_layers)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training loop
    for epoch in range(num_epochs):
        # Generate synthetic data
        input_sequences, target_sequences = generate_synthetic_data(
            vocab_size, batch_size, sequence_length
        )

        # Forward pass
        outputs = model(input_sequences)
        
        # Reshape for loss computation
        loss = criterion(
            outputs.view(-1, vocab_size), 
            target_sequences.view(-1)
        )

        # Backward pass and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Print progress
        if epoch % 5 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

    return model

def main():
    # Train the model
    trained_model = train_mamba_model()

    # Optional: Simple inference demonstration
    print("\nInference Demonstration:")
    with torch.no_grad():
        # Generate a sample input sequence
        sample_input = torch.randint(0, 1000, (1, 50))
        sample_output = trained_model(sample_input)
        print("Sample input shape:", sample_input.shape)
        print("Sample output shape:", sample_output.shape)

if __name__ == "__main__":
    main()