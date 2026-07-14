# TLS Protocol Detection ML Training Guide

## Overview

This guide describes how to train and export ML models for TLS protocol classification in nDPI.

## Training Data Requirements

### Feature Extraction Script

```python
# extract_tls_features.py

import pandas as pd
import numpy as np
from scapy.all import rdpcap, TLS, TCP, IP, IPv6
import json

def extract_flow_features(pcap_file, protocol_labels):
    """
    Extract features from PCAP for TLS protocol classification
    
    Args:
        pcap_file: Path to PCAP file
        protocol_labels: Dict mapping port/protocol to label
    
    Returns:
        features: numpy array [num_flows, packets_per_flow, num_features]
        labels: list of protocol labels
    """
    packets = rdpcap(pcap_file)
    flows = {}  # flow_key -> list of features
    
    for pkt in packets:
        if not pkt.haslayer(TCP):
            continue
        
        tcp = pkt[TCP]
        flow_key = (pkt[IP].src, pkt[IP].dst, tcp.sport, tcp.dport)
        
        if flow_key not in flows:
            flows[flow_key] = []
        
        # Extract single packet features
        features = extract_packet_feature_vector(pkt)
        flows[flow_key].append(features)
    
    return flows

def extract_packet_feature_vector(pkt):
    """Extract features from a single packet"""
    features = np.zeros(15, dtype=np.float32)
    
    # Feature 0: Direction (0=c2s, 1=s2c)
    # Note: direction determined at flow level
    
    # Feature 1: Packet length
    features[1] = len(pkt)
    
    # Feature 2: Payload length
    if pkt.haslayer(TCP):
        features[2] = len(pkt[TCP].payload)
    
    # Feature 3: IP total length
    if pkt.haslayer(IP):
        features[3] = pkt[IP].len
    elif pkt.haslayer(IPv6):
        features[3] = pkt[IPv6].plen
    
    # Feature 4: TTL
    if pkt.haslayer(IP):
        features[4] = pkt[IP].ttl
    elif pkt.haslayer(IPv6):
        features[4] = pkt[IPv6].hlim
    
    # Feature 5: Relative timestamp (computed per flow)
    
    # Feature 6: Inter-arrival time (computed per flow)
    
    # Feature 7: TCP window
    if pkt.haslayer(TCP):
        features[7] = pkt[TCP].window
    
    # Feature 8: TCP header length
    if pkt.haslayer(TCP):
        features[8] = pkt[TCP].dataofs * 4
    
    # Feature 9: TCP flags
    if pkt.haslayer(TCP):
        flags = 0
        if pkt[TCP].flags.S: flags |= 1
        if pkt[TCP].flags.A: flags |= 2
        if pkt[TCP].flags.P: flags |= 4
        if pkt[TCP].flags.R: flags |= 8
        if pkt[TCP].flags.F: flags |= 16
        if pkt[TCP].flags.U: flags |= 32
        features[9] = flags
    
    # Feature 10: TCP sequence delta (computed)
    # Feature 11: TCP ACK delta (computed)
    
    # Feature 12: TLS record type
    if pkt.haslayer(TLS):
        tls = pkt[TLS]
        features[12] = get_tls_record_type(tls)
    
    # Feature 13: TLS record length
    if pkt.haslayer(TLS) and len(tls.load) >= 3:
        features[13] = int.from_bytes(tls.load[3:5], 'big')
    
    # Feature 14: TLS version
    if pkt.haslayer(TLS) and len(tls.load) >= 2:
        version = int.from_bytes(tls.load[1:3], 'big')
        features[14] = map_tls_version(version)
    
    return features

def get_tls_record_type(tls):
    """Map TLS record type to numeric value"""
    # TLS record types
    type_map = {
        'change_cipher': 3,
        'alert': 4,
        'handshake': 1,  # generic
        'application_data': 2,
        'heartbeat': 11
    }
    record_type_str = str(type(tls) if tls else '')
    for key, val in type_map.items():
        if key in record_type_str:
            return val
    return 0

def map_tls_version(version_bytes):
    """Map TLS version bytes to numeric value"""
    version_map = {
        0x0300: 1,  # SSLv3
        0x0301: 2,  # TLS 1.0
        0x0302: 3,  # TLS 1.1
        0x0303: 4,  # TLS 1.2
        0x0304: 5,  # TLS 1.3
    }
    return version_map.get(version_bytes, 0)
```

## CNN Model Architecture

```python
# train_cnn.py

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

class TLSProtocolDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class TLSProtocolCNN(nn.Module):
    def __init__(self, input_size=15, hidden_size=64, num_classes=7, sequence_length=10):
        super(TLSProtocolCNN, self).__init__()
        
        self.conv1 = nn.Conv1d(input_size, hidden_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc1 = nn.Linear(hidden_size, hidden_size // 2)
        self.fc2 = nn.Linear(hidden_size // 2, num_classes)
        
    def forward(self, x):
        # x shape: [batch, sequence, features]
        x = x.transpose(1, 2)  # [batch, features, sequence]
        
        x = self.conv1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.conv3(x)
        x = self.relu(x)
        
        x = self.pool(x).squeeze(-1)  # [batch, hidden_size]
        
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        
        return x

def train_cnn_model(train_loader, num_classes=7, epochs=100):
    model = TLSProtocolCNN(num_classes=num_classes)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    for epoch in range(epochs):
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
    
    return model

def export_to_onnx(model, output_path):
    model.eval()
    dummy_input = torch.randn(1, 10, 15)  # [batch, sequence, features]
    torch.onnx.export(model, dummy_input, output_path,
                      input_names=['input'],
                      output_names=['output'],
                      dynamic_axes={'input': {0: 'batch_size'},
                                   'output': {0: 'batch_size'}})
```

## LSTM Model Architecture

```python
# train_lstm.py

import torch
import torch.nn as nn

class TLSProtocolLSTM(nn.Module):
    def __init__(self, input_size=15, hidden_size=64, num_layers=2, num_classes=7):
        super(TLSProtocolLSTM, self).__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=0.3
        )
        
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)
        
    def forward(self, x):
        # LSTM output
        lstm_out, _ = self.lstm(x)  # [batch, seq, hidden*2]
        
        # Take last output
        out = lstm_out[:, -1, :]
        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.fc2(out)
        
        return out
```

## Transformer Model Architecture

```python
# train_transformer.py

import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class TLSProtocolTransformer(nn.Module):
    def __init__(self, input_size=15, d_model=64, nhead=4, num_layers=2, num_classes=7, sequence_length=10):
        super(TLSProtocolTransformer, self).__init__()
        
        self.embedding = nn.Linear(input_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=sequence_length)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=128),
            num_layers
        )
        self.fc_out = nn.Linear(d_model, num_classes)
        
    def forward(self, x):
        # x: [batch, seq, features]
        x = self.embedding(x)  # [batch, seq, d_model]
        x = self.pos_encoder(x)
        x = x.transpose(0, 1)  # [seq, batch, d_model]
        x = self.transformer(x)
        x = x[-1]  # Take last token [batch, d_model]
        x = self.fc_out(x)
        return x
```

## Feature Normalization

```python
# normalize_features.py

import json
import numpy as np
from sklearn.preprocessing import StandardScaler

def compute_scaler(features_array, output_path):
    """
    Compute normalization parameters and save to JSON
    
    Args:
        features_array: numpy array [num_flows, packets, features]
        output_path: path to save scaler.json
    """
    # Flatten for fitting
    features_flat = features_array.reshape(-1, features_array.shape[-1])
    
    scaler = StandardScaler()
    scaler.fit(features_flat)
    
    scaler_data = {
        "mean": scaler.mean_.tolist(),
        "std": scaler.scale_.tolist(),
        "feature_names": [
            "direction", "packet_len", "payload_len", "ip_total_len", "ttl",
            "rel_timestamp", "inter_arrival_time", "tcp_window", "tcp_hdr_len",
            "tcp_flags", "tcp_seq_delta", "tcp_ack_delta", "tls_record_type",
            "tls_record_len", "tls_version"
        ]
    }
    
    with open(output_path, 'w') as f:
        json.dump(scaler_data, f, indent=2)
    
    return scaler

def apply_normalization(features_array, scaler_path):
    """Apply saved normalization to features"""
    with open(scaler_path, 'r') as f:
        scaler_data = json.load(f)
    
    mean = np.array(scaler_data['mean'])
    std = np.array(scaler_data['std'])
    
    features_normalized = (features_array - mean) / std
    features_normalized = np.nan_to_num(features_normalized, nan=0.0, posinf=0.0, neginf=0.0)
    
    return features_normalized
```

## Training Pipeline Script

```python
# run_training.py

import argparse
import json
from sklearn.model_selection import train_test_split
import numpy as np

def main():
    parser = argparse.ArgumentParser(description='Train TLS Protocol Classifier')
    parser.add_argument('--model', choices=['cnn', 'lstm', 'transformer'], required=True)
    parser.add_argument('--pcap-dir', required=True, help='Directory with protocol PCAPs')
    parser.add_argument('--output-dir', required=True, help='Output directory for models')
    parser.add_argument('--packets-per-flow', type=int, default=10)
    
    args = parser.parse_args()
    
    # Protocol to label mapping
    protocol_map = {
        'https': 0,
        'mqtts': 1,
        'dot': 2,
        'ldaps': 3,
        'imaps': 4,
        'smtps': 5,
        'amqps': 6
    }
    
    # Extract features from all PCAPs
    all_features = []
    all_labels = []
    
    for protocol, label in protocol_map.items():
        pcap_path = f"{args.pcap_dir}/{protocol}/*.pcap"
        features = extract_flow_features(pcap_path)
        all_features.extend(features)
        all_labels.extend([label] * len(features))
    
    # Convert to arrays
    X = np.array(all_features)
    y = np.array(all_labels)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.1, random_state=42, stratify=y
    )
    
    # Normalize features
    scaler = compute_scaler(X_train, f"{args.output_dir}/scaler.json")
    X_train_norm = apply_normalization(X_train, f"{args.output_dir}/scaler.json")
    X_test_norm = apply_normalization(X_test, f"{args.output_dir}/scaler.json")
    
    # Create data loaders
    train_dataset = TLSProtocolDataset(X_train_norm, y_train)
    test_dataset = TLSProtocolDataset(X_test_norm, y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    # Train model
    if args.model == 'cnn':
        model = train_cnn_model(train_loader, num_classes=len(protocol_map))
    elif args.model == 'lstm':
        model = train_lstm_model(train_loader, num_classes=len(protocol_map))
    else:  # transformer
        model = train_transformer_model(train_loader, num_classes=len(protocol_map))
    
    # Export to ONNX
    export_to_onnx(model, f"{args.output_dir}/tls_{args.model}.onnx")
    
    # Evaluate
    evaluate_model(model, test_loader)

if __name__ == '__main__':
    main()
```

## Model Export Verification

```python
# verify_model.py

import onnxruntime as ort
import numpy as np

def verify_onnx_model(model_path):
    """Verify ONNX model can be loaded and produces correct output shape"""
    session = ort.InferenceSession(model_path)
    
    # Get input/output info
    input_info = session.get_inputs()[0]
    output_info = session.get_outputs()[0]
    
    print(f"Input: {input_info.name}, shape: {input_info.shape}")
    print(f"Output: {output_info.name}, shape: {output_info.shape}")
    
    # Test inference
    dummy_input = np.random.randn(1, 10, 15).astype(np.float32)
    outputs = session.run(None, {'input': dummy_input})
    
    print(f"Output shape: {outputs[0].shape}")
    
    # Verify softmax output
    if outputs[0].shape[1] > 1:
        probs = outputs[0][0]
        assert np.abs(probs.sum() - 1.0) < 0.01, "Output should sum to 1 for softmax"
```

## Confidence Threshold Tuning

```python
# tune_threshold.py

import numpy as np
from sklearn.metrics import precision_recall_curve

def find_optimal_threshold(y_true, y_scores, metric='f1'):
    """
    Find optimal confidence threshold for classification
    
    Args:
        y_true: true labels
        y_scores: prediction scores/probabilities
        metric: optimization metric
    
    Returns:
        optimal threshold
    """
    thresholds = np.arange(0.1, 1.0, 0.05)
    
    if metric == 'f1':
        precisions, recalls, thresh = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
        best_idx = np.argmax(f1_scores)
        return thresh[best_idx]
    
    return 0.5  # Default
```

## Feature Matching Verification Script

```python
# verify_feature_equivalence.py

"""
Verify that online feature extraction matches offline dataset generation
"""

import numpy as np
import json

def compare_feature_tensors(online_features, offline_features, tolerance=1e-5):
    """
    Compare feature tensors from online vs offline extraction
    
    Args:
        online_features: Features extracted by nDPI runtime
        offline_features: Features from training pipeline
        tolerance: Floating point comparison tolerance
    
    Returns:
        bool: True if tensors match within tolerance
    """
    if online_features.shape != offline_features.shape:
        print(f"Shape mismatch: {online_features.shape} vs {offline_features.shape}")
        return False
    
    diff = np.abs(online_features - offline_features)
    max_diff = diff.max()
    
    if max_diff > tolerance:
        print(f"Max difference: {max_diff}")
        print(f"Mismatched indices: {np.where(diff > tolerance)}")
        return False
    
    return True

def verify_packet_order(online_features, offline_features):
    """
    Verify packet order matches between online and offline
    """
    # Check direction column (should match)
    online_dirs = online_features[:, :, 0]
    offline_dirs = offline_features[:, :, 0]
    
    if not np.array_equal(online_dirs, offline_dirs):
        print("Packet direction order mismatch!")
        return False
    
    return True