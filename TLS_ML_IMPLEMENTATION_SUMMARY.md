# TLS ML Protocol Detection - Implementation Summary

## Deliverables

This technical specification provides a complete implementation plan for adding ML-based TLS protocol detection to nDPI. Two documents have been created:

### 1. `nDPI_TLS_ML_DETECTION_TECH_SPEC.md`
Main technical specification containing:
- Architecture and component diagrams
- Data structures (ndpi_flow_ml_state, ndpi_tls_ml_config)
- API interface definitions
- Feature extraction implementation
- ONNX Runtime integration
- Configuration file schemas
- Testing framework
- Build integration instructions

### 2. `nDPI_TLS_ML_TRAINING_GUIDE.md`
Training guide including:
- Feature extraction scripts
- CNN, LSTM, and Transformer model architectures
- Normalization implementation
- Training pipeline
- Model verification scripts

## Key Implementation Points

### Feature Set (15 features per packet)
1. **packet direction** (client-to-server / server-to-client)
2. **packet length** (total IP packet size)
3. **payload length** (data after TCP header)
4. **IP total length** (from IP header)
5. **TTL** (time to live from IP header)
6. **relative timestamp** (ms from flow start)
7. **inter-arrival time** (ms since last packet)
8. **TCP window size**
9. **TCP header length**
10. **TCP flags** (combined syn/ack/psh/rst/fin/urg)
11. **TCP sequence delta**
12. **TCP acknowledgement delta**
13. **TLS record type** (handshake/app_data/alert/etc.)
14. **TLS record length**
15. **TLS version** (SSLv3/TLS1.0/1.1/1.2/1.3)

### Supported Protocols
- HTTPS (port 443)
- MQTTS (port 8883, 443)
- DoT (port 853)
- LDAPS (port 636)
- IMAPS (port 993)
- SMTPS (port 465, 587)
- AMQPS (port 5671)

### Model Architecture Support
- **CNN**: 1D convolutions over packet sequence
- **LSTM**: Bidirectional LSTM with attention
- **Transformer**: Positional encoding + transformer encoder

### Integration Points
1. **Configuration**: Add to `ndpi_detection_module_config_struct`
2. **Flow state**: Extend `ndpi_flow_struct` with ML state
3. **TLS dissector**: Modify `ndpi_search_tls_wrapper()` in `tls.c`
4. **Build system**: Optional ONNX Runtime dependency

### Performance Targets
- O(1) feature extraction per packet
- Single inference per flow
- Minimal memory overhead (pre-allocated buffers)
- Lock-free operation (per-flow state)
- < 1ms inference latency target

## Files to Create/Modify

### New Files
```
src/lib/ndpi_tls_ml.c          # ML inference implementation
src/include/ndpi_tls_ml.h      # Header (if needed)
models/tls_cnn.onnx            # Trained CNN model
models/tls_lstm.onnx           # Trained LSTM model
models/tls_transformer.onnx      # Trained Transformer model
models/scaler.json             # Normalization parameters
tests/unit/tls_ml_test.c       # Unit tests
tests/integration/tls_ml_test.c # Integration tests
```

### Modified Files
```
src/include/ndpi_private.h     # Add ndpi_flow_ml_state to ndpi_flow_struct
src/include/ndpi_api.h         # Add public API functions
src/lib/protocols/tls.c        # Integrate ML inference
src/lib/Makefile.am            # Add ndpi_tls_ml.c
configure.ac                   # Add ONNX Runtime option
```

## Testing Strategy

### Unit Tests
- Packet direction calculation
- TLS record type extraction
- Feature normalization
- Model loading/unloading

### Integration Tests
- End-to-end PCAP processing
- Prediction correctness verification
- Feature tensor equivalence (online vs offline)

### Regression Tests
- Reference PCAP corpus
- Expected prediction comparison
- Performance benchmarks

## Next Steps

1. **Review and approve** the technical specification
2. **Implement** the core ML module (`ndpi_tls_ml.c`)
3. **Integrate** with TLS dissector
4. **Train** initial models with labeled data
5. **Test** with reference PCAPs
6. **Benchmark** performance impact