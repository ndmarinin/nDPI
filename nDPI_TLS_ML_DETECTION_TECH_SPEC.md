# nDPI TLS-based Protocol Detection with Machine Learning - Technical Specification

## Project Overview

This document describes the technical implementation of an ML-based TLS protocol detection module for nDPI that can classify TLS-based protocols (HTTPS, MQTTS, DoT, LDAPS, IMAPS, SMTPS, AMQPS) without decrypting traffic.

## Architecture

### Module Integration Point

The module integrates into the nDPI flow processing pipeline at the TLS dissector level (`src/lib/protocols/tls.c`), specifically in the `ndpi_search_tls_wrapper()` function after TLS handshake detection.

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                         ndpi_detection_process_packet()              │
│                              (main entry)                            │
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                           TLS Dissector                             │
│                      (ndpi_search_tls_wrapper)                        │
├─────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐ │
│  │ Handshake Phase │───▶│ Feature Buffer  │───▶│ ML Classifier   │ │
│  │ Detection       │    │ (per flow)      │    │ (CNN/LSTM/Trans)│ │
│  └─────────────────┘    └─────────────────┘    └─────────────────┘ │
│                                  │                       │        │
│                                  ▼                         │        │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ Packet Feature Extractor                                          ││
│  │ - Direction, Length, TTL, Timestamps                              ││
│  │ - TCP deltas, TLS metadata                                        ││
│  └─────────────────────────────────────────────────────────────────┘│
│                                  │                                 │
│                                  ▼                                 │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │ Feature Preprocessor (Normalization)                              ││
│  │ - Standardization (z-score)                                       ││
│  │ - Categorical encoding                                            ││
│  │ - Zero padding                                                    ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                                  │
                                  ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    Prediction Result Storage                        │
│  - protocol_id, confidence, inference_time                           │
└─────────────────────────────────────────────────────────────────────┘
```

## Data Structures

### Configuration Structure (`ndpi_tls_ml_config`)

```c
// Location: src/include/ndpi_private.h

typedef struct ndpi_tls_ml_config {
    /* Model configuration */
    int model_type;           /* 0=CNN, 1=LSTM, 2=Transformer */
    char model_path[256];     /* Path to ONNX/TorchScript model */
    int num_threads;          /* Inference threads (default: 1) */
    
    /* Feature extraction */
    int packets_per_flow;       /* N packets to collect (default: 10) */
    char scaler_path[256];    /* Path to normalization scaler JSON */
    
    /* Runtime settings */
    int enabled;              /* Enable/disable ML classification */
    int log_level;            /* Debug logging level */
    int confidence_threshold; /* Minimum confidence for valid prediction */
} ndpi_tls_ml_config_t;
```

### Flow ML State Structure (`ndpi_flow_ml_state`)

```c
// Location: src/include/ndpi_typedefs.h (within ndpi_flow_struct extension)

typedef struct ndpi_flow_ml_state {
    /* Packet buffer for feature extraction */
    float packet_features[MAX_PACKETS_PER_FLOW][NUM_FEATURES_PER_PACKET];
    u_int16_t num_packets_collected;
    
    /* Inference state */
    u_int8_t inference_performed:1;
    u_int8_t handshake_complete:1;
    u_int8_t _pad:6;
    
    /* Prediction results */
    u_int16_t predicted_protocol_id;
    float confidence_score;
    u_int32_t inference_time_us;  /* Microseconds */
    
    /* Timestamps for inter-arrival calculation */
    u_int64_t last_packet_time_ms;
    u_int64_t first_packet_time_ms;
    
    /* TCP sequence tracking */
    u_int32_t last_seq[2];  /* [0]=c2s, [1]=s2c */
    u_int32_t last_ack[2];
} ndpi_flow_ml_state_t;
```

### Feature Vector Definition

```c
/* Features per packet (order MUST match training dataset) */
#define NUM_FEATURES_PER_PACKET 15

typedef enum {
    FEAT_DIRECTION = 0,        /* 0=client->server, 1=server->client */
    FEAT_PACKET_LEN,           /* Total packet length */
    FEAT_PAYLOAD_LEN,          /* Payload length (IP total - headers) */
    FEAT_IP_TOTAL_LEN,         /* IP total length field */
    FEAT_TTL,                  /* Time to live */
    FEAT_REL_TIMESTAMP,        /* Relative timestamp from flow start (ms) */
    FEAT_INTER_ARRIVAL_TIME,   /* Time since last packet (ms) */
    FEAT_TCP_WINDOW,           /* TCP window size */
    FEAT_TCP_HDR_LEN,          /* TCP header length */
    FEAT_TCP_FLAGS,          /* TCP flags (syn, ack, psh, rst, fin, urg) */
    FEAT_TCP_SEQ_DELTA,      /* Sequence number delta */
    FEAT_TCP_ACK_DELTA,      /* Acknowledgement number delta */
    FEAT_TLS_RECORD_TYPE,    /* TLS record type (handshake, app_data, alert) */
    FEAT_TLS_RECORD_LEN,     /* TLS record length */
    FEAT_TLS_VERSION         /* TLS version (TLS1.0=1, TLS1.1=2, TLS1.2=3, TLS1.3=4) */
} ndpi_packet_feature_t;
```

## API Interface

### New Public Functions (ndpi_api.h)

```c
/**
 * Load ML model for TLS protocol detection
 * 
 * @param ndpi_str Detection module
 * @param model_path Path to model file (ONNX format)
 * @param model_type Model type (0=CNN, 1=LSTM, 2=Transformer)
 * @return 0 on success, -1 on error
 */
int ndpi_load_tls_ml_model(struct ndpi_detection_module_struct *ndpi_str,
                           const char *model_path, int model_type);

/**
 * Configure TLS ML detection parameters
 * 
 * @param ndpi_str Detection module
 * @param packets_per_flow Number of packets to collect per flow
 * @param scaler_path Path to normalization scaler
 * @return 0 on success, -1 on error
 */
int ndpi_configure_tls_ml(struct ndpi_detection_module_struct *ndpi_str,
                          int packets_per_flow,
                          const char *scaler_path);

/**
 * Get TLS ML prediction for a flow
 * 
 * @param flow Flow to query
 * @param protocol_id Output: predicted protocol ID
 * @param confidence Output: confidence score (0.0-1.0)
 * @param inference_time_us Output: inference time in microseconds
 * @return 1 if prediction available, 0 if not yet performed
 */
int ndpi_get_tls_ml_prediction(struct ndpi_flow_struct *flow,
                               u_int16_t *protocol_id,
                               float *confidence,
                               u_int32_t *inference_time_us);

/**
 * Check if TLS ML inference has been performed for flow
 * 
 * @param flow Flow to check
 * @return 1 if inference done, 0 otherwise
 */
static inline int ndpi_tls_ml_performed(const struct ndpi_flow_struct *flow) {
    return flow->ml_state.inference_performed;
}
```

## Implementation Files

### 1. Core ML Module (`src/lib/ndpi_tls_ml.c`)

```c
/*
 * ndpi_tls_ml.c - TLS ML-based protocol detection
 * 
 * Main implementation file for ML inference
 */

#include "ndpi_private.h"
#include "ndpi_api.h"
#include <onnxruntime_c_api.h>  /* ONNX Runtime */

/* Internal structures */
struct ndpi_tls_ml_runtime {
    OrtEnv *ort_env;
    OrtSession *session;
    OrtAllocator *allocator;
    int model_type;
    int packets_per_flow;
    float scaler_mean[NUM_FEATURES_PER_PACKET];
    float scaler_std[NUM_FEATURES_PER_PACKET];
};

/* Global ML runtime (per detection module) */
static struct ndpi_tls_ml_runtime *tls_ml_runtime = NULL;

/* Feature extraction functions */
static void extract_packet_features(struct ndpi_detection_module_struct *ndpi_struct,
                                   struct ndpi_flow_struct *flow,
                                   float *features);

static int calculate_packet_direction(struct ndpi_detection_module_struct *ndpi_struct,
                                       struct ndpi_flow_struct *flow);

static u_int8_t extract_tls_record_type(const u_int8_t *payload, u_int16_t payload_len);

static u_int16_t extract_tls_version(const u_int8_t *payload, u_int16_t payload_len);

/* Normalization functions */
static void normalize_features(float *features, int num_packets,
                             const float *mean, const float *std,
                             int num_features);

/* Inference functions */
static int run_inference(struct ndpi_detection_module_struct *ndpi_struct,
                         struct ndpi_flow_struct *flow,
                         float *input_tensor, int *output_class,
                         float *output_confidence);

/* Model loading */
static int load_onnx_model(const char *model_path);
static int load_scaler(const char *scaler_path);
```

### 2. Feature Extraction Implementation

```c
/**
 * Extract features from a single packet
 */
static void extract_packet_features(struct ndpi_detection_module_struct *ndpi_struct,
                                   struct ndpi_flow_struct *flow,
                                   float *features) {
    struct ndpi_packet_struct *packet = &ndpi_struct->packet;
    struct ndpi_flow_ml_state *ml_state = &flow->ml_state;
    
    /* Feature 0: Direction (0=client->server, 1=server->client) */
    features[FEAT_DIRECTION] = calculate_packet_direction(ndpi_struct, flow);
    
    /* Feature 1: Packet length */
    features[FEAT_PACKET_LEN] = (float)ndpi_struct->packet.l3_packet_len;
    
    /* Feature 2: Payload length */
    features[FEAT_PAYLOAD_LEN] = (float)ndpi_struct->packet.payload_packet_len;
    
    /* Feature 3: IP total length */
    if (packet->iph) {
        features[FEAT_IP_TOTAL_LEN] = (float)ntohs(packet->iph->tot_len);
    } else if (packet->iphv6) {
        features[FEAT_IP_TOTAL_LEN] = (float)ntohs(packet->iphv6->ip6_hdr.ip6_un1_plen);
    } else {
        features[FEAT_IP_TOTAL_LEN] = 0.0f;
    }
    
    /* Feature 4: TTL */
    if (packet->iph) {
        features[FEAT_TTL] = (float)packet->iph->ttl;
    } else if (packet->iphv6) {
        features[FEAT_TTL] = (float)packet->iphv6->ip6_hdr.ip6_un1_hlim;
    } else {
        features[FEAT_TTL] = 0.0f;
    }
    
    /* Feature 5: Relative timestamp */
    features[FEAT_REL_TIMESTAMP] = (float)(ndpi_struct->current_ts - ml_state->first_packet_time_ms);
    
    /* Feature 6: Inter-arrival time */
    features[FEAT_INTER_ARRIVAL_TIME] = (float)(ndpi_struct->current_ts - ml_state->last_packet_time_ms);
    
    /* Feature 7: TCP window size */
    if (packet->tcp) {
        features[FEAT_TCP_WINDOW] = (float)ntohs(packet->tcp->window);
    } else {
        features[FEAT_TCP_WINDOW] = 0.0f;
    }
    
    /* Feature 8: TCP header length */
    if (packet->tcp) {
        features[FEAT_TCP_HDR_LEN] = (float)(packet->tcp->doff * 4);
    } else {
        features[FEAT_TCP_HDR_LEN] = 0.0f;
    }
    
    /* Feature 9: TCP flags (encoded as sum of set flags) */
    if (packet->tcp) {
        u_int8_t flags = 0;
        if (packet->tcp->syn) flags |= 1;
        if (packet->tcp->ack) flags |= 2;
        if (packet->tcp->psh) flags |= 4;
        if (packet->tcp->rst) flags |= 8;
        if (packet->tcp->fin) flags |= 16;
        if (packet->tcp->urg) flags |= 32;
        features[FEAT_TCP_FLAGS] = (float)flags;
    } else {
        features[FEAT_TCP_FLAGS] = 0.0f;
    }
    
    /* Feature 10: TCP sequence delta */
    if (packet->tcp) {
        u_int32_t seq = ntohl(packet->tcp->seq);
        u_int8_t direction = (int)features[FEAT_DIRECTION];
        features[FEAT_TCP_SEQ_DELTA] = (float)(seq - ml_state->last_seq[direction]);
        ml_state->last_seq[direction] = seq;
    } else {
        features[FEAT_TCP_SEQ_DELTA] = 0.0f;
    }
    
    /* Feature 11: TCP ACK delta */
    if (packet->tcp) {
        u_int32_t ack = ntohl(packet->tcp->ack_seq);
        features[FEAT_TCP_ACK_DELTA] = (float)(ack - ml_state->last_ack[!((int)features[FEAT_DIRECTION]]);
        ml_state->last_ack[!((int)features[FEAT_DIRECTION])] = ack;
    } else {
        features[FEAT_TCP_ACK_DELTA] = 0.0f;
    }
    
    /* Feature 12: TLS record type */
    features[FEAT_TLS_RECORD_TYPE] = (float)extract_tls_record_type(
        packet->payload, ndpi_struct->packet.payload_packet_len);
    
    /* Feature 13: TLS record length */
    /* Extract from first TLS record in payload */
    if (ndpi_struct->packet.payload_packet_len >= 5) {
        features[FEAT_TLS_RECORD_LEN] = (float)(
            (packet->payload[3] << 8) | packet->payload[4]);
    } else {
        features[FEAT_TLS_RECORD_LEN] = 0.0f;
    }
    
    /* Feature 14: TLS version */
    features[FEAT_TLS_VERSION] = (float)extract_tls_version(
        packet->payload, ndpi_struct->packet.payload_packet_len);
}
```

### 3. TLS Record Type Extraction

```c
/**
 * Extract TLS record type from payload
 * Returns: 0=unknown, 1=handshake, 2=application_data, 3=alert, etc.
 */
static u_int8_t extract_tls_record_type(const u_int8_t *payload, u_int16_t payload_len) {
    if (payload_len < 5) return 0;
    
    /* TLS record type is the first byte */
    switch (payload[0]) {
        case 0x14: return 3;  /* change_cipher */
        case 0x15: return 4;  /* alert */
        case 0x16: {         /* handshake */
            if (payload_len >= 10) {
                /* Handshake type is at offset 5 + 1 */
                switch (payload[9]) {
                    case 0x01: return 5;  /* client_hello */
                    case 0x02: return 6;  /* server_hello */
                    case 0x0b: return 7;  /* certificate */
                    case 0x0e: return 8;  /* server_hello_done */
                    case 0x10: return 9;  /* client_finished */
                    case 0x14: return 10; /* server_finished */
                    default: return 1;
                }
            }
            return 1;
        }
        case 0x17: return 2;  /* application_data */
        case 0x18: return 11; /* heartbeat */
        default: return 0;
    }
}

/**
 * Extract TLS version from payload
 * Returns: 0=unknown, 1=SSLv3, 2=TLS1.0, 3=TLS1.1, 4=TLS1.2, 5=TLS1.3
 */
static u_int16_t extract_tls_version(const u_int8_t *payload, u_int16_t payload_len) {
    if (payload_len < 3) return 0;
    
    u_int16_t version = (payload[1] << 8) | payload[2];
    
    switch (version) {
        case 0x0300: return 1;  /* SSLv3 */
        case 0x0301: return 2;  /* TLS 1.0 */
        case 0x0302: return 3;  /* TLS 1.1 */
        case 0x0303: return 4;  /* TLS 1.2 */
        case 0x0304: return 5;  /* TLS 1.3 */
        default: return 0;
    }
}
```

### 4. Integration with TLS Dissector

```c
// Location: src/lib/protocols/tls.c (modified)

static void ndpi_search_tls_wrapper(struct ndpi_detection_module_struct *ndpi_struct,
                                   struct ndpi_flow_struct *flow) {
    struct ndpi_flow_ml_state *ml_state = &flow->ml_state;
    
    /* Existing TLS processing (unchanged) */
    /* ... existing code ... */
    
    /* TLS ML integration */
    if (ndpi_struct->cfg.tls_ml_enabled && !ml_state->inference_performed) {
        /* Check if this is a TLS handshake packet */
        if (is_tls_handshake_packet(ndpi_struct)) {
            /* Mark handshake as seen */
            if (!ml_state->handshake_complete) {
                if (is_client_hello(ndpi_struct)) {
                    ml_state->handshake_complete = 1;
                    ml_state->first_packet_time_ms = ndpi_struct->current_ts;
                }
            }
        }
        
        /* Collect features for TLS flows */
        if (ml_state->handshake_complete || ndpi_struct->proto_defaults[NDPI_PROTOCOL_TLS].isClearTextProto == 0) {
            if (ml_state->num_packets_collected < ndpi_struct->cfg.tls_ml_packets_per_flow) {
                extract_packet_features(ndpi_struct, flow, 
                    ml_state->packet_features[ml_state->num_packets_collected]);
                ml_state->num_packets_collected++;
            } else if (ml_state->num_packets_collected == ndpi_struct->cfg.tls_ml_packets_per_flow) {
                /* Initialize first timestamp if not set */
                if (ml_state->first_packet_time_ms == 0) {
                    ml_state->first_packet_time_ms = ndpi_struct->current_ts;
                }
                ml_state->last_packet_time_ms = ndpi_struct->current_ts;
                
                /* Run inference */
                run_inference(ndpi_struct, flow, NULL, NULL, NULL);
                ml_state->inference_performed = 1;
                
                /* Update flow classification if confidence is high enough */
                if (ml_state->confidence_score >= ndpi_struct->cfg.tls_ml_confidence_threshold) {
                    ndpi_set_detected_protocol(ndpi_struct, flow,
                        ml_state->predicted_protocol_id, NDPI_PROTOCOL_TLS,
                        ndpi_confidence_to_ndpi(ml_state->confidence_score));
                }
            }
        }
    }
    
    /* Continue with existing TLS processing */
    /* ... existing code ... */
}
```

### 5. ONNX Runtime Integration

```c
/**
 * Run ONNX model inference
 */
static int run_inference(struct ndpi_detection_module_struct *ndpi_struct,
                         struct ndpi_flow_struct *flow,
                         float *input_tensor, int *output_class,
                         float *output_confidence) {
    struct ndpi_flow_ml_state *ml_state = &flow->ml_state;
    struct ndpi_tls_ml_runtime *runtime = tls_ml_runtime;
    
    if (!runtime || !runtime->session) {
        return -1;
    }
    
    u_int64_t start_ts = ndpi_get_current_time_ms();
    
    /* Prepare input tensor */
    int64_t input_dims[3] = {1, ndpi_struct->cfg.tls_ml_packets_per_flow, NUM_FEATURES_PER_PACKET};
    float input_data[ndpi_struct->cfg.tls_ml_packets_per_flow * NUM_FEATURES_PER_PACKET];
    
    /* Copy and pad features */
    int actual_packets = ml_state->num_packets_collected;
    memcpy(input_data, ml_state->packet_features, 
           actual_packets * NUM_FEATURES_PER_PACKET * sizeof(float));
    
    /* Zero-pad if fewer than N packets */
    if (actual_packets < ndpi_struct->cfg.tls_ml_packets_per_flow) {
        memset(input_data + actual_packets * NUM_FEATURES_PER_PACKET, 0,
               (ndpi_struct->cfg.tls_ml_packets_per_flow - actual_packets) * NUM_FEATURES_PER_PACKET * sizeof(float));
    }
    
    /* Create input tensor */
    OrtValue *input_tensor_ptr = NULL;
    OrtCreateTensor(element_type_float, input_dims, 3, input_data, &input_tensor_ptr);
    
    /* Run inference */
    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};
    
    OrtValue *output_tensor_ptr = NULL;
    OrtRun(runtime->session, NULL, input_names, &input_tensor_ptr, 1, output_names, 1, &output_tensor_ptr);
    
    /* Extract results */
    float *output_data = NULL;
    OrtGetValueFloat(output_tensor_ptr, &output_data);
    
    /* Find predicted class and confidence */
    ml_state->predicted_protocol_id = ndpi_tls_ml_classes[(int)output_data[0]];
    ml_state->confidence_score = output_data[1];
    
    ml_state->inference_time_us = (u_int32_t)((ndpi_get_current_time_ms() - start_ts) * 1000);
    
    /* Cleanup */
    OrtReleaseValue(input_tensor_ptr);
    OrtReleaseValue(output_tensor_ptr);
    
    if (output_class) *output_class = ml_state->predicted_protocol_id;
    if (output_confidence) *output_confidence = ml_state->confidence_score;
    
    return 0;
}
```

## Configuration File Schema

### YAML Configuration (`models/tls_ml_config.yaml`)

```yaml
model:
  type: transformer           # Options: cnn, lstm, transformer
  path: models/tls_transformer.onnx
  threads: 1

features:
  packets_per_flow: 10        # Number of packets to analyze
  scaler: models/scaler.json  # Optional: path to normalization scaler

runtime:
  confidence_threshold: 0.7   # Minimum confidence for protocol classification
  enabled: true

logging:
  enabled: false              # Enable debug logging
  level: 1                    # 0=error, 1=trace, 2=debug
```

### JSON Scaler Format (`models/scaler.json`)

```json
{
  "mean": [250.5, 1400.2, 1200.8, 1500.0, 64.0, 0.0, 10.5, 65535.0, 20.0, 28.0, 0.0, 0.0, 1.0, 500.0, 3.0],
  "std": [100.3, 300.5, 200.0, 300.0, 20.0, 100.0, 50.0, 30000.0, 4.0, 15.0, 10000.0, 10000.0, 2.0, 300.0, 1.5],
  "feature_names": ["direction", "packet_len", "payload_len", "ip_total_len", "ttl", 
                    "rel_timestamp", "inter_arrival", "tcp_window", "tcp_hdr_len", 
                    "tcp_flags", "tcp_seq_delta", "tcp_ack_delta", "tls_record_type", 
                    "tls_record_len", "tls_version"]
}
```

## Testing Framework

### Unit Tests (`tests/unit/tls_ml_test.c`)

```c
/*
 * Unit tests for TLS ML feature extraction
 */

static void test_packet_direction_calculation(void) {
    struct ndpi_detection_module_struct *ndpi_struct;
    struct ndpi_flow_struct *flow;
    
    ndpi_struct = ndpi_init_detection_module(NULL);
    
    /* Test case 1: Client to server */
    memset(flow, 0, sizeof(*flow));
    flow->c_port = htons(54321);
    flow->s_port = htons(443);
    
    /* Mock packet from client port */
    ndpi_struct->packet.tcp = &(struct ndpi_tcphdr){
        .source = htons(54321),
        .dest = htons(443)
    };
    
    float features[NUM_FEATURES_PER_PACKET];
    extract_packet_features(ndpi_struct, flow, features);
    assert(features[FEAT_DIRECTION] == 0.0f);
    
    ndpi_exit_detection_module(ndpi_struct);
}

static void test_tls_record_extraction(void) {
    /* Test ClientHello */
    u_int8_t client_hello[] = {0x16, 0x03, 0x01, 0x00, 0x50, 0x01};
    assert(extract_tls_record_type(client_hello, sizeof(client_hello)) == 5); /* client_hello */
    
    /* Test Application Data */
    u_int8_t app_data[] = {0x17, 0x03, 0x03, 0x00, 0x10};
    assert(extract_tls_record_type(app_data, sizeof(app_data)) == 2); /* application_data */
}

static void test_normalization(void) {
    float features[] = {200, 1500, 1300, 1500, 64};
    float mean[] = {250, 1400, 1200, 1500, 64};
    float std[] = {50, 300, 200, 300, 0};
    
    normalize_features(features, 1, mean, std, 5);
    
    assert(fabsf(features[0] - (-1.0f)) < 0.001); /* (200-250)/50 = -1 */
}
```

### Integration Tests

```c
/*
 * Integration test using PCAP files
 */

static void test_https_classification(void) {
    const char *pcap_file = "tests/pcap/tls_https.pcap";
    struct ndpi_flow_struct flow;
    struct ndpi_detection_module_struct *ndpi_struct;
    
    ndpi_struct = ndpi_init_detection_module(NULL);
    ndpi_load_tls_ml_model(ndpi_struct, "models/https_cnn.onnx", 0); /* CNN */
    
    /* Replay PCAP */
    pcap_loop(pcap, process_pcap_packet_wrapper, &flow);
    
    /* Verify prediction */
    assert(ndpi_tls_ml_performed(&flow));
    assert(flow.ml_state.predicted_protocol_id == NDPI_PROTOCOL_TLS);
    assert(flow.ml_state.confidence_score > 0.8);
}
```

## Performance Considerations

### Memory Allocation Strategy

1. **Pre-allocated buffers**: All packet feature buffers are pre-allocated in `ndpi_flow_struct` to avoid dynamic allocation during packet processing
2. **Global model instance**: ML model is loaded once and shared across all flows
3. **Zero dynamic allocation**: Feature extraction uses stack allocation where possible

### Lock-Free Operation

```c
/* No locks needed - each flow has its own state */
struct ndpi_flow_ml_state {
    /* ... per-flow data ... */
} __attribute__((aligned(64))); /* Cache line alignment */
```

### O(1) Feature Extraction

Each feature extraction is O(1) - simple arithmetic and field access:
- No loops over packet data
- No string operations
- Direct field reads from packet headers

## Build Integration

### Makefile.am additions

```makefile
# src/lib/Makefile.am

if HAVE_ONNX_RUNTIME
libndpi_la_SOURCES += ndpi_tls_ml.c
libndpi_la_CFLAGS += $(ONNX_CFLAGS)
libndpi_la_LIBADD += $(ONNX_LIBS)
endif

# Conditional compilation
AM_CPPFLAGS += -DNDPI_TLS_ML_SUPPORT=1
```

### configure.ac additions

```autoconf
# configure.ac

AC_ARG_WITH([onnx-runtime],
  [AS_HELP_STRING([--with-onnx-runtime], [Enable ONNX Runtime support @<:@default=no@:>@])],
  [],
  [with_onnx_runtime=no])

AS_IF([test "x$with_onnx_runtime" = "xyes"],
  [AC_CHECK_HEADERS([onnxruntime_c_api.h])
   AC_CHECK_LIB([onnxruntime, onnxruntime])
   AC_DEFINE([HAVE_ONNX_RUNTIME], [1], [Define if ONNX Runtime is available])
  ])
```

## Supported Protocol IDs

```c
/* Protocol mapping for ML predictions */
static const u_int16_t ndpi_tls_ml_protocols[] = {
    NDPI_PROTOCOL_TLS,      /* 0 */
    NDPI_PROTOCOL_MQTTS,    /* 1 */
    NDPI_PROTOCOL_DoT,      /* 2 */
    NDPI_PROTOCOL_LDAPS,    /* 3 */
    NDPI_PROTOCOL_IMAPS,    /* 4 */
    NDPI_PROTOCOL_SMTPS,    /* 5 */
    NDPI_PROTOCOL_AMQPS     /* 6 */
};

#define NDPI_TLS_ML_NUM_PROTOCOLS (sizeof(ndpi_tls_ml_protocols) / sizeof(ndpi_tls_ml_protocols[0]))
```

## Error Handling

```c
/* Error codes */
typedef enum {
    NDPI_TLS_ML_SUCCESS = 0,
    NDPI_TLS_ML_MODEL_NOT_LOADED = -1,
    NDPI_TLS_ML_INVALID_INPUT = -2,
    NDPI_TLS_ML_INFERENCE_FAILED = -3,
    NDPI_TLS_ML_MEMORY_ERROR = -4
} ndpi_tls_ml_error_t;

static int handle_inference_error(struct ndpi_detection_module_struct *ndpi_struct,
                                 const char *error_msg) {
    NDPI_LOG_ERR(ndpi_struct, "%s", error_msg);
    return -1;
}
```

## Feature Tensor Format

### Expected Input Format (ONNX)

```
Input tensor shape: [1, N, 15]
- Batch dimension: 1 (single flow)
- Sequence dimension: N (packets per flow, configurable)
- Feature dimension: 15 (features per packet)

Output tensor shape: [1, M] where M = number of classes
- Softmax probabilities for each protocol class
```

### Tensor Construction Example

```python
# For model training compatibility
def create_feature_tensor(packets, n_packets=10):
    """
    Create feature tensor from packet list
    
    Args:
        packets: List of packet dictionaries with extracted features
        n_packets: Fixed number of packets for tensor padding
    
    Returns:
        numpy array of shape [1, n_packets, 15]
    """
    features = np.zeros((1, n_packets, 15), dtype=np.float32)
    
    for i, pkt in enumerate(packets[:n_packets]):
        features[0, i, 0] = pkt['direction']
        features[0, i, 1] = pkt['packet_len']
        features[0, i, 2] = pkt['payload_len']
        features[0, i, 3] = pkt['ip_total_len']
        features[0, i, 4] = pkt['ttl']
        features[0, i, 5] = pkt['rel_timestamp']
        features[0, i, 6] = pkt['inter_arrival_time']
        features[0, i, 7] = pkt['tcp_window']
        features[0, i, 8] = pkt['tcp_hdr_len']
        features[0, i, 9] = pkt['tcp_flags']
        features[0, i, 10] = pkt['tcp_seq_delta']
        features[0, i, 11] = pkt['tcp_ack_delta']
        features[0, i, 12] = pkt['tls_record_type']
        features[0, i, 13] = pkt['tls_record_len']
        features[0, i, 14] = pkt['tls_version']
    
    return features
```

## Deployment Checklist

- [ ] Add `ndpi_tls_ml.c` to build system
- [ ] Update `ndpi_flow_struct` with `ml_state` member
- [ ] Add configuration fields to `ndpi_detection_module_config_struct`
- [ ] Implement `ndpi_load_tls_ml_model()` and related APIs
- [ ] Integrate feature extraction into TLS dissector
- [ ] Add ONNX Runtime dependency (optional compile-time)
- [ ] Create model files in `models/` directory
- [ ] Add unit tests to test suite
- [ ] Update documentation
- [ ] Performance benchmarking