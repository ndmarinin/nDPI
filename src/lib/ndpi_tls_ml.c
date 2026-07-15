/*
 * ndpi_tls_ml.c - TLS ML-based protocol detection
 *
 * Copyright (C) 2016-26 - ntop.org
 *
 * nDPI is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * nDPI is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public License
 * along with nDPI.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "ndpi_protocol_ids.h"
#define NDPI_CURRENT_PROTO NDPI_PROTOCOL_TLS
#include "ndpi_api.h"
#include "ndpi_private.h"

#ifdef HAVE_ONNX_RUNTIME
#include <onnxruntime_c_api.h>
#endif

/* Feature indices for TLS ML packet features */
#define FEAT_DIRECTION          0
#define FEAT_PACKET_LEN         1
#define FEAT_PAYLOAD_LEN        2
#define FEAT_IP_TOTAL_LEN       3
#define FEAT_TTL                4
#define FEAT_REL_TIMESTAMP      5
#define FEAT_INTER_ARRIVAL_TIME 6
#define FEAT_TCP_WINDOW         7
#define FEAT_TCP_HDR_LEN        8
#define FEAT_TCP_FLAGS          9
#define FEAT_TCP_SEQ_DELTA      10
#define FEAT_TCP_ACK_DELTA      11
#define FEAT_TLS_RECORD_TYPE    12
#define FEAT_TLS_RECORD_LEN     13
#define FEAT_TLS_VERSION        14

/* TLS ML supported protocols mapping */
static const u_int16_t ndpi_tls_ml_protocols[] = {
    NDPI_PROTOCOL_TLS,           /* 0 - base TLS */
    NDPI_PROTOCOL_MQTT,          /* 1 */
    NDPI_PROTOCOL_DOH_DOT,       /* 2 - DoT */
    NDPI_PROTOCOL_LDAP,          /* 3 */
    NDPI_PROTOCOL_MAIL_IMAPS,    /* 4 */
    NDPI_PROTOCOL_MAIL_SMTPS,    /* 5 */
    NDPI_PROTOCOL_AMQP           /* 6 */
};

#define NDPI_TLS_ML_NUM_PROTOCOLS (sizeof(ndpi_tls_ml_protocols) / sizeof(ndpi_tls_ml_protocols[0]))

/* ONNX Runtime state (conditionally compiled) */
#ifdef HAVE_ONNX_RUNTIME
struct ndpi_tls_ml_runtime {
    OrtEnv *ort_env;
    OrtSession *session;
    OrtAllocator *allocator;
    int model_type;
    float scaler_mean[NUM_FEATURES_PER_PACKET_FOR_TLS_ML];
    float scaler_std[NUM_FEATURES_PER_PACKET_FOR_TLS_ML];
};

static struct ndpi_tls_ml_runtime *tls_ml_runtime = NULL;
#endif

/* Forward declarations */
static void extract_packet_features(struct ndpi_detection_module_struct *ndpi_struct,
                                   struct ndpi_flow_struct *flow,
                                   float *features);

static int calculate_packet_direction(const struct ndpi_detection_module_struct *ndpi_struct,
                                       const struct ndpi_flow_struct *flow);

static u_int8_t extract_tls_record_type(const u_int8_t *payload, u_int16_t payload_len);

static u_int16_t extract_tls_version(const u_int8_t *payload, u_int16_t payload_len);

/* Extract TLS record type from payload */
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

/* Extract TLS version from payload */
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

/* Calculate packet direction (0 = client->server, 1 = server->client) */
static int calculate_packet_direction(const struct ndpi_detection_module_struct *ndpi_struct,
                                       const struct ndpi_flow_struct *flow) {
    struct ndpi_packet_struct *packet = (struct ndpi_packet_struct *)&ndpi_struct->packet;
    
    if (packet->tcp) {
        u_int16_t src_port = ntohs(packet->tcp->source);
        if (src_port == flow->c_port) {
            return 0;  /* client -> server */
        } else {
            return 1;  /* server -> client */
        }
    }
    return 0;  /* Default to client->server */
}

/* Extract features from a single packet */
static void extract_packet_features(struct ndpi_detection_module_struct *ndpi_struct,
                                   struct ndpi_flow_struct *flow,
                                   float *features) {
    struct ndpi_packet_struct *packet = &ndpi_struct->packet;
    struct ndpi_flow_ml_state *ml_state = &flow->ml_state;
    
    /* Feature 0: Direction */
    features[FEAT_DIRECTION] = (float)calculate_packet_direction(ndpi_struct, flow);
    
    /* Feature 1: Packet length */
    features[FEAT_PACKET_LEN] = (float)packet->l3_packet_len;
    
    /* Feature 2: Payload length */
    features[FEAT_PAYLOAD_LEN] = (float)packet->payload_packet_len;
    
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
    
    /* Feature 9: TCP flags */
    if (packet->tcp) {
        u_int8_t flags = 0;
        /* Use bit fields for TCP flags */
#if defined(__LITTLE_ENDIAN__)
        if (packet->tcp->syn) flags |= 1;
        if (packet->tcp->ack) flags |= 2;
        if (packet->tcp->psh) flags |= 4;
        if (packet->tcp->rst) flags |= 8;
        if (packet->tcp->fin) flags |= 16;
        if (packet->tcp->urg) flags |= 32;
#else
        if (packet->tcp->cwr) flags |= 1;
        if (packet->tcp->ece) flags |= 2;
        if (packet->tcp->urg) flags |= 4;
        if (packet->tcp->ack) flags |= 8;
        if (packet->tcp->psh) flags |= 16;
        if (packet->tcp->rst) flags |= 32;
        if (packet->tcp->syn) flags |= 64;
        if (packet->tcp->fin) flags |= 128;
#endif
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
        u_int8_t direction = (int)features[FEAT_DIRECTION];
        features[FEAT_TCP_ACK_DELTA] = (float)(ack - ml_state->last_ack[!direction]);
        ml_state->last_ack[!direction] = ack;
    } else {
        features[FEAT_TCP_ACK_DELTA] = 0.0f;
    }
    
    /* Feature 12: TLS record type */
    features[FEAT_TLS_RECORD_TYPE] = (float)extract_tls_record_type(
        packet->payload, packet->payload_packet_len);
    
    /* Feature 13: TLS record length */
    if (packet->payload_packet_len >= 5) {
        features[FEAT_TLS_RECORD_LEN] = (float)((packet->payload[3] << 8) | packet->payload[4]);
    } else {
        features[FEAT_TLS_RECORD_LEN] = 0.0f;
    }
    
    /* Feature 14: TLS version */
    features[FEAT_TLS_VERSION] = (float)extract_tls_version(
        packet->payload, packet->payload_packet_len);
}

/* Normalize features using z-score normalization */
static void normalize_features(float *features, int num_packets,
                             const float *mean, const float *std) {
    int total_features = num_packets * NUM_FEATURES_PER_PACKET_FOR_TLS_ML;
    int i;
    for (i = 0; i < total_features; i++) {
        if (std[i] != 0.0f) {
            features[i] = (features[i] - mean[i]) / std[i];
        }
        /* Replace NaN/inf values with 0 */
        if (features[i] != features[i] || features[i] == 1.0f/0.0f || features[i] == -1.0f/0.0f) {
            features[i] = 0.0f;
        }
    }
}

/* Check if TLS ML inference has been performed */
int ndpi_tls_ml_inference_done(const struct ndpi_flow_struct *flow) {
    return flow->ml_state.inference_performed;
}

/* Initialize ML state for a flow */
void ndpi_init_tls_ml_state(struct ndpi_flow_struct *flow) {
    memset(&flow->ml_state, 0, sizeof(flow->ml_state));
    /* Initialize TCP sequence numbers */
    flow->ml_state.last_seq[0] = 0;
    flow->ml_state.last_seq[1] = 0;
    flow->ml_state.last_ack[0] = 0;
    flow->ml_state.last_ack[1] = 0;
}

/* Run inference on collected features */
#ifdef HAVE_ONNX_RUNTIME
static int run_inference(struct ndpi_detection_module_struct *ndpi_struct,
                         struct ndpi_flow_struct *flow) {
    struct ndpi_flow_ml_state *ml_state = &flow->ml_state;
    
    if (!tls_ml_runtime || !tls_ml_runtime->session) {
        return -1;
    }
    
    u_int64_t start_ts = ndpi_struct->current_ts;
    
    /* Build input tensor */
    int actual_packets = ml_state->num_packets_collected;
    int seq_len = ndpi_struct->cfg.tls_ml_packets_per_flow;
    
    float input_data[seq_len * NUM_FEATURES_PER_PACKET_FOR_TLS_ML];
    memcpy(input_data, ml_state->packet_features, 
           actual_packets * NUM_FEATURES_PER_PACKET_FOR_TLS_ML * sizeof(float));
    
    /* Zero-pad remaining packets */
    if (actual_packets < seq_len) {
        memset(input_data + actual_packets * NUM_FEATURES_PER_PACKET_FOR_TLS_ML, 0,
               (seq_len - actual_packets) * NUM_FEATURES_PER_PACKET_FOR_TLS_ML * sizeof(float));
    }
    
    /* TODO: Apply normalization with loaded scaler */
    if (tls_ml_runtime->scaler_mean[0] != 0.0f) {
        normalize_features(input_data, seq_len,
                          tls_ml_runtime->scaler_mean, tls_ml_runtime->scaler_std);
    }
    
    /* Create input tensor */
    int64_t input_dims[3] = {1, seq_len, NUM_FEATURES_PER_PACKET_FOR_TLS_ML};
    OrtValue *input_tensor = NULL;
    OrtStatus *status;
    
    status = OrtCreateTensor(tls_ml_runtime->allocator, input_data, sizeof(input_data),
                             input_dims, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_tensor);
    if (status) {
        OrtReleaseStatus(status);
        return -1;
    }
    
    /* Run inference */
    const char* input_names[] = {"input"};
    const char* output_names[] = {"output"};
    
    OrtValue *output_tensor = NULL;
    status = OrtRun(tls_ml_runtime->session, NULL, input_names, &input_tensor, 1, output_names, 1, &output_tensor);
    
    if (status) {
        OrtReleaseStatus(status);
        OrtReleaseValue(input_tensor);
        return -1;
    }
    
    /* Extract results */
    float *output_data = NULL;
    size_t output_dims[3];
    status = OrtGetTensorData(output_tensor, (const float**)&output_data, output_dims, 3);
    
    if (status) {
        OrtReleaseStatus(status);
        OrtReleaseValue(output_tensor);
        OrtReleaseValue(input_tensor);
        return -1;
    }
    
    /* Find predicted class (argmax) and confidence (softmax max) */
    int max_idx = 0;
    float max_val = output_data[0];
    for (int i = 1; i < NDPI_TLS_ML_NUM_PROTOCOLS; i++) {
        if (output_data[i] > max_val) {
            max_val = output_data[i];
            max_idx = i;
        }
    }
    
    /* Apply softmax to get confidence */
    float sum = 0.0f;
    for (int i = 0; i < NDPI_TLS_ML_NUM_PROTOCOLS; i++) {
        sum += expf(output_data[i]);
    }
    ml_state->confidence_score = expf(max_val) / sum;
    ml_state->predicted_protocol_id = ndpi_tls_ml_protocols[max_idx];
    ml_state->inference_time_us = (u_int32_t)((ndpi_struct->current_ts - start_ts) * 1000);
    ml_state->inference_performed = 1;
    
    OrtReleaseValue(output_tensor);
    OrtReleaseValue(input_tensor);
    
    return 0;
}
#endif

/* Process packet for TLS ML feature extraction - called from tls.c */
void ndpi_process_tls_ml_packet(struct ndpi_detection_module_struct *ndpi_struct,
                                struct ndpi_flow_struct *flow) {
    struct ndpi_flow_ml_state *ml_state = &flow->ml_state;
    
    if (!ndpi_struct->cfg.tls_ml_enabled || ml_state->inference_performed) {
        return;
    }
    
    /* Check if this is a TLS handshake packet */
    bool is_tls_handshake = false;
    u_int8_t record_type = extract_tls_record_type(
        ndpi_struct->packet.payload, ndpi_struct->packet.payload_packet_len);
    
    if (record_type >= 5 && record_type <= 10) { /* ClientHello to ServerFinished */
        is_tls_handshake = true;
    }
    
    /* Track handshake completion */
    if (!ml_state->handshake_complete && is_tls_handshake) {
        if (record_type == 5) { /* ClientHello */
            ml_state->handshake_complete = 1;
            if (ml_state->first_packet_time_ms == 0) {
                ml_state->first_packet_time_ms = ndpi_struct->current_ts;
            }
        }
    }
    
    /* Only collect features after handshake is seen or for TLS traffic */
    if (ml_state->handshake_complete || flow->l4_proto == IPPROTO_TCP) {
        if (ml_state->packet_features == NULL &&
            ml_state->max_packets > 0 &&
            ml_state->num_packets_collected == 0) {
            ml_state->packet_features = (float *)ndpi_calloc(
                ml_state->max_packets * NUM_FEATURES_PER_PACKET_FOR_TLS_ML,
                sizeof(float));
        }
        if (ml_state->packet_features != NULL &&
            ml_state->num_packets_collected < MAX_PACKETS_PER_FLOW_FOR_TLS_ML) {
            float *feature_slot = ml_state->packet_features +
                ml_state->num_packets_collected * NUM_FEATURES_PER_PACKET_FOR_TLS_ML;
            extract_packet_features(ndpi_struct, flow, feature_slot);
            ml_state->num_packets_collected++;
            ml_state->last_packet_time_ms = ndpi_struct->current_ts;
        }
    }
    
    /* Run inference when we have enough packets */
#ifdef HAVE_ONNX_RUNTIME
    if (ml_state->num_packets_collected >= ndpi_struct->cfg.tls_ml_packets_per_flow &&
        !ml_state->inference_performed) {
        run_inference(ndpi_struct, flow);
    }
#endif
}

/* Get prediction results */
int ndpi_get_tls_ml_prediction(struct ndpi_flow_struct *flow,
                                u_int16_t *protocol_id,
                                float *confidence,
                                u_int32_t *inference_time_us) {
    if (!flow->ml_state.inference_performed) {
        return 0;
    }
    
    if (protocol_id) {
        *protocol_id = flow->ml_state.predicted_protocol_id;
    }
    if (confidence) {
        *confidence = flow->ml_state.confidence_score;
    }
    if (inference_time_us) {
        *inference_time_us = flow->ml_state.inference_time_us;
    }
    
    return 1;
}