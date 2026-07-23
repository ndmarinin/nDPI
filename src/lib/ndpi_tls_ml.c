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

#include <string.h>
#include <stdio.h>
#include <stdlib.h>

#ifdef HAVE_ONNX_RUNTIME
#include <onnxruntime_c_api.h>
#include <math.h>
#endif

/* Feature indices for TLS ML packet features (19 features as per training) */
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
#define FEAT_TLS_HANDSHAKE_TYPE 15  /* Additional: 0=client_hello, 1=server_hello, etc */
#define FEAT_TLS_CIPHER_LEN     16  /* Additional: total cipher suite length */
#define FEAT_TLS_EXT_LEN        17  /* Additional: total extensions length */
#define FEAT_TLS_CERT_LEN       18  /* Additional: certificate length */

#define TLS_ML_MAX_CLASSES      16

/* ONNX Runtime state (conditionally compiled) */
#ifdef HAVE_ONNX_RUNTIME
struct ndpi_tls_ml_runtime {
    const OrtApi *api;
    OrtEnv *ort_env;
    OrtSession *session;
    OrtAllocator *allocator;
    int model_type;
    char *input_name;
    char *output_name;
    char class_names[TLS_ML_MAX_CLASSES][32];
    u_int16_t class_proto_ids[TLS_ML_MAX_CLASSES];
    int num_classes;
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

#ifdef HAVE_ONNX_RUNTIME
static u_int16_t class_name_to_proto_id(const char *name);
#endif

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

    /* Features 15-18: additional TLS features (set to 0 for now, can be extended) */
    features[FEAT_TLS_HANDSHAKE_TYPE] = 0.0f;
    features[FEAT_TLS_CIPHER_LEN] = 0.0f;
    features[FEAT_TLS_EXT_LEN] = 0.0f;
    features[FEAT_TLS_CERT_LEN] = 0.0f;
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
    /* max_packets will be set on first packet based on cfg.tls_ml_packets_per_flow */
    flow->ml_state.max_packets = 0;
    flow->ml_state.packet_features = NULL;
    flow->ml_state.num_packets_collected = 0;
    flow->ml_state.inference_performed = 0;
}

#ifdef HAVE_ONNX_RUNTIME

/* Map class name to nDPI protocol ID */
static u_int16_t class_name_to_proto_id(const char *name) {
    if (!name) return NDPI_PROTOCOL_TLS;

    if (strcmp(name, "DoT") == 0 || strcmp(name, "DOT") == 0)
        return NDPI_PROTOCOL_DOH_DOT;
    if (strcmp(name, "HTTPS") == 0)
        return NDPI_PROTOCOL_TLS;
    if (strcmp(name, "IMAPS") == 0)
        return NDPI_PROTOCOL_MAIL_IMAPS;
    if (strcmp(name, "LDAPS") == 0)
        return NDPI_PROTOCOL_LDAP;
    if (strcmp(name, "MQTTS") == 0)
        return NDPI_PROTOCOL_MQTT;

    /* Default fallback */
    return NDPI_PROTOCOL_TLS;
}

/* Load ML model for TLS protocol detection */
int ndpi_load_tls_ml_model(struct ndpi_detection_module_struct *ndpi_str,
                           const char *model_path, int model_type) {
    const OrtApiBase *api_base;
    OrtStatus *status;
    OrtSessionOptions *session_options = NULL;

    if (!ndpi_str || !model_path) {
        return -1;
    }

    /* If already loaded, free previous */
    if (tls_ml_runtime) {
        if (tls_ml_runtime->session) {
            tls_ml_runtime->api->ReleaseSession(tls_ml_runtime->session);
        }
        if (tls_ml_runtime->ort_env) {
            tls_ml_runtime->api->ReleaseEnv(tls_ml_runtime->ort_env);
        }
        if (tls_ml_runtime->input_name) {
            ndpi_free(tls_ml_runtime->input_name);
        }
        if (tls_ml_runtime->output_name) {
            ndpi_free(tls_ml_runtime->output_name);
        }
        ndpi_free(tls_ml_runtime);
        tls_ml_runtime = NULL;
    }

    /* Allocate runtime structure */
    tls_ml_runtime = (struct ndpi_tls_ml_runtime *)ndpi_calloc(1, sizeof(struct ndpi_tls_ml_runtime));
    if (!tls_ml_runtime) {
        fprintf(stderr, "TLS ML: failed to allocate runtime\n");
        return -1;
    }

    /* Get ONNX Runtime API */
    api_base = OrtGetApiBase();
    if (!api_base) {
        fprintf(stderr, "TLS ML: OrtGetApiBase() returned NULL\n");
        ndpi_free(tls_ml_runtime);
        tls_ml_runtime = NULL;
        return -1;
    }

    tls_ml_runtime->api = api_base->GetApi(ORT_API_VERSION);
    if (!tls_ml_runtime->api) {
        fprintf(stderr, "TLS ML: GetApi(%d) returned NULL\n", ORT_API_VERSION);
        ndpi_free(tls_ml_runtime);
        tls_ml_runtime = NULL;
        return -1;
    }

    /* Create environment */
    status = tls_ml_runtime->api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "nDPI_TLS_ML", &tls_ml_runtime->ort_env);
    if (status) {
        fprintf(stderr, "TLS ML: CreateEnv failed: %s\n", tls_ml_runtime->api->GetErrorMessage(status));
        tls_ml_runtime->api->ReleaseStatus(status);
        ndpi_free(tls_ml_runtime);
        tls_ml_runtime = NULL;
        return -1;
    }

    /* Create session options */
    status = tls_ml_runtime->api->CreateSessionOptions(&session_options);
    if (status) {
        fprintf(stderr, "TLS ML: CreateSessionOptions failed: %s\n", tls_ml_runtime->api->GetErrorMessage(status));
        tls_ml_runtime->api->ReleaseStatus(status);
        tls_ml_runtime->api->ReleaseEnv(tls_ml_runtime->ort_env);
        ndpi_free(tls_ml_runtime);
        tls_ml_runtime = NULL;
        return -1;
    }

    /* Set session options: single thread, disable optimizations for speed */
    tls_ml_runtime->api->SetIntraOpNumThreads(session_options, 1);
    tls_ml_runtime->api->SetSessionGraphOptimizationLevel(session_options, ORT_ENABLE_BASIC);

    /* Create session from model file */
    status = tls_ml_runtime->api->CreateSession(tls_ml_runtime->ort_env, model_path, session_options, &tls_ml_runtime->session);
    tls_ml_runtime->api->ReleaseSessionOptions(session_options);

    if (status) {
        fprintf(stderr, "TLS ML: CreateSession failed for %s: %s\n", model_path, tls_ml_runtime->api->GetErrorMessage(status));
        tls_ml_runtime->api->ReleaseStatus(status);
        tls_ml_runtime->api->ReleaseEnv(tls_ml_runtime->ort_env);
        ndpi_free(tls_ml_runtime);
        tls_ml_runtime = NULL;
        return -1;
    }

    /* Get default allocator */
    status = tls_ml_runtime->api->GetAllocatorWithDefaultOptions(&tls_ml_runtime->allocator);
    if (status) {
        fprintf(stderr, "TLS ML: GetAllocatorWithDefaultOptions failed: %s\n", tls_ml_runtime->api->GetErrorMessage(status));
        tls_ml_runtime->api->ReleaseStatus(status);
        tls_ml_runtime->api->ReleaseSession(tls_ml_runtime->session);
        tls_ml_runtime->api->ReleaseEnv(tls_ml_runtime->ort_env);
        ndpi_free(tls_ml_runtime);
        tls_ml_runtime = NULL;
        return -1;
    }

    /* Get input name (index 0) */
    char *input_name = NULL;
    status = tls_ml_runtime->api->SessionGetInputName(tls_ml_runtime->session, 0, tls_ml_runtime->allocator, &input_name);
    if (status) {
        fprintf(stderr, "TLS ML: SessionGetInputName failed: %s\n", tls_ml_runtime->api->GetErrorMessage(status));
        tls_ml_runtime->api->ReleaseStatus(status);
    } else {
        tls_ml_runtime->input_name = ndpi_strdup(input_name);
        /* Free the allocator-owned string using allocator's Free */
        tls_ml_runtime->allocator->Free(tls_ml_runtime->allocator, input_name);
    }

    /* Get output name (index 0) */
    char *output_name = NULL;
    status = tls_ml_runtime->api->SessionGetOutputName(tls_ml_runtime->session, 0, tls_ml_runtime->allocator, &output_name);
    if (status) {
        fprintf(stderr, "TLS ML: SessionGetOutputName failed: %s\n", tls_ml_runtime->api->GetErrorMessage(status));
        tls_ml_runtime->api->ReleaseStatus(status);
    } else {
        tls_ml_runtime->output_name = ndpi_strdup(output_name);
        tls_ml_runtime->allocator->Free(tls_ml_runtime->allocator, output_name);
    }

    tls_ml_runtime->model_type = model_type;
    tls_ml_runtime->num_classes = 0;

    /* Enable TLS ML in config */
    ndpi_str->cfg.tls_ml_enabled = 1;
    if (ndpi_str->cfg.tls_ml_packets_per_flow <= 0) {
        ndpi_str->cfg.tls_ml_packets_per_flow = 10;
    }
    if (ndpi_str->cfg.tls_ml_confidence_threshold <= 0) {
        ndpi_str->cfg.tls_ml_confidence_threshold = 50; /* stored as int * 100 for threshold 0.50 */
    }

    printf("TLS ML: model loaded successfully from %s (type=%d, input=%s, output=%s)\n",
           model_path, model_type,
           tls_ml_runtime->input_name ? tls_ml_runtime->input_name : "?",
           tls_ml_runtime->output_name ? tls_ml_runtime->output_name : "?");

    return 0;
}

/* Load class labels for TLS ML model */
int ndpi_load_tls_ml_classes(struct ndpi_detection_module_struct *ndpi_str,
                             const char *classes_path) {
    FILE *fd;
    char line[256];
    int count = 0;

    if (!ndpi_str || !classes_path || !tls_ml_runtime) {
        return -1;
    }

    fd = fopen(classes_path, "r");
    if (!fd) {
        fprintf(stderr, "TLS ML: cannot open classes file %s\n", classes_path);
        return -1;
    }

    while (fgets(line, sizeof(line), fd) && count < TLS_ML_MAX_CLASSES) {
        /* Strip trailing newline/whitespace */
        char *p = line;
        size_t len = strlen(p);
        while (len > 0 && (p[len-1] == '\n' || p[len-1] == '\r' || p[len-1] == ' ' || p[len-1] == '\t')) {
            p[--len] = '\0';
        }
        /* Skip empty lines */
        if (len == 0) continue;

        /* Copy class name */
        strncpy(tls_ml_runtime->class_names[count], p, sizeof(tls_ml_runtime->class_names[count]) - 1);
        tls_ml_runtime->class_names[count][sizeof(tls_ml_runtime->class_names[count]) - 1] = '\0';

        /* Map to protocol ID */
        tls_ml_runtime->class_proto_ids[count] = class_name_to_proto_id(p);

        printf("TLS ML: class[%d] = %s -> proto_id=%u\n", count, p, tls_ml_runtime->class_proto_ids[count]);

        count++;
    }

    fclose(fd);
    tls_ml_runtime->num_classes = count;

    printf("TLS ML: loaded %d classes from %s\n", count, classes_path);

    return 0;
}

/* Configure TLS ML detection parameters */
int ndpi_configure_tls_ml(struct ndpi_detection_module_struct *ndpi_str,
                          int packets_per_flow,
                          const char *scaler_path) {
    if (!ndpi_str) {
        return -1;
    }

    if (packets_per_flow > 0 && packets_per_flow <= MAX_PACKETS_PER_FLOW_FOR_TLS_ML) {
        ndpi_str->cfg.tls_ml_packets_per_flow = packets_per_flow;
    } else if (packets_per_flow <= 0) {
        ndpi_str->cfg.tls_ml_packets_per_flow = 10;
    }

    /* scaler_path is currently unused (normalization not implemented) */
    (void)scaler_path;

    return 0;
}

/* Run inference on collected features */
static int run_inference(struct ndpi_detection_module_struct *ndpi_struct,
                         struct ndpi_flow_struct *flow) {
    struct ndpi_flow_ml_state *ml_state = &flow->ml_state;

    if (!tls_ml_runtime || !tls_ml_runtime->session || !tls_ml_runtime->api) {
        return -1;
    }

    const OrtApi *api = tls_ml_runtime->api;
    u_int64_t start_ts = ndpi_struct->current_ts;

    /* Build input tensor */
    int actual_packets = ml_state->num_packets_collected;
    /* Model expects fixed seq_len=10 */
    int seq_len = 10;
    int packets_to_collect = ndpi_struct->cfg.tls_ml_packets_per_flow;
    if (packets_to_collect <= 0 || packets_to_collect > seq_len) packets_to_collect = seq_len;

    size_t input_data_count = (size_t)seq_len * NUM_FEATURES_PER_PACKET_FOR_TLS_ML;
    float *input_data = (float *)ndpi_calloc(input_data_count, sizeof(float));
    if (!input_data) {
        return -1;
    }

    /* Copy collected features */
    if (actual_packets > 0 && ml_state->packet_features) {
        size_t copy_count = (size_t)actual_packets * NUM_FEATURES_PER_PACKET_FOR_TLS_ML;
        if (copy_count > input_data_count) copy_count = input_data_count;
        memcpy(input_data, ml_state->packet_features, copy_count * sizeof(float));
    }
    /* Remaining is zero-padded (already zeroed by calloc) */

    /* Create input tensor */
    int64_t input_dims[3] = {1, seq_len, NUM_FEATURES_PER_PACKET_FOR_TLS_ML};
    OrtMemoryInfo *memory_info = NULL;
    OrtStatus *status;

    status = api->CreateCpuMemoryInfo(OrtDeviceAllocator, OrtMemTypeDefault, &memory_info);
    if (status) {
        fprintf(stderr, "TLS ML: CreateCpuMemoryInfo failed: %s\n", api->GetErrorMessage(status));
        api->ReleaseStatus(status);
        ndpi_free(input_data);
        return -1;
    }

    OrtValue *input_tensor = NULL;
    status = api->CreateTensorWithDataAsOrtValue(
        memory_info,
        input_data,
        input_data_count * sizeof(float),
        input_dims,
        3,
        ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
        &input_tensor);

    api->ReleaseMemoryInfo(memory_info);

    if (status) {
        fprintf(stderr, "TLS ML: CreateTensorWithDataAsOrtValue failed: %s\n", api->GetErrorMessage(status));
        api->ReleaseStatus(status);
        ndpi_free(input_data);
        return -1;
    }

    /* Run inference */
    const char *input_names[] = {tls_ml_runtime->input_name ? tls_ml_runtime->input_name : "input"};
    const char *output_names[] = {tls_ml_runtime->output_name ? tls_ml_runtime->output_name : "output"};

    OrtValue *output_tensor = NULL;
    status = api->Run(tls_ml_runtime->session, NULL, input_names, (const OrtValue *const *)&input_tensor, 1,
                      output_names, 1, &output_tensor);

    if (status) {
        fprintf(stderr, "TLS ML: Run failed: %s\n", api->GetErrorMessage(status));
        api->ReleaseStatus(status);
        api->ReleaseValue(input_tensor);
        ndpi_free(input_data);
        return -1;
    }

    /* Extract results */
    float *output_data = NULL;
    status = api->GetTensorMutableData(output_tensor, (void **)&output_data);

    if (status) {
        fprintf(stderr, "TLS ML: GetTensorMutableData failed: %s\n", api->GetErrorMessage(status));
        api->ReleaseStatus(status);
        api->ReleaseValue(output_tensor);
        api->ReleaseValue(input_tensor);
        ndpi_free(input_data);
        return -1;
    }

    /* Determine number of classes */
    int num_classes = tls_ml_runtime->num_classes;
    if (num_classes <= 0) {
        num_classes = 5; /* default */
    }

    /* Find predicted class (argmax) and confidence (softmax max) */
    int max_idx = 0;
    float max_val = output_data[0];
    int i;
    for (i = 1; i < num_classes; i++) {
        if (output_data[i] > max_val) {
            max_val = output_data[i];
            max_idx = i;
        }
    }

    /* Numerically stable softmax: subtract max before exp to avoid overflow */
    float sum = 0.0f;
    for (i = 0; i < num_classes; i++) {
        sum += expf(output_data[i] - max_val);
    }
    if (sum > 0.0f && isfinite(sum)) {
        ml_state->confidence_score = 1.0f / sum;
    } else {
        ml_state->confidence_score = 0.0f;
    }

    /* Map class index to protocol ID */
    if (max_idx >= 0 && max_idx < num_classes) {
        ml_state->predicted_protocol_id = tls_ml_runtime->class_proto_ids[max_idx];
    } else {
        ml_state->predicted_protocol_id = NDPI_PROTOCOL_TLS;
    }

    ml_state->inference_time_us = (u_int32_t)((ndpi_struct->current_ts - start_ts) * 1000);
    ml_state->inference_performed = 1;

    /* Log all class probabilities */
    printf("TLS ML: prediction - ");
    for (i = 0; i < num_classes; i++) {
        float prob = expf(output_data[i] - max_val) / sum;
        const char *name = (i < tls_ml_runtime->num_classes) ? tls_ml_runtime->class_names[i] : "?";
        printf("%s=%.2f%% ", name, prob * 100.0f);
    }
    printf("(best=%s proto=%u conf=%.4f)\n",
           (max_idx < tls_ml_runtime->num_classes) ? tls_ml_runtime->class_names[max_idx] : "?",
           ml_state->predicted_protocol_id, ml_state->confidence_score);

    api->ReleaseValue(output_tensor);
    api->ReleaseValue(input_tensor);
    ndpi_free(input_data);

    return 0;
}
#endif /* HAVE_ONNX_RUNTIME */

/* Process packet for TLS ML feature extraction - called from tls.c */
void ndpi_process_tls_ml_packet(struct ndpi_detection_module_struct *ndpi_struct,
                                struct ndpi_flow_struct *flow) {
    struct ndpi_flow_ml_state *ml_state = &flow->ml_state;

    if (!ndpi_struct->cfg.tls_ml_enabled || ml_state->inference_performed) {
        return;
    }
    
    /* Debug: log first call */
    if (ml_state->num_packets_collected == 0 && ml_state->max_packets == 0) {
        printf("TLS ML: process_packet called - tls_ml_enabled=%d packets_per_flow=%d\n",
               ndpi_struct->cfg.tls_ml_enabled, ndpi_struct->cfg.tls_ml_packets_per_flow);
    }

    /* Initialize max_packets from config if not set */
    if (ml_state->max_packets == 0) {
        /* Model expects fixed seq_len=10 */
        ml_state->max_packets = 10;
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
            printf("TLS ML: allocated features buffer max=%u handshake=%d l4=%d\n",
                   ml_state->max_packets, ml_state->handshake_complete, flow->l4_proto);
        }
        if (ml_state->packet_features != NULL &&
            ml_state->num_packets_collected < ml_state->max_packets) {
            float *feature_slot = ml_state->packet_features +
                ml_state->num_packets_collected * NUM_FEATURES_PER_PACKET_FOR_TLS_ML;
            extract_packet_features(ndpi_struct, flow, feature_slot);
            ml_state->num_packets_collected++;
            ml_state->last_packet_time_ms = ndpi_struct->current_ts;
            if (ml_state->num_packets_collected == ml_state->max_packets) {
                printf("TLS ML: collected %u packets, ready for inference\n", ml_state->num_packets_collected);
            }
        }
    }

    /* Run inference when we have enough packets */
#ifdef HAVE_ONNX_RUNTIME
    if (tls_ml_runtime && tls_ml_runtime->session &&
        ml_state->num_packets_collected >= ndpi_struct->cfg.tls_ml_packets_per_flow &&
        !ml_state->inference_performed) {
        printf("TLS ML: running inference - collected=%u packets_per_flow=%d max=%u\n",
               ml_state->num_packets_collected, ndpi_struct->cfg.tls_ml_packets_per_flow, ml_state->max_packets);
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

/* Free TLS ML state resources for a flow (called from ndpi_free_flow_data) */
void ndpi_free_tls_ml_state(struct ndpi_flow_struct *flow) {
    if (flow && flow->ml_state.packet_features) {
        ndpi_free(flow->ml_state.packet_features);
        flow->ml_state.packet_features = NULL;
    }
}