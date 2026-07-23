#!/bin/bash
#
# run_tls_ml.sh - Запуск nDPI с ML-моделью для классификации TLS-протоколов
#
# Использование:
#   ./run_tls_ml.sh <pcap_file> [packets_per_flow]
#
# Пример:
#   ./run_tls_ml.sh tests/pcap/netflix.pcap 10
#

MODEL_PATH="/home/nikita/tls_trained_models/cnn_model/exported/tls_classifier.onnx"
CLASSES_PATH="/home/nikita/tls_trained_models/cnn_model/exported/tls_classifier_classes.txt"
NDPI_READER="/home/nikita/nDPI/example/ndpiReader"

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: ML model not found at $MODEL_PATH"
    exit 1
fi

if [ ! -f "$CLASSES_PATH" ]; then
    echo "ERROR: Classes file not found at $CLASSES_PATH"
    exit 1
fi

if [ ! -f "$NDPI_READER" ]; then
    echo "ERROR: ndpiReader not found at $NDPI_READER"
    echo "Please build nDPI first: cd /home/nikita/nDPI && ./configure --enable-tls-ml && make"
    exit 1
fi

PCAP_FILE="${1:-tests/pcap/netflix.pcap}"
PACKETS_PER_FLOW="${2:-10}"

if [ ! -f "$PCAP_FILE" ]; then
    echo "ERROR: pcap file not found: $PCAP_FILE"
    exit 1
fi

echo "=== nDPI TLS ML Detection ==="
echo "Model: $MODEL_PATH"
echo "Classes: $CLASSES_PATH"
echo "Packets per flow: $PACKETS_PER_FLOW"
echo "PCAP: $PCAP_FILE"
echo "============================="
echo ""

# Запуск ndpiReader с ML-моделью
# -v 2 = verbose level 2 (показать детали потоков)
# --tls-ml-model = путь к ONNX-модели
# --tls-ml-classes = путь к файлу классов
# --tls-ml-packets = количество пакетов для сбора перед инференсом
"$NDPI_READER" \
    --tls-ml-model "$MODEL_PATH" \
    --tls-ml-classes "$CLASSES_PATH" \
    --tls-ml-packets "$PACKETS_PER_FLOW" \
    -v 2 \
    -i "$PCAP_FILE" \
    2>/dev/null

echo ""
echo "=== Detection Complete ==="