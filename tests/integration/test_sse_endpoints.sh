#!/bin/bash

# SSE Endpoint Curl Tests
# Tests the SSE endpoints directly using curl

set -e

BASE_URL="http://localhost:8001"
API_PATH="/api/training"
TIMEOUT=30

echo "=============================================================================="
echo "SSE ENDPOINT CURL TESTS"
echo "=============================================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test 1: Basic SSE Connection
echo "Test 1: Basic SSE Connection"
echo "Testing: curl -N ${BASE_URL}${API_PATH}/stream/test_curl_123"
echo ""
echo "Expected:"
echo "  - Connected event"
echo "  - Keepalive comments every 30s"
echo "  - Connection remains open"
echo ""
echo "To test:"
echo "  1. Run: timeout 35 curl -N ${BASE_URL}${API_PATH}/stream/test_curl_123"
echo "  2. Observe keepalive comments after initial connected event"
echo "  3. Connection should remain open for 35 seconds"
echo ""
echo -e "${YELLOW}Manual test - run in terminal${NC}"
echo ""

# Test 2: Stats Endpoint
echo "=============================================================================="
echo "Test 2: Stats Endpoint"
echo "Testing: curl ${BASE_URL}${API_PATH}/stream/stats"
echo ""
echo "Expected JSON response with:"
echo "  - total_queues"
echo "  - active_experiments"
echo "  - total_clients"
echo "  - experiments object"
echo ""

if command -v jq &> /dev/null; then
    echo "Running stats test..."
    RESPONSE=$(curl -s ${BASE_URL}${API_PATH}/stream/stats)

    if [ -n "$RESPONSE" ]; then
        echo -e "${GREEN}✅ Stats endpoint accessible${NC}"
        echo "$RESPONSE" | jq '.'

        # Check fields
        TOTAL_QUEUES=$(echo "$RESPONSE" | jq '.total_queues')
        TOTAL_CLIENTS=$(echo "$RESPONSE" | jq '.total_clients')

        echo ""
        echo "Summary:"
        echo "  - Total queues: $TOTAL_QUEUES"
        echo "  - Total clients: $TOTAL_CLIENTS"
    else
        echo -e "${RED}❌ Stats endpoint failed${NC}"
    fi
else
    echo -e "${YELLOW}jq not installed - showing raw response:${NC}"
    curl -s ${BASE_URL}${API_PATH}/stream/stats
fi

echo ""
echo "=============================================================================="
echo "Test 3: Multiple Concurrent Connections"
echo "Testing multiple concurrent SSE connections"
echo ""
echo "To test:"
echo "  Terminal 1: curl -N ${BASE_URL}${API_PATH}/stream/test_multi_123"
echo "  Terminal 2: curl -N ${BASE_URL}${API_PATH}/stream/test_multi_123"
echo "  Terminal 3: curl -N ${BASE_URL}${API_PATH}/stream/test_multi_123"
echo "  Terminal 4: curl ${BASE_URL}${API_PATH}/stream/stats | jq '.experiments.test_multi_123.clients'"
echo ""
echo "Expected: clients field should show 3"
echo ""
echo -e "${YELLOW}Manual test - run in separate terminals${NC}"
echo ""

# Test 4: Check backend is running
echo "=============================================================================="
echo "Test 4: Backend Availability Check"
echo ""

if timeout 5 curl -s ${BASE_URL}${API_PATH}/stream/stats > /dev/null 2>&1; then
    echo -e "${GREEN}✅ Backend SSE endpoints are running${NC}"
else
    echo -e "${RED}❌ Backend SSE endpoints not responding${NC}"
    echo ""
    echo "To start backend:"
    echo "  cd federated_pneumonia_detection"
    echo "  uvicorn src.api.main:app --reload --host 127.0.0.1 --port 8001"
fi

echo ""
echo "=============================================================================="
echo "MANUAL TEST INSTRUCTIONS"
echo "=============================================================================="
echo ""
echo "Test: Direct Event Streaming"
echo "1. Terminal 1 - Start listening to SSE stream:"
echo "   timeout 30 curl -N http://localhost:8001/api/training/stream/test_integration_123"
echo ""
echo "2. Terminal 2 - Run Python event publisher:"
echo "   python tests/integration/test_backend_sse_endpoints.py"
echo ""
echo "Expected in Terminal 1:"
echo "  event: connected"
echo "  data: {\"experiment_id\": \"test_integration_123\", \"timestamp\": 1737041600.0}"
echo ""
echo "  event: epoch_start"
echo "  data: {\"epoch\": 1, \"total_epochs\": 10}"
echo ""
echo "  event: batch_metrics"
echo "  data: {\"step\": 1, \"loss\": 0.563, \"accuracy\": 0.812}"
echo ""
echo "  event: epoch_end"
echo "  data: {\"epoch\": 1, \"phase\": \"train\", \"metrics\": {\"loss\": 0.450, \"accuracy\": 0.85}}"
echo ""
echo "=============================================================================="
echo ""
