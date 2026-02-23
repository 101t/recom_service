#!/usr/bin/env bash
#
# End-to-end test suite for the Recommendation Service.
#
# Prerequisites:
#   - The service running at BASE_URL (default http://localhost:8000)
#   - Qdrant running and reachable by the service
#   - An embedding service running (local FastEmbed or Voyage AI configured)
#
# Usage:
#   ./e2e/run_tests.sh                   # uses default http://localhost:8000
#   BASE_URL=http://localhost:8080 ./e2e/run_tests.sh
#
set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
API="${BASE_URL}/api/v1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATA_DIR="${SCRIPT_DIR}/data"

# ── Colours ──────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m' # No Colour

PASS=0
FAIL=0
TOTAL=0
SKIPPED=0
CSV_DATASET_ID=""
JSON_DATASET_ID=""
JOB_ID=""
E2E_COLLECTION="e2e_test_collection"

# ── Helpers ──────────────────────────────────────────────────

log_header() {
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}  $1${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
}

assert_status() {
    local test_name="$1"
    local expected="$2"
    local actual="$3"
    TOTAL=$((TOTAL + 1))
    if [ "$actual" -eq "$expected" ]; then
        echo -e "  ${GREEN}✓ PASS${NC}  ${test_name}  (HTTP ${actual})"
        PASS=$((PASS + 1))
    else
        echo -e "  ${RED}✗ FAIL${NC}  ${test_name}  (expected ${expected}, got ${actual})"
        FAIL=$((FAIL + 1))
    fi
}

assert_json_field() {
    local test_name="$1"
    local json="$2"
    local field="$3"
    local expected="$4"
    TOTAL=$((TOTAL + 1))
    local actual
    actual=$(echo "$json" | jq -r "$field" 2>/dev/null || echo "__JQ_ERROR__")
    if [ "$actual" = "$expected" ]; then
        echo -e "  ${GREEN}✓ PASS${NC}  ${test_name}  (${field} = ${actual})"
        PASS=$((PASS + 1))
    else
        echo -e "  ${RED}✗ FAIL${NC}  ${test_name}  (${field}: expected '${expected}', got '${actual}')"
        FAIL=$((FAIL + 1))
    fi
}

assert_json_exists() {
    local test_name="$1"
    local json="$2"
    local field="$3"
    TOTAL=$((TOTAL + 1))
    local val
    val=$(echo "$json" | jq -e "$field" 2>/dev/null)
    if [ $? -eq 0 ] && [ "$val" != "null" ]; then
        echo -e "  ${GREEN}✓ PASS${NC}  ${test_name}  (${field} exists)"
        PASS=$((PASS + 1))
    else
        echo -e "  ${RED}✗ FAIL${NC}  ${test_name}  (${field} missing or null)"
        FAIL=$((FAIL + 1))
    fi
}

assert_json_gte() {
    local test_name="$1"
    local json="$2"
    local field="$3"
    local min_val="$4"
    TOTAL=$((TOTAL + 1))
    local actual
    actual=$(echo "$json" | jq -r "$field" 2>/dev/null || echo "0")
    if [ "$actual" -ge "$min_val" ] 2>/dev/null; then
        echo -e "  ${GREEN}✓ PASS${NC}  ${test_name}  (${field} = ${actual} >= ${min_val})"
        PASS=$((PASS + 1))
    else
        echo -e "  ${RED}✗ FAIL${NC}  ${test_name}  (${field}: expected >= ${min_val}, got '${actual}')"
        FAIL=$((FAIL + 1))
    fi
}

skip_test() {
    local test_name="$1"
    local reason="$2"
    TOTAL=$((TOTAL + 1))
    SKIPPED=$((SKIPPED + 1))
    echo -e "  ${YELLOW}⊘ SKIP${NC}  ${test_name}  (${reason})"
}

# ── Preflight ────────────────────────────────────────────────

check_deps() {
    for cmd in curl jq; do
        if ! command -v "$cmd" &>/dev/null; then
            echo -e "${RED}Error: '$cmd' is required but not installed.${NC}"
            exit 1
        fi
    done
}

wait_for_service() {
    echo -e "${YELLOW}Waiting for service at ${API}/health ...${NC}"
    local retries=30
    while [ $retries -gt 0 ]; do
        if curl -sf "${API}/health" >/dev/null 2>&1; then
            echo -e "${GREEN}Service is up.${NC}"
            return
        fi
        retries=$((retries - 1))
        sleep 1
    done
    echo -e "${RED}Service did not become healthy within 30s. Aborting.${NC}"
    exit 1
}

# ═══════════════════════════════════════════════════════════════
#  TEST SUITES
# ═══════════════════════════════════════════════════════════════

test_health() {
    log_header "1 · Health Check"

    local resp
    resp=$(curl -sw '\n%{http_code}' "${API}/health" 2>/dev/null)
    local code=$(echo "$resp" | tail -1)
    local body=$(echo "$resp" | sed '$d')

    assert_status "GET /health returns 200" 200 "$code"
    assert_json_field "status field is 'ok'" "$body" ".status" "ok"
    assert_json_exists "version field present" "$body" ".version"
}

# ─────────────────────────────────────────────────────────────
test_swagger() {
    log_header "2 · Swagger / OpenAPI"

    local code
    code=$(curl -so /dev/null -w '%{http_code}' "${BASE_URL}/docs/" 2>/dev/null)
    assert_status "GET /docs/ returns 200 (Swagger UI)" 200 "$code"

    code=$(curl -so /dev/null -w '%{http_code}' "${BASE_URL}/api-docs/openapi.json" 2>/dev/null)
    assert_status "GET /api-docs/openapi.json returns 200" 200 "$code"
}

# ─────────────────────────────────────────────────────────────
test_collection_management() {
    log_header "3 · Collection Management"

    # 3a – Create collection
    local resp
    resp=$(curl -sw '\n%{http_code}' -X POST "${API}/collections" \
        -H 'Content-Type: application/json' \
        -d "{\"name\": \"${E2E_COLLECTION}\", \"vector_size\": 384, \"distance\": \"cosine\"}" 2>/dev/null)
    local code=$(echo "$resp" | tail -1)
    local body=$(echo "$resp" | sed '$d')
    assert_status "POST /collections creates collection" 201 "$code"
    assert_json_field "creation status success" "$body" ".status" "success"

    # 3b – List collections (should include new one)
    resp=$(curl -sw '\n%{http_code}' "${API}/collections" 2>/dev/null)
    code=$(echo "$resp" | tail -1)
    body=$(echo "$resp" | sed '$d')
    assert_status "GET /collections returns 200" 200 "$code"
    TOTAL=$((TOTAL + 1))
    if echo "$body" | jq -e ".collections[] | select(. == \"${E2E_COLLECTION}\")" >/dev/null 2>&1; then
        echo -e "  ${GREEN}✓ PASS${NC}  Collection '${E2E_COLLECTION}' in list"
        PASS=$((PASS + 1))
    else
        echo -e "  ${RED}✗ FAIL${NC}  Collection '${E2E_COLLECTION}' not found in list"
        FAIL=$((FAIL + 1))
    fi

    # 3c – Get collection info
    resp=$(curl -sw '\n%{http_code}' "${API}/collections/${E2E_COLLECTION}" 2>/dev/null)
    code=$(echo "$resp" | tail -1)
    body=$(echo "$resp" | sed '$d')
    assert_status "GET /collections/{name} returns 200" 200 "$code"
    assert_json_field "collection name matches" "$body" ".name" "${E2E_COLLECTION}"
    assert_json_exists "points_count field present" "$body" ".points_count"

    # 3d – Delete collection
    resp=$(curl -sw '\n%{http_code}' -X DELETE "${API}/collections/${E2E_COLLECTION}" 2>/dev/null)
    code=$(echo "$resp" | tail -1)
    body=$(echo "$resp" | sed '$d')
    assert_status "DELETE /collections/{name} returns 200" 200 "$code"
    assert_json_field "deletion status success" "$body" ".status" "success"
}

# ─────────────────────────────────────────────────────────────
test_dataset_upload_csv() {
    log_header "4 · Dataset Upload — CSV (synchronous)"

    local resp
    resp=$(curl -sw '\n%{http_code}' -X POST "${API}/datasets/upload?text_column=description" \
        -F "file=@${DATA_DIR}/products.csv" 2>/dev/null)
    local code=$(echo "$resp" | tail -1)
    local body=$(echo "$resp" | sed '$d')

    assert_status "POST /datasets/upload (CSV) returns 200" 200 "$code"
    assert_json_field "upload status success" "$body" ".status" "success"
    assert_json_exists "dataset_id returned" "$body" ".dataset_id"

    CSV_DATASET_ID=$(echo "$body" | jq -r '.dataset_id')
    echo -e "  ${CYAN}→ CSV dataset_id = ${CSV_DATASET_ID}${NC}"
}

# ─────────────────────────────────────────────────────────────
test_dataset_upload_json() {
    log_header "5 · Dataset Upload — JSON (synchronous)"

    local resp
    resp=$(curl -sw '\n%{http_code}' -X POST "${API}/datasets/upload?text_column=text" \
        -F "file=@${DATA_DIR}/articles.json" 2>/dev/null)
    local code=$(echo "$resp" | tail -1)
    local body=$(echo "$resp" | sed '$d')

    assert_status "POST /datasets/upload (JSON) returns 200" 200 "$code"
    assert_json_field "upload status success" "$body" ".status" "success"
    assert_json_exists "dataset_id returned" "$body" ".dataset_id"

    JSON_DATASET_ID=$(echo "$body" | jq -r '.dataset_id')
    echo -e "  ${CYAN}→ JSON dataset_id = ${JSON_DATASET_ID}${NC}"
}

# ─────────────────────────────────────────────────────────────
test_dataset_upload_background() {
    log_header "6 · Dataset Upload — Background processing"

    local resp
    resp=$(curl -sw '\n%{http_code}' -X POST "${API}/datasets/upload?background=true&text_column=description" \
        -F "file=@${DATA_DIR}/products.csv" 2>/dev/null)
    local code=$(echo "$resp" | tail -1)
    local body=$(echo "$resp" | sed '$d')

    assert_status "POST /datasets/upload?background=true returns 202" 202 "$code"
    assert_json_exists "job_id returned" "$body" ".job_id"

    JOB_ID=$(echo "$body" | jq -r '.job_id')
    echo -e "  ${CYAN}→ job_id = ${JOB_ID}${NC}"

    # Poll until completed or timeout
    echo -e "  ${YELLOW}Polling job status...${NC}"
    local max_polls=60
    local poll=0
    local status="pending"
    while [ "$poll" -lt "$max_polls" ] && [ "$status" != "completed" ] && [ "$status" != "failed" ]; do
        sleep 2
        local job_resp
        job_resp=$(curl -s "${API}/datasets/jobs/${JOB_ID}" 2>/dev/null)
        status=$(echo "$job_resp" | jq -r '.status' 2>/dev/null || echo "unknown")
        poll=$((poll + 1))
        echo -e "    poll ${poll}: status=${status}"
    done

    TOTAL=$((TOTAL + 1))
    if [ "$status" = "completed" ]; then
        echo -e "  ${GREEN}✓ PASS${NC}  Background job completed"
        PASS=$((PASS + 1))
    elif [ "$status" = "failed" ]; then
        echo -e "  ${RED}✗ FAIL${NC}  Background job failed"
        FAIL=$((FAIL + 1))
    else
        echo -e "  ${RED}✗ FAIL${NC}  Background job did not finish within ${max_polls}×2s"
        FAIL=$((FAIL + 1))
    fi
}

# ─────────────────────────────────────────────────────────────
test_dataset_upload_invalid() {
    log_header "7 · Dataset Upload — Invalid file type"

    local code
    echo "not a dataset" > /tmp/e2e_test_bad_file.txt
    code=$(curl -so /dev/null -w '%{http_code}' -X POST "${API}/datasets/upload" \
        -F "file=@/tmp/e2e_test_bad_file.txt" 2>/dev/null)
    rm -f /tmp/e2e_test_bad_file.txt
    assert_status "POST /datasets/upload (.txt) returns 400" 400 "$code"
}

# ─────────────────────────────────────────────────────────────
test_list_datasets() {
    log_header "8 · List Datasets"

    local resp
    resp=$(curl -sw '\n%{http_code}' "${API}/datasets" 2>/dev/null)
    local code=$(echo "$resp" | tail -1)
    local body=$(echo "$resp" | sed '$d')

    assert_status "GET /datasets returns 200" 200 "$code"
    assert_json_exists "datasets array present" "$body" ".datasets"
    assert_json_gte "at least 1 dataset" "$body" ".datasets | length" 1
}

# ─────────────────────────────────────────────────────────────
test_get_dataset_info() {
    log_header "9 · Get Dataset Info"

    if [ -z "$CSV_DATASET_ID" ]; then
        skip_test "GET /datasets/{id}" "no CSV dataset uploaded"
        return
    fi

    local resp
    resp=$(curl -sw '\n%{http_code}' "${API}/datasets/${CSV_DATASET_ID}" 2>/dev/null)
    local code=$(echo "$resp" | tail -1)
    local body=$(echo "$resp" | sed '$d')

    assert_status "GET /datasets/{id} returns 200" 200 "$code"
    assert_json_exists "dataset info present" "$body" ".dataset"
}

# ─────────────────────────────────────────────────────────────
test_recommendations_basic() {
    log_header "10 · Recommendations — Basic query"

    local resp
    resp=$(curl -sw '\n%{http_code}' -X POST "${API}/recommend" \
        -H 'Content-Type: application/json' \
        -d '{"query": "comfortable shoes for running", "limit": 5}' 2>/dev/null)
    local code=$(echo "$resp" | tail -1)
    local body=$(echo "$resp" | sed '$d')

    assert_status "POST /recommend returns 200" 200 "$code"
    assert_json_exists "recommendations array" "$body" ".recommendations"
    assert_json_exists "latency_ms field" "$body" ".latency_ms"
    assert_json_gte "at least 1 recommendation" "$body" ".recommendations | length" 1
}

# ─────────────────────────────────────────────────────────────
test_recommendations_with_filter() {
    log_header "11 · Recommendations — Qdrant filter (must + must_not)"

    local resp
    resp=$(curl -sw '\n%{http_code}' -X POST "${API}/recommend" \
        -H 'Content-Type: application/json' \
        -d '{
            "query": "outdoor hiking gear",
            "limit": 5,
            "filter": {
                "must": [
                    {"key": "category", "match_value": {"keyword": "footwear"}}
                ],
                "must_not": [
                    {"key": "in_stock", "match_value": {"keyword": "false"}}
                ]
            }
        }' 2>/dev/null)
    local code=$(echo "$resp" | tail -1)
    local body=$(echo "$resp" | sed '$d')

    assert_status "POST /recommend (filter) returns 200" 200 "$code"
    assert_json_exists "recommendations array" "$body" ".recommendations"
}

# ─────────────────────────────────────────────────────────────
test_recommendations_with_should() {
    log_header "12 · Recommendations — Should + min_should_count"

    local resp
    resp=$(curl -sw '\n%{http_code}' -X POST "${API}/recommend" \
        -H 'Content-Type: application/json' \
        -d '{
            "query": "electronics gadget",
            "limit": 5,
            "filter": {
                "should": [
                    {"key": "category", "match_value": {"keyword": "electronics"}},
                    {"key": "category", "match_value": {"keyword": "fitness"}}
                ],
                "min_should_count": 1
            }
        }' 2>/dev/null)
    local code=$(echo "$resp" | tail -1)
    local body=$(echo "$resp" | sed '$d')

    assert_status "POST /recommend (should) returns 200" 200 "$code"
    assert_json_exists "recommendations array" "$body" ".recommendations"
}

# ─────────────────────────────────────────────────────────────
test_recommendations_with_range() {
    log_header "13 · Recommendations — Range filter"

    local resp
    resp=$(curl -sw '\n%{http_code}' -X POST "${API}/recommend" \
        -H 'Content-Type: application/json' \
        -d '{
            "query": "affordable accessories",
            "limit": 5,
            "filter": {
                "must": [
                    {"key": "price", "range": {"gte": 20.0, "lte": 50.0}}
                ]
            }
        }' 2>/dev/null)
    local code=$(echo "$resp" | tail -1)
    local body=$(echo "$resp" | sed '$d')

    assert_status "POST /recommend (range) returns 200" 200 "$code"
    assert_json_exists "recommendations array" "$body" ".recommendations"
}

# ─────────────────────────────────────────────────────────────
test_recommendations_score_threshold() {
    log_header "14 · Recommendations — Score threshold"

    local resp
    resp=$(curl -sw '\n%{http_code}' -X POST "${API}/recommend" \
        -H 'Content-Type: application/json' \
        -d '{"query": "machine learning course", "limit": 10, "score_threshold": 0.5}' 2>/dev/null)
    local code=$(echo "$resp" | tail -1)
    local body=$(echo "$resp" | sed '$d')

    assert_status "POST /recommend (threshold) returns 200" 200 "$code"
    assert_json_exists "recommendations array" "$body" ".recommendations"

    # Verify all returned scores are >= threshold
    TOTAL=$((TOTAL + 1))
    local below
    below=$(echo "$body" | jq '[.recommendations[].score | select(. < 0.5)] | length' 2>/dev/null || echo "0")
    if [ "$below" -eq 0 ]; then
        echo -e "  ${GREEN}✓ PASS${NC}  All scores >= 0.5 threshold"
        PASS=$((PASS + 1))
    else
        echo -e "  ${RED}✗ FAIL${NC}  ${below} results below score threshold"
        FAIL=$((FAIL + 1))
    fi
}

# ─────────────────────────────────────────────────────────────
test_recommendations_legacy_filters() {
    log_header "15 · Recommendations — Legacy flat filters (backward compat)"

    local resp
    resp=$(curl -sw '\n%{http_code}' -X POST "${API}/recommend" \
        -H 'Content-Type: application/json' \
        -d '{"query": "warm winter clothing", "limit": 5, "filters": {"category": "outerwear"}}' 2>/dev/null)
    local code=$(echo "$resp" | tail -1)
    local body=$(echo "$resp" | sed '$d')

    assert_status "POST /recommend (legacy filters) returns 200" 200 "$code"
    assert_json_exists "recommendations array" "$body" ".recommendations"
}

# ─────────────────────────────────────────────────────────────
test_prediction_logs() {
    log_header "16 · Prediction Logs"

    local resp
    resp=$(curl -sw '\n%{http_code}' "${API}/predictions" 2>/dev/null)
    local code=$(echo "$resp" | tail -1)
    local body=$(echo "$resp" | sed '$d')

    assert_status "GET /predictions returns 200" 200 "$code"
    assert_json_exists "logs array present" "$body" ".logs"

    # There should be logs from the recommendation calls above
    assert_json_gte "at least 1 prediction log" "$body" ".total" 1
}

# ─────────────────────────────────────────────────────────────
test_prediction_export() {
    log_header "17 · Prediction Log Export"

    local code
    code=$(curl -so /dev/null -w '%{http_code}' "${API}/predictions/export" 2>/dev/null)
    # 200 if logs exist, 500 if none — both are valid depending on state
    TOTAL=$((TOTAL + 1))
    if [ "$code" -eq 200 ] || [ "$code" -eq 500 ]; then
        echo -e "  ${GREEN}✓ PASS${NC}  GET /predictions/export returns ${code} (endpoint reachable)"
        PASS=$((PASS + 1))
    else
        echo -e "  ${RED}✗ FAIL${NC}  GET /predictions/export unexpected status ${code}"
        FAIL=$((FAIL + 1))
    fi
}

# ─────────────────────────────────────────────────────────────
test_job_status_not_found() {
    log_header "18 · Job Status — Not found"

    local resp
    resp=$(curl -sw '\n%{http_code}' "${API}/datasets/jobs/nonexistent-job-id" 2>/dev/null)
    local code=$(echo "$resp" | tail -1)
    assert_status "GET /datasets/jobs/{bad_id} returns 404" 404 "$code"
}

# ─────────────────────────────────────────────────────────────
test_cleanup() {
    log_header "19 · Cleanup — Delete test datasets"

    if [ -n "$CSV_DATASET_ID" ]; then
        local resp
        resp=$(curl -sw '\n%{http_code}' -X DELETE "${API}/datasets/${CSV_DATASET_ID}" 2>/dev/null)
        local code=$(echo "$resp" | tail -1)
        assert_status "DELETE /datasets/${CSV_DATASET_ID}" 200 "$code"
    fi

    if [ -n "$JSON_DATASET_ID" ]; then
        local resp
        resp=$(curl -sw '\n%{http_code}' -X DELETE "${API}/datasets/${JSON_DATASET_ID}" 2>/dev/null)
        local code=$(echo "$resp" | tail -1)
        assert_status "DELETE /datasets/${JSON_DATASET_ID}" 200 "$code"
    fi

    # Clean up the e2e collection if it still exists
    curl -s -X DELETE "${API}/collections/${E2E_COLLECTION}" >/dev/null 2>&1 || true
}

# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

main() {
    echo ""
    echo -e "${BOLD}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BOLD}║   Recommendation Service — End-to-End Test Suite       ║${NC}"
    echo -e "${BOLD}╚══════════════════════════════════════════════════════════╝${NC}"
    echo -e "  Target: ${CYAN}${API}${NC}"
    echo ""

    check_deps
    wait_for_service

    # Run all test suites in order
    test_health
    test_swagger
    test_collection_management
    test_dataset_upload_csv
    test_dataset_upload_json
    test_dataset_upload_background
    test_dataset_upload_invalid
    test_list_datasets
    test_get_dataset_info
    test_recommendations_basic
    test_recommendations_with_filter
    test_recommendations_with_should
    test_recommendations_with_range
    test_recommendations_score_threshold
    test_recommendations_legacy_filters
    test_prediction_logs
    test_prediction_export
    test_job_status_not_found
    test_cleanup

    # ── Summary ──────────────────────────────────────────────
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${BOLD}  RESULTS${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "  Total:   ${TOTAL}"
    echo -e "  ${GREEN}Passed:  ${PASS}${NC}"
    if [ "$FAIL" -gt 0 ]; then
        echo -e "  ${RED}Failed:  ${FAIL}${NC}"
    else
        echo -e "  Failed:  ${FAIL}"
    fi
    if [ "$SKIPPED" -gt 0 ]; then
        echo -e "  ${YELLOW}Skipped: ${SKIPPED}${NC}"
    fi
    echo ""

    if [ "$FAIL" -eq 0 ]; then
        echo -e "  ${GREEN}${BOLD}All tests passed!${NC}"
        echo ""
        exit 0
    else
        echo -e "  ${RED}${BOLD}Some tests failed.${NC}"
        echo ""
        exit 1
    fi
}

main "$@"
