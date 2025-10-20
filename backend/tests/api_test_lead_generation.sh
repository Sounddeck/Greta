#!/bin/bash

#############################################################################
# Lead Generation API Test Script
#
# Tests all 12 API endpoints for the lead generation module using curl.
# Includes assertions to verify expected responses.
#
# Usage: ./api_test_lead_generation.sh [BASE_URL]
# Example: ./api_test_lead_generation.sh http://localhost:8000
#############################################################################

set -e  # Exit on error

# Configuration
BASE_URL="${1:-http://localhost:8000}"
API_PREFIX="/api/v1/lead-generation"
TEST_USER_ID="test_user_$(date +%s)"
CAMPAIGN_ID=""
LEAD_ID=""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Helper functions
print_test() {
    echo -e "${YELLOW}[TEST]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[✓]${NC} $1"
}

print_error() {
    echo -e "${RED}[✗]${NC} $1"
}

assert_contains() {
    local response="$1"
    local expected="$2"
    local test_name="$3"
    
    if echo "$response" | grep -q "$expected"; then
        print_success "$test_name"
        return 0
    else
        print_error "$test_name - Expected to find: $expected"
        echo "Response: $response"
        return 1
    fi
}

assert_status() {
    local status="$1"
    local expected="$2"
    local test_name="$3"
    
    if [ "$status" -eq "$expected" ]; then
        print_success "$test_name (HTTP $status)"
        return 0
    else
        print_error "$test_name - Expected HTTP $expected, got HTTP $status"
        return 1
    fi
}

#############################################################################
# Test 1: Health Check
#############################################################################
print_test "Test 1: Health Check"

response=$(curl -s -w "\n%{http_code}" "${BASE_URL}${API_PREFIX}/health")
http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')

assert_status "$http_code" "200" "Health check endpoint"
assert_contains "$body" "healthy" "Health status is healthy"
assert_contains "$body" "active_campaigns" "Response contains active_campaigns"
assert_contains "$body" "total_leads" "Response contains total_leads"

echo ""

#############################################################################
# Test 2: Create Campaign
#############################################################################
print_test "Test 2: Create Campaign"

campaign_payload=$(cat <<EOF
{
  "user_id": "$TEST_USER_ID",
  "config": {
    "campaign_name": "Test AI Startups Campaign",
    "target_industry": "Artificial Intelligence",
    "target_location": "San Francisco, CA",
    "company_size": "small",
    "keywords": ["AI", "machine learning", "LLM"],
    "research_depth": "medium",
    "monitoring_frequency": "weekly",
    "max_leads": 50,
    "auto_research": true
  }
}
EOF
)

response=$(curl -s -w "\n%{http_code}" \
    -X POST "${BASE_URL}${API_PREFIX}/campaigns" \
    -H "Content-Type: application/json" \
    -d "$campaign_payload")

http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')

assert_status "$http_code" "200" "Create campaign endpoint"
assert_contains "$body" "campaign_id" "Response contains campaign_id"
assert_contains "$body" "campaign_" "Campaign ID has correct prefix"

# Extract campaign ID for later tests
CAMPAIGN_ID=$(echo "$body" | grep -o '"campaign_id":"[^"]*"' | cut -d'"' -f4)
print_success "Created campaign: $CAMPAIGN_ID"

echo ""

#############################################################################
# Test 3: Get Campaign
#############################################################################
print_test "Test 3: Get Campaign"

response=$(curl -s -w "\n%{http_code}" \
    "${BASE_URL}${API_PREFIX}/campaigns/${CAMPAIGN_ID}")

http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')

assert_status "$http_code" "200" "Get campaign endpoint"
assert_contains "$body" "$CAMPAIGN_ID" "Response contains correct campaign ID"
assert_contains "$body" "Test AI Startups Campaign" "Response contains campaign name"
assert_contains "$body" "active" "Campaign status is active"

echo ""

#############################################################################
# Test 4: List Campaigns
#############################################################################
print_test "Test 4: List Campaigns"

response=$(curl -s -w "\n%{http_code}" \
    "${BASE_URL}${API_PREFIX}/campaigns?user_id=${TEST_USER_ID}")

http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')

assert_status "$http_code" "200" "List campaigns endpoint"
assert_contains "$body" "campaigns" "Response contains campaigns array"
assert_contains "$body" "$CAMPAIGN_ID" "Response contains our campaign"

echo ""

#############################################################################
# Test 5: Create Lead
#############################################################################
print_test "Test 5: Create Lead"

lead_payload=$(cat <<EOF
{
  "campaign_id": "$CAMPAIGN_ID",
  "user_id": "$TEST_USER_ID",
  "company_name": "Acme AI Corp",
  "industry": "Artificial Intelligence",
  "location": "San Francisco, CA",
  "website": "https://acmeai.com",
  "company_size": "50-200"
}
EOF
)

response=$(curl -s -w "\n%{http_code}" \
    -X POST "${BASE_URL}${API_PREFIX}/leads" \
    -H "Content-Type: application/json" \
    -d "$lead_payload")

http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')

assert_status "$http_code" "200" "Create lead endpoint"
assert_contains "$body" "lead_id" "Response contains lead_id"
assert_contains "$body" "lead_" "Lead ID has correct prefix"

# Extract lead ID for later tests
LEAD_ID=$(echo "$body" | grep -o '"lead_id":"[^"]*"' | cut -d'"' -f4)
print_success "Created lead: $LEAD_ID"

echo ""

#############################################################################
# Test 6: Get Lead
#############################################################################
print_test "Test 6: Get Lead"

response=$(curl -s -w "\n%{http_code}" \
    "${BASE_URL}${API_PREFIX}/leads/${LEAD_ID}")

http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')

assert_status "$http_code" "200" "Get lead endpoint"
assert_contains "$body" "$LEAD_ID" "Response contains correct lead ID"
assert_contains "$body" "Acme AI Corp" "Response contains company name"
assert_contains "$body" "new" "Lead status is new"

echo ""

#############################################################################
# Test 7: List Leads
#############################################################################
print_test "Test 7: List Leads"

response=$(curl -s -w "\n%{http_code}" \
    "${BASE_URL}${API_PREFIX}/leads?campaign_id=${CAMPAIGN_ID}")

http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')

assert_status "$http_code" "200" "List leads endpoint"
assert_contains "$body" "leads" "Response contains leads array"
assert_contains "$body" "$LEAD_ID" "Response contains our lead"
assert_contains "$body" "total" "Response contains total count"

echo ""

#############################################################################
# Test 8: Research Lead
#############################################################################
print_test "Test 8: Research Lead"

research_payload=$(cat <<EOF
{
  "research_depth": "medium"
}
EOF
)

response=$(curl -s -w "\n%{http_code}" \
    -X POST "${BASE_URL}${API_PREFIX}/leads/${LEAD_ID}/research" \
    -H "Content-Type: application/json" \
    -d "$research_payload")

http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')

assert_status "$http_code" "200" "Research lead endpoint"
assert_contains "$body" "research_id" "Response contains research_id"
assert_contains "$body" "completed" "Research status is completed"

echo ""

#############################################################################
# Test 9: Start Monitoring
#############################################################################
print_test "Test 9: Start Monitoring"

monitoring_payload=$(cat <<EOF
{
  "frequency": "daily"
}
EOF
)

response=$(curl -s -w "\n%{http_code}" \
    -X POST "${BASE_URL}${API_PREFIX}/leads/${LEAD_ID}/monitor" \
    -H "Content-Type: application/json" \
    -d "$monitoring_payload")

http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')

assert_status "$http_code" "200" "Start monitoring endpoint"
assert_contains "$body" "monitoring_active" "Response contains monitoring_active"
assert_contains "$body" "true" "Monitoring is active"

echo ""

#############################################################################
# Test 10: Get Campaign Stats
#############################################################################
print_test "Test 10: Get Campaign Stats"

response=$(curl -s -w "\n%{http_code}" \
    "${BASE_URL}${API_PREFIX}/campaigns/${CAMPAIGN_ID}/stats")

http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')

assert_status "$http_code" "200" "Campaign stats endpoint"
assert_contains "$body" "total_leads" "Response contains total_leads"
assert_contains "$body" "leads_by_status" "Response contains leads_by_status"
assert_contains "$body" "average_score" "Response contains average_score"

echo ""

#############################################################################
# Test 11: Stop Monitoring
#############################################################################
print_test "Test 11: Stop Monitoring"

response=$(curl -s -w "\n%{http_code}" \
    -X DELETE "${BASE_URL}${API_PREFIX}/leads/${LEAD_ID}/monitor")

http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')

assert_status "$http_code" "200" "Stop monitoring endpoint"
assert_contains "$body" "monitoring_active" "Response contains monitoring_active"
assert_contains "$body" "false" "Monitoring is stopped"

echo ""

#############################################################################
# Test 12: List Leads with Filters
#############################################################################
print_test "Test 12: List Leads with Filters"

response=$(curl -s -w "\n%{http_code}" \
    "${BASE_URL}${API_PREFIX}/leads?status=new&page=1&page_size=20")

http_code=$(echo "$response" | tail -n1)
body=$(echo "$response" | sed '$d')

assert_status "$http_code" "200" "List leads with filters endpoint"
assert_contains "$body" "leads" "Response contains leads array"
assert_contains "$body" "page" "Response contains page number"

echo ""

#############################################################################
# Summary
#############################################################################
echo -e "${GREEN}========================================${NC}"
echo -e "${GREEN}All API tests completed successfully!${NC}"
echo -e "${GREEN}========================================${NC}"
echo ""
echo "Test Summary:"
echo "  - Campaign ID: $CAMPAIGN_ID"
echo "  - Lead ID: $LEAD_ID"
echo "  - User ID: $TEST_USER_ID"
echo ""
echo "All 12 endpoints tested:"
echo "  ✓ Health check"
echo "  ✓ Create campaign"
echo "  ✓ Get campaign"
echo "  ✓ List campaigns"
echo "  ✓ Create lead"
echo "  ✓ Get lead"
echo "  ✓ List leads"
echo "  ✓ Research lead"
echo "  ✓ Start monitoring"
echo "  ✓ Campaign stats"
echo "  ✓ Stop monitoring"
echo "  ✓ List leads with filters"

