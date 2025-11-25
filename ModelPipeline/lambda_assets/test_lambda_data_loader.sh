#!/bin/bash
# Test S3StreamingLoader in Lambda environment via SAM Local
# Requires: Docker Desktop running

set -e  # Exit on error

echo "========================================================================"
echo "DATA LOADER - SAM LOCAL TEST"
echo "========================================================================"
echo ""
echo "Testing S3StreamingLoader in Lambda container with REAL S3 access"
echo ""

# Check Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ ERROR: Docker is not running"
    echo "   Start Docker Desktop and try again"
    exit 1
fi

echo "✓ Docker is running"
echo ""

# Navigate to lambda_assets directory
cd "$(dirname "$0")"

# Build Lambda package
echo "[1/3] Building Lambda package..."
echo "This may take 2-3 minutes on first run (downloads base images)"
echo ""
sam build --use-container

echo ""
echo "[2/3] Testing S3StreamingLoader via Lambda invoke..."
echo ""

# Invoke Lambda with test event
sam local invoke DataLoaderTestFunction \
    --event test_events/test_data_loader.json

echo ""
echo "[3/3] Test complete!"
echo ""
echo "========================================================================"
echo "✅ If you see 'ALL TESTS PASSED', S3StreamingLoader works in Lambda"
echo "========================================================================"