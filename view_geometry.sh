#!/bin/bash

# Xeno Manifold Geometry Viewer Launcher
# This script starts a local web server and opens the 3D geometry viewer

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Configuration
PORT=8888
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HTML_FILE="view_geometry.html"

echo -e "${BLUE}╔════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║   🛸 Xeno Manifold Geometry Viewer Launcher      ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════╝${NC}"
echo ""

# Check if HTML file exists
if [ ! -f "$SCRIPT_DIR/$HTML_FILE" ]; then
    echo -e "${YELLOW}⚠️  Error: $HTML_FILE not found in $SCRIPT_DIR${NC}"
    exit 1
fi

# Check if port is already in use
if lsof -Pi :$PORT -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Port $PORT is already in use${NC}"
    echo -e "${BLUE}🌐 Opening viewer at http://localhost:$PORT/$HTML_FILE${NC}"
    open "http://localhost:$PORT/$HTML_FILE"
    exit 0
fi

# Start the web server
echo -e "${GREEN}🚀 Starting web server on port $PORT...${NC}"
cd "$SCRIPT_DIR"

# Start Python HTTP server in background
python3 -m http.server $PORT > /dev/null 2>&1 &
SERVER_PID=$!

# Save PID to file for cleanup
echo $SERVER_PID > .geometry_viewer_server.pid

# Wait a moment for server to start
sleep 1

# Check if server started successfully
if ! ps -p $SERVER_PID > /dev/null 2>&1; then
    echo -e "${YELLOW}⚠️  Failed to start server${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Server started successfully (PID: $SERVER_PID)${NC}"
echo -e "${BLUE}🌐 Opening viewer at http://localhost:$PORT/$HTML_FILE${NC}"
echo ""
echo -e "${GREEN}📂 To view geometry:${NC}"
echo -e "   1. Click '📂 Choose OBJ File' button"
echo -e "   2. Select a .obj file (e.g., manifold_geometry.obj)"
echo -e "   3. Or drag & drop an OBJ file onto the canvas"
echo ""
echo -e "${YELLOW}⌨️  Press Ctrl+C to stop the server${NC}"
echo ""

# Open in default browser
open "http://localhost:$PORT/$HTML_FILE"

# Trap Ctrl+C to cleanup
trap cleanup SIGINT SIGTERM

cleanup() {
    echo ""
    echo -e "${YELLOW}🛑 Stopping server...${NC}"
    kill $SERVER_PID 2>/dev/null
    rm -f .geometry_viewer_server.pid
    echo -e "${GREEN}✓ Server stopped${NC}"
    exit 0
}

# Keep script running
echo -e "${GREEN}✓ Viewer is now open in your browser${NC}"
echo -e "${BLUE}ℹ️  Server is running... Press Ctrl+C to stop${NC}"

# Wait for server process
wait $SERVER_PID
