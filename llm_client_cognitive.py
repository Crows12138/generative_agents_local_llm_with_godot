#!/usr/bin/env python
"""
LLM Client for Cognitive Bar Server
Connects to optimized_cozy_bar_server.py on port 9999
Usage: python llm_client_cognitive.py [ping|dialogue] [npc_name] [message]
"""

import sys
import socket
import time

def ping_server(port=9999):
    """Check if server is running"""
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(1)
        client.connect(('127.0.0.1', port))
        client.close()
        return "Server is running"
    except:
        return "Server not running"

def send_dialogue(npc_name, message, port=9999):
    """Send dialogue to server using clean protocol"""
    try:
        # Connect to server
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(30)
        client.connect(('127.0.0.1', port))
        
        # Clean protocol: NPC_NAME|MESSAGE
        # Message already comes in this format from Godot
        client.send(message.encode('utf-8'))
        
        # Receive response with larger buffer and loop to get full response
        response = ""
        client.settimeout(5)  # 5 second timeout for receiving
        
        while True:
            try:
                data = client.recv(4096)
                if not data:
                    break
                response += data.decode('utf-8', errors='ignore')
                # Check if we got a complete response (simple heuristic)
                if len(data) < 4096:
                    break
            except socket.timeout:
                break
        
        client.close()
        
        # Clean the response - remove any extra formatting
        response = response.strip()
        if not response:
            return "..."
        
        return response
        
    except socket.error as e:
        return f"Server not running. Please run: START_OPTIMIZED_COZY_BAR.bat"
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    # Fix encoding for Windows
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    
    if len(sys.argv) < 2:
        print("Usage: python llm_client_cognitive.py [ping|dialogue] [npc_name] [message]")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "ping":
        result = ping_server()
        print(result)
    elif command == "dialogue" and len(sys.argv) >= 4:
        npc_name = sys.argv[2]
        message = sys.argv[3]  # This already contains NPC_NAME|MESSAGE format
        response = send_dialogue(npc_name, message)
        print(response)
    else:
        print("Invalid command")