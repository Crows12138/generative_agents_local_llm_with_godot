#!/usr/bin/env python
"""
LLM Client - Fast client for the no-cache server
Usage: python llm_client.py "Your message"
"""

import sys
import socket

def get_response(message, port=9999):
    """Get response from LLM server"""
    try:
        # Connect to server
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.settimeout(30)  # 30 second timeout
        client.connect(('127.0.0.1', port))
        
        # Send message
        client.send(message.encode('utf-8'))
        
        # Receive response
        response = client.recv(4096).decode('utf-8')
        client.close()
        
        return response
        
    except socket.error as e:
        return f"Server not running. Start with: python llm_server_nocache.py"
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == "__main__":
    if len(sys.argv) > 1:
        message = " ".join(sys.argv[1:])
        response = get_response(message)
        print(response)
    else:
        print("Usage: python llm_client.py 'Your message'")