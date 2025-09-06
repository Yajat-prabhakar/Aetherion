from http.server import BaseHTTPRequestHandler, HTTPServer
import json
import webbrowser
import urllib.parse
import subprocess
import pyautogui
import time
import requests
import re
import os
from groq import Groq

HOST = "0.0.0.0"
PORT = 8000

# Initialize Groq client
client = Groq(api_key=os.getenv('GROQ_API_KEY', ''))

class MCPHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        content_length = int(self.headers['Content-Length'])
        post_data = self.rfile.read(content_length)
        data = json.loads(post_data.decode("utf-8"))
        
        action = data.get("action")
        
        if action == "open_browser":
            query = data.get("query", "how are you")
            url = f"https://www.google.com/search?q={urllib.parse.quote(query)}"
            webbrowser.open(url)
            response = {"status": "ok", "message": f"Opened browser with query: {query}"}
            print("✅ Action executed:", response)
            
        elif action == "execute_prompt":
            prompt = data.get("prompt", "")
            response = self.process_prompt(prompt)
            print("✅ Prompt processed:", response)
            
        else:
            response = {"status": "error", "message": "Unknown action"}
        
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(response).encode("utf-8"))
    
    def process_prompt(self, prompt):
        """Process natural language prompt using Groq AI"""
        try:
            print(f"🧠 Processing prompt: {prompt}")
            
            # Create a system prompt for the AI to understand the task
            system_prompt = """You are an AI assistant that converts natural language commands into executable Python code using pyautogui, subprocess, and OpenCV.

Your job is to analyze the user's request and generate Python code that can accomplish the task.

Available modules:
- pyautogui (for GUI automation)
- subprocess (for running programs)
- time (for delays)
- os (for system operations)
- cv2 (for camera operations)

IMPORTANT RULES:
1. Return ONLY the Python code, no explanations or markdown formatting
2. Always include necessary imports at the top
3. Add appropriate time.sleep() delays for GUI operations
4. Use try-except blocks for error handling
5. For opening applications, use subprocess.Popen()
6. Be specific about application names (e.g., "excel.exe", "notepad.exe", "calc.exe")
7. Include comments explaining each step
8. Do not use markdown code blocks or backticks

For Excel operations:
- Use subprocess.Popen([r"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE"]) to open Excel
- Wait 3-5 seconds for Excel to load
- Click on specific cells using pyautogui.click()
- Use pyautogui.typewrite() to enter text
- Use pyautogui.press('enter') to move to next cell

Example for "open excel and list all states in India":

import subprocess
import pyautogui
import time
from mcp.server.fastmcp import FastMCP

# Create MCP server instance
mcp = FastMCP("excel-agent")

# ---- Tool: Open Excel and type Indian states ----
@mcp.tool()
def open_excel_and_type_states() -> str:
    states = [
        "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh",
        "Goa", "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka",
        "Kerala", "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya",
        "Mizoram", "Nagaland", "Odisha", "Punjab", "Rajasthan", "Sikkim",
        "Tamil Nadu", "Telangana", "Tripura", "Uttar Pradesh", "Uttarakhand",
        "West Bengal"
    ]

    try:
        # 1. Open Excel
        subprocess.Popen([r"C:\Program Files\Microsoft Office\root\Office16\EXCEL.EXE"])
        time.sleep(5)  # wait for Excel to launch

        # 2. Type header
        pyautogui.typewrite("State Name")
        pyautogui.press("enter")

        # 3. Type each state
        for state in states:
            pyautogui.typewrite(state)
            pyautogui.press("enter")
        
        print("Successfully listed all states in Excel")
    except Exception as e:
        print(f"Error: {e}")

For Camera operations:
- Use OpenCV to access the webcam
- Example:

import cv2

try:
    # Open webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Webcam", frame)

        # Press 'q' to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print("Webcam closed successfully")

except Exception as e:
    print(f"Error: {e}")
"""

            # Get AI response
            chat_completion = client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.1,
                max_tokens=1000
            )
            
            ai_response = chat_completion.choices[0].message.content
            print(f"🤖 AI Response: {ai_response}")
            
            # Extract Python code from the response
            code_match = re.search(r'python\n(.*?)\n', ai_response, re.DOTALL)
            if code_match:
                python_code = code_match.group(1)
            else:
                # If no code blocks found, assume the entire response is code
                python_code = ai_response.strip()
            
            print(f"🐍 Extracted code:\n{python_code}")
            
            # Execute the generated code
            try:
                # Create a safe execution environment
                exec_globals = {
                    'subprocess': subprocess,
                    'pyautogui': pyautogui,
                    'time': time,
                    'os': os,
                    'print': print
                }
                
                # Execute the code
                exec(python_code, exec_globals)
                
                return {
                    "status": "success", 
                    "message": f"Successfully executed prompt: {prompt}",
                    "generated_code": python_code
                }
                
            except Exception as exec_error:
                print(f"❌ Execution error: {exec_error}")
                return {
                    "status": "error", 
                    "message": f"Error executing code: {exec_error}",
                    "generated_code": python_code
                }
                
        except Exception as e:
            print(f"❌ Error processing prompt: {e}")
            return {
                "status": "error", 
                "message": f"Error processing prompt: {e}"
            }

if _name_ == "_main_":
    print(f"💻 Enhanced Server running on {HOST}:{PORT}")
    print("🧠 Groq AI integration enabled")
    print("🤖 Ready to process natural language prompts")
    server = HTTPServer((HOST, PORT), MCPHandler)
    server.serve_forever()