import requests
import time

# IP & Port of AI laptop (AI system runs on port 9000, not 8000)
AI_IP = "172.16.3.180"
AI_PORT = 9001         # CORRECTED: AI system runs on port 9000

print(f"🔗 Connecting to AI system at {AI_IP}:{AI_PORT}")

# Gas leak detection payload
payload = {"sensor": "gas_sensor_1", "status": "leak_detected"}

try:
    print("📡 Sending gas leak alert...")
    r = requests.post(f"http://{AI_IP}:{AI_PORT}", json=payload, timeout=10)
    print("✅ Sent sensor data:", payload)
    print("📥 AI Response:", r.text)
    print("✅ Alert successfully transmitted!")
    
except requests.exceptions.ConnectTimeout:
    print(f"❌ Connection timeout - AI system at {AI_IP}:{AI_PORT} is not responding")
    print("💡 Make sure:")
    print("   1. AI system (ai.py) is running")
    print("   2. AI system is listening on port 9000")
    print("   3. IP address is correct")
    
except requests.exceptions.ConnectionError as e:
    print(f"❌ Connection error - Cannot reach AI system at {AI_IP}:{AI_PORT}")
    print("💡 Check if AI system is running and network is accessible")
    print(f"🔍 Error details: {e}")
    
except Exception as e:
    print(f"❌ Unexpected error: {e}")

print("🔄 Sensor simulation complete")
time.sleep(2)