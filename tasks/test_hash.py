import time
import hashlib

start = time.time()
data = b"edge-cloud-task"
hashes = 0
while time.time() - start < 1.0:
    data = hashlib.sha256(data).digest()
    hashes += 1

print(f"Hashes per second: {hashes}")
