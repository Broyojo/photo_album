import json
import subprocess

# Run ffprobe command
command = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', "images/video.webm"]
result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

# Parse the output to JSON
metadata = json.loads(result.stdout)

print(metadata)