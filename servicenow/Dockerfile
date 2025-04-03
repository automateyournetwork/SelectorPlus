FROM python:3.11-slim

WORKDIR /app

RUN pip install requests
RUN pip install python-dotenv
RUN apt-get update && apt-get install -y curl

COPY . .

CMD ["python", "-u", "server.py"]

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s \
  CMD python -c "import socket; s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM); s.connect('/tmp/mcp.sock'); s.send(b'{\"tool_name\": \"ready\"}\\n'); data = s.recv(1024); s.close(); import json; result = json.loads(data); exit(0 if result.get('status') == 'ready' else 1)" || exit 1