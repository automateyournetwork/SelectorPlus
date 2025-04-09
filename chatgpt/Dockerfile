FROM python:3.11-slim

WORKDIR /app

RUN pip install requests
RUN pip install python-dotenv
RUN pip install openai 

COPY . .

CMD ["python", "-u", "server.py"]