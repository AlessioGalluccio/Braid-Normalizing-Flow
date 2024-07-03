FROM python:3.8

WORKDIR /exp

# Install extras
#COPY requirements.txt requirements.txt
COPY . .
RUN pip install -r requirements.txt

#COPY . .
CMD ["python", "main.py"]


