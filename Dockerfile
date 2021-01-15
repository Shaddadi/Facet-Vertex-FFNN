FROM python:3.6
WORKDIR /facet-vertex-ffnn
COPY . ./
RUN pip install --no-cache-dir -r requirements.txt
RUN useradd -m docker && echo "docker:docker" | chpasswd && adduser docker sudo
