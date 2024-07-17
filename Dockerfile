# Use an official Python runtime as the base image
FROM python:3.12  

# Install system dependencies for PyQt6 and OpenGL
RUN apt-get update && apt-get install -y \
    python3-dev \
    qtbase5-dev \
    libqt5gui5 \
    libgl1-mesa-glx \
    libxkbcommon-x11-0

# Set up a virtual environment
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --upgrade pip
# Install Python dependencies
RUN apt-get update && apt-get install -y \
    python3-dev \  
    qtbase5-dev \ 
    libqt5gui5 \   
    libgl1-mesa-glx \ 
    libxkbcommon-x11-0 \
    build-essential \ 
    cmake \
    python3-pyqt5 \
    libqt5core5a \
    python3-sipbuild
# Install Python dependencies
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN pip install PyQt6 PyQt6-Qt6 PyQt6-stubs      
RUN pip install -r requirements.txt
RUN pip install PyQt6 PyQt6-Qt6 PyQt6-stubs  # Install PyQt6 and its stubs
RUN pip install pytest pytest-qt 

# Set the working directory within the container
WORKDIR /app

# Install illustris_python (as a submodule or package)
COPY .gitmodules .gitmodules
RUN git submodule update --init --recursive
RUN pip install -e ./illustris_python

# Copy the current directory contents into the container at /app
COPY . .

# Set the display variable
ENV DISPLAY=:99

# Set Qt to run in offscreen mode
ENV QT_QPA_PLATFORM=offscreen

# Command to run when the container starts
CMD ["pytest", "tests"]  # Or your specific stubtest command