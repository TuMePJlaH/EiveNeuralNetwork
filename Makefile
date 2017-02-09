TARGET=libeivennetwork.so

NVCC=/usr/local/cuda/bin/nvcc
SOURCES_DIR=./src/
INCLUDE_DIR=./include/
SOURCES=$(wildcard $(SOURCES_DIR)*.cu)
OBJECTS=$(SOURCES:.cu=.o)

default: $(TARGET)
all: default

$(TARGET): $(OBJECTS)
	$(NVCC) -shared -o $@ $(OBJECTS) -Wno-deprecated-gpu-targets 

%.o: %.cu
	$(NVCC) -c -Xcompiler "-fPIC" -std=c++11 -I $(INCLUDE_DIR) -o $@ $< -Wno-deprecated-gpu-targets  

clean:
	rm $(OBJECTS) $(TARGET)
