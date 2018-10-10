CXX := nvcc
CUDNN_PATH := ../cuda
HEADERS := -I $(CUDNN_PATH)/include
LIBS := -L $(CUDNN_PATH)/lib64 -L/usr/local/lib
CXXFLAGS := -std=c++11 -O2 -D_FORCE_INLINES

all: conv
conv: conv.cu
	$(CXX) $(CXXFLAGS) $(HEADERS) $(LIBS) conv.cu -o conv
	-lcudnn -lopencv_imgcodecs -lopencv_imgproc -lopencv_core
vecadd: vecadd.cu
	$(CXX) vecadd.cu -o vecadd
matrix: matrix1.cu
	$(CXX) matrix1.cu -o matrix
clean:
	rm $(TARGET) vecadd
