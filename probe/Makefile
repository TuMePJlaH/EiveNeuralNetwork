TARGET=binary
LIB=-leivennetwork
LIB_FOLDER=../
LIB_FILE=libeivennetwork.so
SRC=main.cpp
INCLUDE_DIR=../include

$(TARGET): $(SRC) $(LIB_FOLDER)$(LIB_FILE)
	g++ -o $@ $(SRC) -I $(INCLUDE_DIR) -std=c++11 -L$(LIB_FOLDER) $(LIB) -Wl,-rpath,$(LIB_FOLDER)

clean:
	rm $(TARGET)
