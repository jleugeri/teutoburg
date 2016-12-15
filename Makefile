SONAME = teutoburg.so
CC = g++
# -Wall -Wextra -O4
CCFLAGS = -fPIC  -std=c++11 -g -Wall -I/usr/local/include/ -I/usr/include/python3.4/ -I/usr/include/ -I./Sherwood/cpp/lib/ -I./wrapper/
LDFLAGS = -shared -fPIC -Wl,-soname,"$(SONAME)" -std=c++11 -Wall -Wextra -L/usr/local/lib/ -O4
BOOSTLIBS = -lboost_python-py34 -lboost_numpy -lboost_system

OBJ = wrapper/DataPointCollection.o wrapper/FeatureResponseFunctions.o wrapper/StatisticsAggregators.o wrapper/TrainingContexts.o wrapper.o

all: $(SONAME)

%.o: %.cpp
	$(CC) -c -o $@ $< $(CCFLAGS)

%.so: $(OBJ)
	$(CC) $(LDFLAGS) $(OBJ) $(BOOSTLIBS) -o $(SONAME)

clean:
	rm -f $(OBJ) $(SONAME)

.PHONY: all clean
