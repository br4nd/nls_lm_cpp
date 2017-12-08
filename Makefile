CPP_FILES = $(wildcard src/*.cpp)
O_FILES = $(CPP_FILES:src/%.cpp=build/%.o)

.PHONY: all clean
.DEFAULT: all

all: nls_lm

nls_lm: $(O_FILES)
	g++ -o $@ $^

build:
	@mkdir -p build

build/%.o: src/%.cpp | build
	#g++ --verbose -c $< -o $@
	g++ --std=c++0x -c $< -o $@

clean:
	-rm -rf build
