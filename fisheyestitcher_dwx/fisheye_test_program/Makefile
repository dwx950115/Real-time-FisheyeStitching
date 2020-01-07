
all:
	g++ fish2sphere_test1.cpp `pkg-config opencv --cflags --libs` -o build/fish2sphere_test1
	g++ fish2sphere_test2.cpp `pkg-config opencv --cflags --libs` -o build/fish2sphere_test2
	g++ -std=c++11 fisheye_thread.cpp `pkg-config opencv --cflags --libs` -o build/fisheye_thread -lpthread
	g++ -std=c++11 fisheye_thread_light.cpp `pkg-config opencv --cflags --libs` -o build/fisheye_thread_light -lpthread
clean:
	rm build/fish2sphere_test1
	rm build/fish2sphere_test2
	rm build/fisheye_thread
	rm build/fisheye_thread_light


