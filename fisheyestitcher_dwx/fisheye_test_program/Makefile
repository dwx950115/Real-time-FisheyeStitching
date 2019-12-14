
all:
	g++ fish2sphere_test1.cpp `pkg-config opencv --cflags --libs` -o build/fish2sphere_test1
	g++ fish2sphere_test2.cpp `pkg-config opencv --cflags --libs` -o build/fish2sphere_test2
	g++ singlecam.cpp `pkg-config opencv --cflags --libs` -o build/singlecam
clean:
	rm build/fish2sphere_test1
	rm build/fish2sphere_test2
	rm build/singlecam
