all:
	g++ -std=c++11 videocapture_save.cpp `pkg-config opencv --cflags --libs` -o build/videocapture_save -lpthread
	g++ videocapture_source.cpp `pkg-config opencv --cflags --libs` -o build/videocapture_source
	g++ -std=c++11 videocapture_two_thread.cpp `pkg-config opencv --cflags --libs` -o build/videocapture_two_thread -lpthread
	g++ -std=c++11 videocapture_thread.cpp `pkg-config opencv --cflags --libs` -o build/videocapture_thread -lpthread
clean:
	rm build/videocapture_save
	rm build/videocapture_source
	rm build/videocapture_two_thread
	rm build/videocapture_thread
	
