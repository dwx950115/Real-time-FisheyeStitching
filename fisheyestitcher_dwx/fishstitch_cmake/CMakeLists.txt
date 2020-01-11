cmake_minimum_required(VERSION 2.8)
project( fishstitch_cmake )
find_package( OpenCV REQUIRED )
find_package( Boost REQUIRED system thread timer chrono)

include(CheckCXXCompilerFlag) 
CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)

message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "    thread path: ${Boost_LIBRARIES}")


aux_source_directory(src/ CMAEK_EXERCISE_SRC)

add_executable(
  fisheye_capture
  ${CMAEK_EXERCISE_SRC}
)

target_link_libraries( fisheye_capture ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT} ${bOOST_libraries} )
set_property(TARGET fisheye_capture PROPERTY CXX_STANDARD 11)

