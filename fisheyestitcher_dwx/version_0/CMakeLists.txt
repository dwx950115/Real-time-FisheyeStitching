cmake_minimum_required(VERSION 2.8)
project( fisheye360 )
find_package( OpenCV REQUIRED )


message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")


include_directories(
  include/
  ${OpenCV_INCLUDE_DIRS}
)

aux_source_directory(src/ FISHEYE360_SRC)

add_executable(
  test_unwarp
  tests/test_unwarp.cc
  ${FISHEYE360_SRC}
)

target_link_libraries( test_unwarp ${OpenCV_LIBS} )
set_property(TARGET test_unwarp PROPERTY CXX_STANDARD 11)

