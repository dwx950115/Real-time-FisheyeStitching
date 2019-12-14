cmake_minimum_required(VERSION 2.8)
project( cmake_exercise )
find_package( OpenCV REQUIRED )


message(STATUS "OpenCV library status:")
message(STATUS "    version: ${OpenCV_VERSION}")
message(STATUS "    libraries: ${OpenCV_LIBS}")
message(STATUS "    include path: ${OpenCV_INCLUDE_DIRS}")


aux_source_directory(src/ CMAEK_EXERCISE_SRC)

add_executable(
  fish2sphere_test1
  ${CMAEK_EXERCISE_SRC}
)

target_link_libraries( fish2sphere_test1 ${OpenCV_LIBS} )
set_property(TARGET fish2sphere_test1 PROPERTY CXX_STANDARD 11)

