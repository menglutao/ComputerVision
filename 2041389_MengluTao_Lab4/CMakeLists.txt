cmake_minimum_required(VERSION 3.0.0)
project(Start_YTB VERSION 0.1.0)

include(CTest)
enable_testing()

find_package( OpenCV REQUIRED )
add_executable(Start_YTB main.cpp)

target_link_libraries( Start_YTB ${OpenCV_LIBS} )

include_directories(${OpenCV_INCLUDE_DIRS})

set(CMAKE_CXX_STARDARD 11)
set(CMAKE_CXX_STARDARD_REQUIRED true)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
