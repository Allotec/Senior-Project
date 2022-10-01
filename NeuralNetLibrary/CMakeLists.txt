cmake_minimum_required(VERSION 3.0.0)
project(NeuralNetLibrary VERSION 0.1.0)

include(CTest)
enable_testing()

add_executable(NeuralNetLibrary main.cpp)

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)

INCLUDE_DIRECTORIES(
    C:/Libraries/vcpkg/installed/x86-windows/include
)
LINK_DIRECTORIES(
    C:/Libraries/vcpkg/installed/x86-windows/lib
) 
LINK_DIRECTORIES(
    C:/Libraries
) 