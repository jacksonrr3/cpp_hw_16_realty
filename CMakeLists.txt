cmake_minimum_required(VERSION 3.2)

project(realty VERSION 0.0.$ENV{TRAVIS_BUILD_NUMBER})

include_directories(dlib)
link_directories(dlib/dlib)

find_package(Threads REQUIRED)

add_executable(rclst rclst.cpp)

add_executable(rclss rclss.cpp)

set_target_properties(rclst PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON
  COMPILE_OPTIONS "-Wpedantic;-Wall;-Wextra"
)

set_target_properties(rclss PROPERTIES
  CXX_STANDARD 14
  CXX_STANDARD_REQUIRED ON
  COMPILE_OPTIONS "-Wpedantic;-Wall;-Wextra"
)

target_link_libraries(rclst 
                      dlib
                      openblas 
                      ${CMAKE_THREAD_LIBS_INIT})

target_link_libraries(rclst 
                      dlib
                      openblas
                      ${CMAKE_THREAD_LIBS_INIT})

install(TARGETS rclst RUNTIME DESTINATION bin)
install(TARGETS rclss RUNTIME DESTINATION bin)

set(CPACK_PACKAGE_VERSION_MAJOR "${PROJECT_VERSION_MAJOR}")
set(CPACK_PACKAGE_VERSION_MINOR "${PROJECT_VERSION_MINOR}")
set(CPACK_PACKAGE_VERSION_PATCH "${PROJECT_VERSION_PATCH}")

set(CPACK_PACKAGE_CONTACT jacksonrr3@yandex.ru)

set(CPACK_GENERATOR DEB)

set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS ON)

include (CPack)
