cmake_minimum_required(VERSION 3.15 FATAL_ERROR)

project(DBoW2 VERSION 0.1.0 LANGUAGES CXX)

include(GNUInstallDirs)

set(BUILD_AS_MAIN_PROJECT ON)
if(NOT CMAKE_SOURCE_DIR STREQUAL PROJECT_SOURCE_DIR)
  set(BUILD_AS_MAIN_PROJECT OFF)
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING "Build type" FORCE)
endif()

find_package(OpenCV 4.2 REQUIRED)

set(HDRS
    include/DBoW2/BowVector.h
    include/DBoW2/DescriptorPolicy.h
    include/DBoW2/FBrief.h
    include/DBoW2/QueryResults.h
    include/DBoW2/TemplatedDatabase.h
    include/DBoW2/FORB.h
    include/DBoW2/DBoW2.h
    include/DBoW2/FSIFT.h
    include/DBoW2/FeatureVector.h
    include/DBoW2/FSurf64.h
    include/DBoW2/ScoringObject.h
    include/DBoW2/TemplatedVocabulary.h)
set(SRCS
    src/BowVector.cpp
    src/FBrief.cpp
    src/FORB.cpp
    src/FSIFT.cpp
    src/FeatureVector.cpp
    src/FSurf64.cpp
    src/QueryResults.cpp
    src/ScoringObject.cpp)

if(BUILD_SHARED_LIBS)
  set(DBOW2_LIBRARY_KIND SHARED)
else()
  set(DBOW2_LIBRARY_KIND STATIC)
endif()

add_library(${PROJECT_NAME} ${DBOW2_LIBRARY_KIND} ${SRCS} ${HDRS})
set_target_properties(${PROJECT_NAME} PROPERTIES
                      CXX_STANDARD 20
                      CXX_STANDARD_REQUIRED ON
                      CXX_EXTENSIONS OFF
                      POSITION_INDEPENDENT_CODE ON)
if(NOT TARGET ${PROJECT_NAME}::${PROJECT_NAME})
  add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})
endif()
target_include_directories(${PROJECT_NAME} PUBLIC
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                           $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/DBoW2>
                           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_link_libraries(${PROJECT_NAME} PUBLIC ${OpenCV_LIBS})
target_include_directories(${PROJECT_NAME} PUBLIC ${OpenCV_INCLUDE_DIRS})

if(BUILD_AS_MAIN_PROJECT)
  add_executable(demo demo/demo.cpp)
  target_link_libraries(demo PRIVATE ${PROJECT_NAME} ${OpenCV_LIBS})
  set_target_properties(demo PROPERTIES CXX_STANDARD 20 CXX_STANDARD_REQUIRED ON)
  file(COPY demo/images DESTINATION ${CMAKE_BINARY_DIR})
endif()

include(CMakePackageConfigHelpers)

set(EXPORT_TARGET_DEPS OpenCV)

install(TARGETS ${PROJECT_NAME}
        EXPORT ${PROJECT_NAME}
        ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
        LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
        INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(DIRECTORY include/DBoW2 DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
install(FILES LICENSE.txt DESTINATION ${CMAKE_INSTALL_DATADIR}/${PROJECT_NAME}/licenses)

install(EXPORT ${PROJECT_NAME}
        FILE ${PROJECT_NAME}Target.cmake
        NAMESPACE ${PROJECT_NAME}::
        DESTINATION lib/cmake/${PROJECT_NAME})

write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY AnyNewerVersion)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/src/${PROJECT_NAME}Config.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    INSTALL_DESTINATION lib/cmake/${PROJECT_NAME})

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}Config.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/${PROJECT_NAME}ConfigVersion.cmake"
    DESTINATION lib/cmake/${PROJECT_NAME})

export(EXPORT ${PROJECT_NAME}
       FILE "${CMAKE_CURRENT_BINARY_DIR}/cmake/${PROJECT_NAME}Target.cmake")
