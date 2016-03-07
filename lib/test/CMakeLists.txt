#cmake_minimum_required (VERSION 2.8.11)
#project (object_detection_library)

# Create a library called "Hello" which includes the source file "hello.cxx".
# The extension is already found. Any number of sources could be listed here.
add_executable (object_detection_demo test.c)

# Make sure the compiler can find include files for our Hello library
# when other libraries or executables link to Hello
target_include_directories (object_detection_demo PUBLIC ../)
target_link_libraries (object_detection_demo LINK_PUBLIC object_detection)
