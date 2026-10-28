cmake_minimum_required(VERSION 3.16)
project(tpcdskew LANGUAGES C)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

add_executable(dbgen
  TPCDSkew/build.c
  TPCDSkew/driver.c
  TPCDSkew/bm_utils.c
  TPCDSkew/rnd.c
  TPCDSkew/print.c
  TPCDSkew/load_stub.c
  TPCDSkew/bcd2.c
  TPCDSkew/speed_seed.c
)
# This legacy source uses C89 function declarations. Do not enable
# SUPPORT_64BITS: that selects a different, unfinished key-printing path.
set_target_properties(dbgen PROPERTIES
  C_STANDARD 90
  C_STANDARD_REQUIRED YES
  C_EXTENSIONS YES
)
target_compile_definitions(dbgen PRIVATE _POSIX_C_SOURCE=200809L)
target_link_libraries(dbgen PRIVATE m)
configure_file(TPCDSkew/dists.dss dists.dss COPYONLY)

install(TARGETS dbgen RUNTIME DESTINATION bin)
install(FILES TPCDSkew/dists.dss DESTINATION share/tpcdskew)
