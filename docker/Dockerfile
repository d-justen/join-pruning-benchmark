# syntax=docker/dockerfile:1.7
ARG BASE_IMAGE=ubuntu:24.04@sha256:68ee22004a9109b44973313b02ff7b1de250f1de0f5ecb628d73f579aaff33ac
FROM ${BASE_IMAGE} AS toolchain
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates clang cmake ninja-build make git curl gcc g++ \
    python3 python3-venv pkg-config zip unzip tar autoconf automake libtool \
    libssl-dev zlib1g-dev && rm -rf /var/lib/apt/lists/*
ENV CC=clang CXX=clang++
ARG BUILD_JOBS=2

FROM toolchain AS generators
COPY TPCDSkew /src/tpcdskew
COPY --from=ssb / /src/ssb
RUN cmake -S /src/tpcdskew -B /build/tpcdskew -DCMAKE_INSTALL_PREFIX=/opt/tpcdskew \
 && cmake --build /build/tpcdskew -j ${BUILD_JOBS} \
 && cmake --install /build/tpcdskew \
 && cmake -S /src/ssb -B /build/ssb -DWORKLOAD=SSB \
    -DCSV_OUTPUT_FORMAT=OFF -DEOL_HANDLING=OFF -DYMD_DASH_DATE=OFF \
 && cmake --build /build/ssb --target dbgen -j ${BUILD_JOBS} \
 && mkdir -p /opt/ssb/bin /opt/ssb/share \
 && cp /build/ssb/dbgen /opt/ssb/bin/ \
 && cp /src/ssb/dists.dss /src/ssb/doc/ssb.ddl /opt/ssb/share/

FROM toolchain AS prf
COPY duckdb /src/duckdb
RUN cmake -S /src/duckdb -B /build/prf -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_UNITTESTS=OFF -DENABLE_SANITIZER=OFF \
    -DENABLE_UBSAN=OFF -DNATIVE_ARCH=OFF -DBUILD_DIVA_BENCHMARK=ON \
    -DSKIP_EXTENSIONS=parquet \
 && cmake --build /build/prf --target shell filter_benchmark range_benchmark \
    join_pruning_benchmark filter_benchmark_sve2 range_benchmark_sve2 \
    join_pruning_benchmark_sve2 range_filter_adapter_tests range_filter_adapter_tests_sve2 \
    diva_bit_tests diva_bit_tests_sve2 -j ${BUILD_JOBS} \
 && timeout 120 /build/prf/tools/diva_bit_tests \
 && timeout 120 /build/prf/tools/range_filter_adapter_tests

FROM toolchain AS rpt
COPY dynamic-predicate-transfer /src/rpt
RUN cmake -S /src/rpt -B /build/rpt -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DBUILD_UNITTESTS=OFF -DENABLE_SANITIZER=OFF \
    -DENABLE_UBSAN=OFF -DNATIVE_ARCH=OFF -DSKIP_EXTENSIONS=parquet \
 && cmake --build /build/rpt --target shell -j ${BUILD_JOBS}

# Explicit opt-in target. Both extensions and their source patches are pinned
# by the checkout's .github/config/extensions files.
FROM toolchain AS prf-extensions
COPY duckdb /src/duckdb
WORKDIR /src/duckdb
ENV GEN=ninja BUILD_UNITTESTS=0 DUCKDB_NEW_EXTENSION_BUILD=1 \
    VCPKG_TOOLCHAIN_PATH=/src/duckdb/vcpkg/scripts/buildsystems/vcpkg.cmake \
    EXTENSION_CONFIGS=/src/duckdb/.github/config/extensions/ducklake.cmake;/src/duckdb/.github/config/extensions/httpfs.cmake;/src/duckdb/.github/config/extensions/aws.cmake;/src/duckdb/.github/config/extensions/json.cmake
RUN make setup-vcpkg \
 && test "$(git -C vcpkg rev-parse HEAD)" = 84bab45d415d22042bd0b9081aea57f362da3f35 \
 && make sync_out_of_tree_extensions \
 && VCPKG_TOOLCHAIN_PATH=/src/duckdb/vcpkg/scripts/buildsystems/vcpkg.cmake \
    CMAKE_BUILD_PARALLEL_LEVEL=${BUILD_JOBS} \
    make release EXTRA_CMAKE_VARIABLES="-DBUILD_UNITTESTS=OFF -DBUILD_DIVA_BENCHMARK=OFF -DNATIVE_ARCH=OFF" \
 && build/release/duckdb :memory: -c "LOAD ducklake; LOAD httpfs; LOAD aws; LOAD json; SELECT httpfs_reset_transfer_stats(); SELECT to_json(list(t)) FROM httpfs_transfer_stats() t;"

FROM ${BASE_IMAGE} AS runtime-base
RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates libstdc++6 libssl3t64 zlib1g python3 python3-venv curl \
 && rm -rf /var/lib/apt/lists/*
COPY requirements.txt /tmp/requirements.txt
RUN python3 -m venv /opt/venv \
 && /opt/venv/bin/pip install --no-cache-dir -r /tmp/requirements.txt
ENV PATH=/opt/venv/bin:/opt/prf:/opt/rpt:$PATH \
    PYTHONUNBUFFERED=1 MPLBACKEND=Agg MPLCONFIGDIR=/tmp/matplotlib \
    PYTHONDONTWRITEBYTECODE=1
COPY --from=generators /opt/tpcdskew /opt/tpcdskew
COPY --from=generators /opt/ssb /opt/ssb
COPY --from=rpt /build/rpt/duckdb /opt/rpt/duckdb
COPY --from=prf /build/prf/tools/filter_benchmark /build/prf/tools/range_benchmark /build/prf/tools/join_pruning_benchmark /opt/prf/
COPY --from=prf /build/prf/tools/filter_benchmark_sve2 /build/prf/tools/range_benchmark_sve2 /build/prf/tools/join_pruning_benchmark_sve2 /opt/prf/
COPY --from=prf /build/prf/tools/diva_bit_tests* /build/prf/tools/range_filter_adapter_tests* /opt/prf/
WORKDIR /app
COPY benchmarking /app/benchmarking
COPY paper/figures /app/paper/figures
COPY paper/figure_specs.py paper/__init__.py /app/paper/
COPY configs /app/configs
COPY external_benchmarks /app/external_benchmarks
COPY docker /app/docker
ENTRYPOINT ["python", "/app/docker/entrypoint.py"]
CMD ["--help"]

FROM runtime-base AS native
COPY --from=prf /build/prf/duckdb /opt/prf/duckdb

FROM runtime-base AS extensions
COPY --from=prf-extensions /src/duckdb/build/release/duckdb /opt/prf/duckdb

# The publication image retains each existing executable and its build flags.
# Configs select the extension-enabled CLI only for the DuckLake experiment group.
FROM native AS paper
COPY --from=prf-extensions /src/duckdb/build/release/duckdb /opt/prf/duckdb-extensions
