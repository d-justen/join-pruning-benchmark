FROM ubuntu:24.04 AS build
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /src
COPY CMakeLists.txt ./
COPY TPCDSkew/*.c TPCDSkew/*.h TPCDSkew/dists.dss ./TPCDSkew/
RUN cmake -S . -B /build -DCMAKE_BUILD_TYPE=Release \
    && cmake --build /build --parallel 2 \
    && cmake --install /build --prefix /opt/tpcdskew

FROM ubuntu:24.04
COPY --from=build /opt/tpcdskew /opt/tpcdskew
ENV DSS_CONFIG=/opt/tpcdskew/share/tpcdskew
WORKDIR /data
ENTRYPOINT ["/opt/tpcdskew/bin/dbgen"]
CMD ["-h"]
