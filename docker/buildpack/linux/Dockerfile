FROM quay.io/pypa/manylinux_2_28_x86_64 AS build

# Install Vulkan SDK
RUN curl -LO https://sdk.lunarg.com/sdk/download/1.4.309.0/linux/vulkansdk-linux-x86_64-1.4.309.0.tar.xz && \
    mkdir -p /opt/vulkan-sdk && \
    tar -xJf vulkansdk-linux-x86_64-1.4.309.0.tar.xz --strip-components=2 -C /opt/vulkan-sdk

# Install OpenSSL
RUN yum install -y perl-IPC-Cmd perl-Time-Piece && \
    curl -LO https://www.openssl.org/source/openssl-3.6.0.tar.gz && \
    tar -xzf openssl-3.6.0.tar.gz && \
    cd openssl-3.6.0 && \
    ./config --prefix=/opt/openssl --openssldir=/opt/openssl no-shared no-tests && \
    make -j$(nproc) && \
    make install_sw

# Install Reference-LAPACK
RUN yum install -y gcc-gfortran && \
    curl -LO https://github.com/Reference-LAPACK/lapack/archive/refs/tags/v3.12.1.tar.gz && \
    tar -xzf v3.12.1.tar.gz && \
    cd lapack-3.12.1 && \
    cp make.inc.example make.inc && \
    sed -i 's/librefblas/libblas/g' make.inc && \
    cmake -B./build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/lapack -DCMAKE_INSTALL_LIBDIR=lib . && \
    cmake --build ./build -t install --parallel


FROM quay.io/pypa/manylinux_2_28_x86_64

COPY --from=build /opt/vulkan-sdk /opt/vulkan-sdk
COPY --from=build /opt/openssl /opt/openssl
COPY --from=build /opt/lapack /opt/lapack
