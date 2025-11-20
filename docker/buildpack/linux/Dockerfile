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

# Install OpenBLAS
RUN yum install -y gcc-gfortran && \
    curl -LO https://github.com/OpenMathLib/OpenBLAS/releases/download/v0.3.29/OpenBLAS-0.3.29.tar.gz && \
    tar -xzf OpenBLAS-0.3.29.tar.gz && \
    cd OpenBLAS-0.3.29 && \
    make -j$(nproc) NO_SHARED=1 USE_OPENMP=0 && \
    make PREFIX=/opt/openblas NO_SHARED=1 USE_OPENMP=0 install


FROM quay.io/pypa/manylinux_2_28_x86_64

COPY --from=build /opt/vulkan-sdk /opt/vulkan-sdk
COPY --from=build /opt/openssl /opt/openssl
COPY --from=build /opt/openblas /opt/openblas
