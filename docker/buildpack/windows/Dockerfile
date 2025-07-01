# escape=`

### ─────────────────────────────
### Stage 1: Build environment
### ─────────────────────────────
FROM mcr.microsoft.com/windows/servercore:ltsc2022 AS builder

SHELL ["powershell", "-Command", "$ErrorActionPreference = 'Stop'; $ProgressPreference = 'SilentlyContinue';"]

# Install Git portable
RUN $git_url = "https://github.com/git-for-windows/git/releases/download/v2.41.0.windows.1/PortableGit-2.41.0-64-bit.7z.exe"; `
    $git_installer = "PortableGit.7z.exe"; `
    Invoke-WebRequest -Uri $git_url -OutFile $git_installer; `
    Start-Process -FilePath $git_installer -ArgumentList "-y", "-oC:\git" -Wait
ENV PATH="C:\\git\\cmd;%PATH%"

# Install CMake portable (precompiled win64 zip)
RUN $cmake_url = "https://github.com/Kitware/CMake/releases/download/v4.0.3/cmake-4.0.3-windows-x86_64.zip"; `
    $cmake_zip = "cmake.zip"; `
    Invoke-WebRequest -Uri $cmake_url -OutFile $cmake_zip; `
    Expand-Archive -Path $cmake_zip -DestinationPath C:\cmake -Force
ENV PATH="C:\\cmake\\cmake-3.27.7-windows-x86_64\\bin;%PATH%"

# Clone and bootstrap vcpkg
ENV VCPKG_ROOT="C:\\vcpkg"
RUN git clone https://github.com/microsoft/vcpkg.git $env:VCPKG_ROOT; `
    cd $env:VCPKG_ROOT; `
    .\bootstrap-vcpkg.bat
ENV VCPKG_DEFAULT_TRIPLET="x64-windows-static"
ENV PATH="C:\\vcpkg\\installed\\x64-windows-static\\bin;%PATH%"

# Install OpenBLAS and LAPACK
RUN & "$env:VCPKG_ROOT\\vcpkg.exe" install openblas lapack-reference

# Install OpenSSL into C:\OpenSSL to avoid Program Files
RUN $url = "https://slproweb.com/download/Win64OpenSSL-3_5_0.exe"; `
    $installer = "openssl-installer.exe"; `
    Invoke-WebRequest -Uri $url -OutFile $installer; `
    Start-Process -FilePath $installer -ArgumentList "/silent", "/verysilent", "/sp-", "/SUPPRESSMSGBOXES", "/DIR=C:\OpenSSL" -Wait

# Install Vulkan SDK
RUN $url = "https://sdk.lunarg.com/sdk/download/1.3.283.0/windows/vulkan-sdk.exe"; `
    $installer = "vulkan-sdk.exe"; `
    Invoke-WebRequest -Uri $url -OutFile $installer; `
    Start-Process -FilePath $installer -ArgumentList "--accept-licenses", "--default-answer", "--confirm-command", "install", "--root", "C:\VulkanSDK" -Wait


### ─────────────────────────────
### Stage 2: Final runtime image
### ─────────────────────────────
FROM mcr.microsoft.com/windows/servercore:ltsc2022

SHELL ["powershell", "-Command", "$ErrorActionPreference = 'Stop'; $ProgressPreference = 'SilentlyContinue';"]

# Set environment variables
ENV BLAS_ROOT="C:\\vcpkg\\installed\\x64-windows-static"
ENV LAPACK_ROOT="C:\\vcpkg\\installed\\x64-windows-static"
ENV VULKAN_SDK="C:\\VulkanSDK"
ENV OPENSSL_ROOT_DIR="C:\\OpenSSL"
ENV PATH="C:\\vcpkg\\installed\\x64-windows-static\\bin;C:\\VulkanSDK\\Bin;C:\\VulkanSDK\\Lib;C:\\OpenSSL\\bin;%PATH%"

# Copy runtime dependencies only
COPY --from=builder C:\vcpkg\ C:\vcpkg\
COPY --from=builder C:\VulkanSDK\ C:\VulkanSDK\
COPY --from=builder C:\OpenSSL\ C:\OpenSSL\
