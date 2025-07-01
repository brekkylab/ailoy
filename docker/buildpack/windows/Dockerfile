# escape=`

### ─────────────────────────────
### Stage 1: Build environment
### ─────────────────────────────
FROM mcr.microsoft.com/windows/servercore:ltsc2022 AS builder

SHELL ["powershell", "-Command", "$ErrorActionPreference = 'Stop'; $ProgressPreference = 'SilentlyContinue';"]

# Install Chocolatey and build tools
RUN Set-ExecutionPolicy Bypass -Scope Process -Force; `
    iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1')); `
    choco install -y git cmake
ENV PATH="C:\\ProgramData\\chocolatey\\bin;%PATH%"

# Clone and bootstrap vcpkg
ENV VCPKG_ROOT="C:\\vcpkg" `
    VCPKG_DEFAULT_TRIPLET="x64-windows-static" `
    PATH="C:\\vcpkg\\installed\\x64-windows-static\\bin;%PATH%"
RUN git clone https://github.com/microsoft/vcpkg.git $env:VCPKG_ROOT; `
    cd $env:VCPKG_ROOT; `
    .\bootstrap-vcpkg.bat

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
ENV VCPKG_ROOT="C:\\vcpkg" `
    VCPKG_DEFAULT_TRIPLET="x64-windows-static" `
    BLAS_ROOT="C:\\vcpkg\\installed\\x64-windows-static" `
    LAPACK_ROOT="C:\\vcpkg\\installed\\x64-windows-static" `
    VULKAN_SDK="C:\\VulkanSDK" `
    OPENSSL_ROOT_DIR="C:\\OpenSSL" `
    PATH="C:\\vcpkg\\installed\\x64-windows-static\\bin;C:\\VulkanSDK\\Bin;C:\\VulkanSDK\\Lib;C:\\OpenSSL\\bin;%PATH%"

# Copy runtime dependencies only
COPY --from=builder C:\vcpkg\ C:\vcpkg\
COPY --from=builder C:\VulkanSDK\ C:\VulkanSDK\
COPY --from=builder C:\OpenSSL\ C:\OpenSSL\
