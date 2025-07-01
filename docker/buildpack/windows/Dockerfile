# escape=`

### ─────────────────────────────
### Stage 1: Build environment
### ─────────────────────────────
FROM mcr.microsoft.com/windows/servercore:ltsc2022 AS builder

SHELL ["powershell", "-Command", "$ErrorActionPreference = 'Stop'; $ProgressPreference = 'SilentlyContinue';"]

# Install Git portable
RUN Invoke-WebRequest `
        -Uri "https://github.com/git-for-windows/git/releases/download/v2.50.0.windows.2/PortableGit-2.50.0.2-64-bit.7z.exe" `
        -OutFile "PortableGit.7z.exe"; `
    Start-Process -FilePath "PortableGit.7z.exe" -ArgumentList '-y', '-oC:\git' -Wait
RUN ls C:\; `
    ls C:\git; `
    ls C:\git\cmd
RUN setx /M PATH "%PATH%;C:\git\cmd"

# Install CMake portable
RUN Invoke-WebRequest `
        -Uri "https://github.com/Kitware/CMake/releases/download/v4.0.3/cmake-4.0.3-windows-x86_64.zip" `
        -OutFile "cmake.zip"; `
    Expand-Archive -Path "cmake.zip" -DestinationPath "C:\\cmake" -Force
RUN setx /M PATH "%PATH%;C:\cmake\cmake-4.0.3-windows-x86_64\bin"

# Clone and bootstrap vcpkg
RUN setx /M VCPKG_ROOT "C:\vcpkg"
RUN git clone https://github.com/microsoft/vcpkg.git C:\vcpkg; `
    cd C:\vcpkg; `
    .\bootstrap-vcpkg.bat
RUN setx /M VCPKG_DEFAULT_TRIPLET "x64-windows-static"
RUN setx /M PATH "%PATH%;C:\vcpkg\installed\x64-windows-static\bin"

# Install OpenBLAS and LAPACK
RUN & "C:\\vcpkg\\vcpkg.exe" install openblas:x64-windows-static lapack-reference:x64-windows-static

# Install OpenSSL (to C:\OpenSSL)
RUN Invoke-WebRequest `
        -Uri "https://slproweb.com/download/Win64OpenSSL-3_5_0.exe" `
        -OutFile "openssl-installer.exe"; `
    Start-Process -FilePath "openssl-installer.exe" `
        -ArgumentList '/silent', '/verysilent', '/sp-', '/SUPPRESSMSGBOXES', '/DIR=C:\OpenSSL' -Wait

# Install Vulkan SDK
RUN Invoke-WebRequest `
        -Uri "https://sdk.lunarg.com/sdk/download/1.3.283.0/windows/vulkan-sdk.exe" `
        -OutFile "vulkan-sdk.exe"; `
    Start-Process -FilePath "vulkan-sdk.exe" `
        -ArgumentList '--accept-licenses', '--default-answer', '--confirm-command', 'install', '--root', 'C:\VulkanSDK' -Wait


### ─────────────────────────────
### Stage 2: Final runtime image
### ─────────────────────────────
FROM mcr.microsoft.com/windows/servercore:ltsc2022

SHELL ["powershell", "-Command", "$ErrorActionPreference = 'Stop'; $ProgressPreference = 'SilentlyContinue';"]

# Set environment variables
RUN setx /M BLAS_ROOT "C:\vcpkg\installed\x64-windows-static"
RUN setx /M LAPACK_ROOT "C:\vcpkg\installed\x64-windows-static"
RUN setx /M VULKAN_SDK "C:\VulkanSDK"
RUN setx /M OPENSSL_ROOT_DIR="C:\OpenSSL"
RUN setx /M PATH "%PATH%;C:\vcpkg\installed\x64-windows-static\bin;C:\VulkanSDK\Bin;C:\VulkanSDK\Lib;C:\OpenSSL\bin"

# Copy runtime dependencies only
COPY --from=builder C:\vcpkg\ C:\vcpkg\
COPY --from=builder C:\VulkanSDK\ C:\VulkanSDK\
COPY --from=builder C:\OpenSSL\ C:\OpenSSL\
