# escape=`

FROM mcr.microsoft.com/windows/servercore:ltsc2025

# Set shell to PowerShell
SHELL ["powershell", "-Command", "$ErrorActionPreference = 'Stop'; $ProgressPreference = 'SilentlyContinue';"]

# Install Chocolatey
RUN powershell -NoProfile -ExecutionPolicy Bypass -Command `
    Set-ExecutionPolicy Bypass -Scope Process -Force; `
    [System.Net.ServicePointManager]::SecurityProtocol = [System.Net.ServicePointManager]::SecurityProtocol -bor 3072; `
    iex ((New-Object System.Net.WebClient).DownloadString('https://community.chocolatey.org/install.ps1'))

# Install Git and CMake
RUN choco install -y git cmake

# Set environment variables
ENV VCPKG_ROOT=C:\vcpkg
ENV VCPKG_DEFAULT_TRIPLET=x64-windows
ENV PATH=%VCPKG_ROOT%;%PATH%

# Clond and bootstrap vcpkg
RUN git clone https://github.com/microsoft/vcpkg.git %VCPKG_ROOT% && `
    cd %VCPKG_ROOT% && `
    .\bootstrap-vcpkg.bat

# Install OpenBLAS
RUN %VCPKG_ROOT%\vcpkg install openblas:x64-windows-static lapack-reference:x64-windows-static
ENV BLAS_ROOT=$VCPKG_ROOT%\installed\x64-windows-static
ENV LAPACK_ROOT=%VCPKG_ROOT%\installed\x64-windows-static
ENV PATH=%VCPKG_ROOT%\installed\x64-windows-static\bin;%PATH%

# Install OpenSSL
RUN $url = "https://slproweb.com/download/Win64OpenSSL-3_5_0.exe" `
    $output = "openssl-installer.exe" `
    Invoke-WebRequest -Uri $url -OutFile $output `
    Start-Process -FilePath ".\openssl-installer.exe" -ArgumentList "/silent", "/verysilent", "/sp-", "/SUPPRESSMSGBOXES" -Wait

# Install Vulkan SDK
RUN $url = "https://sdk.lunarg.com/sdk/download/1.3.283.0/windows/vulkan-sdk.exe" `
    $output = "vulkan-sdk.exe" `
    Invoke-WebRequest -Uri $url -OutFile $output `
    Start-Process -FilePath ".\vulkan-sdk.exe" -ArgumentList "--accept-licenses", "--default-answer", "--confirm-command", "install", "--root", "C:\VulkanSDK" -Wait `
    echo "VULKAN_SDK=C:\VulkanSDK" >> $env:GITHUB_ENV `
    echo "PATH=C:\VulkanSDK\Bin;C:\VulkanSDK\Lib;$env:PATH" >> $env:GITHUB_ENV
