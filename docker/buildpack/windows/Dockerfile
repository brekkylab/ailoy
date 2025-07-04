# escape=`

FROM mcr.microsoft.com/windows/servercore:ltsc2022

SHELL ["powershell", "-Command", "$ErrorActionPreference = 'Stop'; $ProgressPreference = 'SilentlyContinue';"]

# Install Visual Studio Build Tools with C++ support
RUN Invoke-WebRequest `
        -Uri "https://aka.ms/vs/17/release/vs_buildtools.exe" `
        -OutFile "vs_buildtools.exe"; `
    Start-Process -FilePath "vs_buildtools.exe" `
        -ArgumentList `
            "--quiet", "--wait", "--norestart", "--nocache", `
            "--installPath", "C:\BuildTools", `
            "--add", "Microsoft.VisualStudio.Workload.VCTools", `
            "--includeRecommended" `
        -Wait; `
    Remove-Item "vs_buildtools.exe"; `
    $msvcPath = Get-ChildItem -Path "C:\BuildTools\VC\Tools\MSVC" -Directory | Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty FullName; `
    setx /M PATH \"$msvcPath\bin\Hostx64\x64;C:\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;$([System.Environment]::GetEnvironmentVariable('PATH', [System.EnvironmentVariableTarget]::Machine))\"

# Install Git portable
RUN Invoke-WebRequest `
        -Uri "https://github.com/git-for-windows/git/releases/download/v2.50.0.windows.2/PortableGit-2.50.0.2-64-bit.7z.exe" `
        -OutFile "PortableGit.7z.exe"; `
    Start-Process -FilePath "PortableGit.7z.exe" -ArgumentList '-y', '-oC:\git' -Wait; `
    Remove-Item "PortableGit.7z.exe"; `
    setx /M PATH \"C:\git\cmd;$([System.Environment]::GetEnvironmentVariable('PATH', [System.EnvironmentVariableTarget]::Machine))\"

# Clone and bootstrap vcpkg
RUN git clone https://github.com/microsoft/vcpkg.git C:\vcpkg; `
    cd C:\vcpkg; `
    .\bootstrap-vcpkg.bat -disableMetrics

# Install OpenBLAS and LAPACK
COPY vcpkg.json ./
RUN "C:\\vcpkg\\vcpkg.exe" install --triplet x64-windows-static-release; `
    rm -fo vcpkg.json; `
    setx /M BLAS_ROOT \"C:\vcpkg_installed\x64-windows-static-release\"; `
    setx /M LAPACK_ROOT \"C:\vcpkg_installed\x64-windows-static-release\"; `
    setx /M PATH \"C:\vcpkg_installed\x64-windows-static-release\bin;$([System.Environment]::GetEnvironmentVariable('PATH', [System.EnvironmentVariableTarget]::Machine))\"

# Install OpenSSL
RUN Invoke-WebRequest `
        -Uri "https://slproweb.com/download/Win64OpenSSL-3_5_1.exe" `
        -OutFile "openssl-installer.exe"; `
    Start-Process -FilePath "openssl-installer.exe" `
        -ArgumentList '/silent', '/verysilent', '/sp-', '/SUPPRESSMSGBOXES', '/DIR=C:\OpenSSL' -Wait; `
    Remove-Item "openssl-installer.exe"

# Install Vulkan SDK
RUN Invoke-WebRequest `
        -Uri "https://sdk.lunarg.com/sdk/download/1.3.283.0/windows/vulkan-sdk.exe" `
        -OutFile "vulkan-sdk.exe"; `
    Start-Process -FilePath "vulkan-sdk.exe" `
        -ArgumentList '--accept-licenses', '--default-answer', '--confirm-command', 'install', '--root', 'C:\VulkanSDK' -Wait; `
    Remove-Item "vulkan-sdk.exe"; `
    setx -M VULKAN_SDK \"C:\VulkanSDK\"

# Install Rust
RUN Invoke-WebRequest `
        -Uri "https://win.rustup.rs/x86_64" `
        -OutFile "rustup-init.exe"; `
    Start-Process -FilePath "rustup-init.exe" `
        -ArgumentList '-y', '--default-toolchain', '1.88.0', '--profile', 'default' -Wait -NoNewWindow; `
    Remove-Item "rustup-init.exe"; `
    setx /M PATH \"$env:USERPROFILE\.cargo\bin;$([System.Environment]::GetEnvironmentVariable('PATH', [System.EnvironmentVariableTarget]::Machine))\"
