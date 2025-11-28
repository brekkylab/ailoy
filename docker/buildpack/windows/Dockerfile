# escape=`

FROM mcr.microsoft.com/windows/servercore:ltsc2022

SHELL ["powershell", "-Command", "$ErrorActionPreference = 'Stop'; $ProgressPreference = 'SilentlyContinue';"]

# Download all installers in parallel first
RUN $downloads = @( `
        @{Url='https://aka.ms/vs/17/release/vs_buildtools.exe'; Out='C:\vs_buildtools.exe'}, `
        @{Url='https://github.com/git-for-windows/git/releases/download/v2.50.0.windows.2/PortableGit-2.50.0.2-64-bit.7z.exe'; Out='C:\PortableGit.7z.exe'}, `
        @{Url='https://slproweb.com/download/Win64OpenSSL_Light-3_6_0.exe'; Out='C:\openssl-installer.exe'}, `
        @{Url='https://sdk.lunarg.com/sdk/download/1.4.309.0/windows/vulkan-sdk.exe'; Out='C:\vulkan-sdk.exe'}, `
        @{Url='https://win.rustup.rs/x86_64'; Out='C:\rustup-init.exe'} `
    ); `
    $jobs = @(); `
    foreach ($d in $downloads) { `
        Write-Host "Starting download: $($d.Url)"; `
        $jobs += Start-Job -ScriptBlock { `
            param($url,$out); `
            Invoke-WebRequest -Uri $url -OutFile $out; `
        } -ArgumentList $d.Url, $d.Out; `
    }; `
    $jobs | Wait-Job | Receive-Job; `
    $jobs | Remove-Job; `
    Write-Host "All downloads completed."

# Install Visual Studio Build Tools with minimal components
RUN Start-Process -FilePath "vs_buildtools.exe" `
        -ArgumentList `
            "--quiet", "--wait", "--norestart", "--nocache", `
            "--installPath", "C:\BuildTools", `
            "--add", "Microsoft.VisualStudio.Workload.VCTools", `
            "--includeRecommended" `
        -Wait; `
    Remove-Item "vs_buildtools.exe"; `
    $msvcPath = Get-ChildItem -Path "C:\BuildTools\VC\Tools\MSVC" -Directory | Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty FullName; `
    setx /M PATH \"$msvcPath\bin\Hostx64\x64;C:\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin;$([System.Environment]::GetEnvironmentVariable('PATH', [System.EnvironmentVariableTarget]::Machine))\"; `
    setx /M LD_LIBRARY_PATH \"$msvcPath\lib\x64;$([System.Environment]::GetEnvironmentVariable('LD_LIBRARY_PATH', [System.EnvironmentVariableTarget]::Machine))\"

# Install Git portable
RUN Start-Process -FilePath "PortableGit.7z.exe" -ArgumentList '-y', '-oC:\git' -Wait; `
    Remove-Item "PortableGit.7z.exe"; `
    setx /M PATH \"C:\git\cmd;$([System.Environment]::GetEnvironmentVariable('PATH', [System.EnvironmentVariableTarget]::Machine))\"

# Clone and bootstrap vcpkg with shallow clone
RUN git clone --depth 1 https://github.com/microsoft/vcpkg.git C:\vcpkg; `
    cd C:\vcpkg; `
    .\bootstrap-vcpkg.bat -disableMetrics

# Install lapack-reference with binary caching
COPY vcpkg.json ./
RUN $env:VCPKG_BINARY_SOURCES = 'clear;x-azblob,https://vcpkg.blob.core.windows.net/public-cache,read'; `
    C:\vcpkg\vcpkg.exe install --triplet x64-windows-static-release; `
    Remove-Item vcpkg.json -Force; `
    setx /M PATH \"C:\vcpkg_installed\x64-windows-static-release\bin;$([System.Environment]::GetEnvironmentVariable('PATH', [System.EnvironmentVariableTarget]::Machine))\"; `
    setx /M LD_LIBRARY_PATH \"C:\vcpkg_installed\x64-windows-static-release\lib;$([System.Environment]::GetEnvironmentVariable('LD_LIBRARY_PATH', [System.EnvironmentVariableTarget]::Machine))\"

# Install OpenSSL, Vulkan SDK, and Rust in parallel
RUN $installers = @( `
        @{Exe='C:\openssl-installer.exe'; Args='/silent /verysilent /sp- /SUPPRESSMSGBOXES /DIR=C:\OpenSSL'}, `
        @{Exe='C:\vulkan-sdk.exe'; Args='--accept-licenses --default-answer --confirm-command install --root C:\VulkanSDK'}, `
        @{Exe='C:\rustup-init.exe'; Args='-y --default-toolchain 1.90.0 --profile minimal'} `
    ); `
    $jobs = @(); `
    foreach ($i in $installers) { `
        Write-Host "Starting install: $($i.Exe)"; `
        $jobs += Start-Job -ScriptBlock { `
            param($exe, $argString); `
            $argList = $argString -split ' '; `
            Start-Process -FilePath $exe -ArgumentList $argList -Wait; `
        } -ArgumentList $i.Exe, $i.Args; `
    }; `
    $jobs | Wait-Job | Receive-Job; `
    $jobs | Remove-Job; `
    Remove-Item "C:\openssl-installer.exe", "C:\vulkan-sdk.exe", "C:\rustup-init.exe" -Force; `
    setx /M VULKAN_SDK \"C:\VulkanSDK\"; `
    setx /M PATH \"C:\VulkanSDK\Bin;C:\Users\ContainerAdministrator\.cargo\bin;$([System.Environment]::GetEnvironmentVariable('PATH', [System.EnvironmentVariableTarget]::Machine))\"; `
    setx /M LD_LIBRARY_PATH \"C:\VulkanSDK\Lib;$([System.Environment]::GetEnvironmentVariable('LD_LIBRARY_PATH', [System.EnvironmentVariableTarget]::Machine))\"
