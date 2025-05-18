@echo off
REM SAM2.1 checkpoints downloader for Windows

where curl >nul 2>nul
if errorlevel 1 (
    echo Please install curl to download the checkpoints.
    exit /b 1
)

set SAM2p1_BASE_URL=https://dl.fbaipublicfiles.com/segment_anything_2/092824
set sam2p1_hiera_t_url=%SAM2p1_BASE_URL%/sam2.1_hiera_tiny.pt
set sam2p1_hiera_s_url=%SAM2p1_BASE_URL%/sam2.1_hiera_small.pt
set sam2p1_hiera_b_plus_url=%SAM2p1_BASE_URL%/sam2.1_hiera_base_plus.pt
set sam2p1_hiera_l_url=%SAM2p1_BASE_URL%/sam2.1_hiera_large.pt

echo Downloading sam2.1_hiera_tiny.pt checkpoint...
curl -L -O %sam2p1_hiera_t_url%
if errorlevel 1 (
    echo Failed to download checkpoint from %sam2p1_hiera_t_url%
    exit /b 1
)

echo Downloading sam2.1_hiera_small.pt checkpoint...
curl -L -O %sam2p1_hiera_s_url%
if errorlevel 1 (
    echo Failed to download checkpoint from %sam2p1_hiera_s_url%
    exit /b 1
)

echo Downloading sam2.1_hiera_base_plus.pt checkpoint...
curl -L -O %sam2p1_hiera_b_plus_url%
if errorlevel 1 (
    echo Failed to download checkpoint from %sam2p1_hiera_b_plus_url%
    exit /b 1
)

echo Downloading sam2.1_hiera_large.pt checkpoint...
curl -L -O %sam2p1_hiera_l_url%
if errorlevel 1 (
    echo Failed to download checkpoint from %sam2p1_hiera_l_url%
    exit /b 1
)

echo All checkpoints are downloaded successfully.