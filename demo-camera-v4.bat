@echo off
setlocal
if not exist "%~dp0artifacts\training\broad-waste-v4-focus-yolo11s\weights\best.pt" (
    echo Checkpoint v4 belum tersedia. Ikuti docs\FOCUS_V4_RESULTS.md.
    exit /b 1
)
if not exist "%~dp0artifacts\models\yolo11s.pt" (
    echo Model benda umum belum tersedia. Jalankan fetch-assets sesuai README.md.
    exit /b 1
)
echo Eksperimen v4 + COCO11s. Semua dugaan memerlukan konfirmasi.
call "%~dp0demo-camera.bat" --port 8100 --device cpu --input-shape square --model "%~dp0artifacts\training\broad-waste-v4-focus-yolo11s\weights\best.pt" --general-model "%~dp0artifacts\models\yolo11s.pt" %*
exit /b %ERRORLEVEL%
