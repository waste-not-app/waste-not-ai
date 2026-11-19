@echo off
setlocal
if not exist "%~dp0artifacts\training\broad-waste-v3-batch2-yolo11s\weights\best.pt" (
    echo v3 checkpoint not found. See docs\archive\BROAD_WASTE_V3_RESULTS.md.
    exit /b 1
)
if not exist "%~dp0artifacts\models\yolo11n.pt" (
    echo General object model not found. Run fetch-assets as described in README.md.
    exit /b 1
)
call "%~dp0demo-camera.bat" --device cpu --input-shape square --model "%~dp0artifacts\training\broad-waste-v3-batch2-yolo11s\weights\best.pt" --general-model "%~dp0artifacts\models\yolo11n.pt" %*
exit /b %ERRORLEVEL%
