@echo off
setlocal
if not exist "%~dp0artifacts\training\broad-waste-v4-focus-yolo11s\weights\best.pt" (
    echo v4 checkpoint not found. See docs\archive\FOCUS_V4_RESULTS.md.
    exit /b 1
)
if not exist "%~dp0artifacts\models\yolo11s.pt" (
    echo General object model not found. Run fetch-assets as described in README.md.
    exit /b 1
)
echo Experimental v4 + COCO11s model. Every detection still needs confirmation.
call "%~dp0demo-camera.bat" --port 8100 --device cpu --input-shape square --model "%~dp0artifacts\training\broad-waste-v4-focus-yolo11s\weights\best.pt" --general-model "%~dp0artifacts\models\yolo11s.pt" %*
exit /b %ERRORLEVEL%
