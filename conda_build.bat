IF "%CONDA_BUILD%" == "" (
    call C:\condatmp\miniconda3\conda-bld\cuda-nvtx_1661472073765\work\build_env_setup.bat
)
REM ===== end generated header =====
if not exist %PREFIX% mkdir %PREFIX%

rem Directories...
for /D %%i in (.\*) do (
    robocopy /E %%i %PREFIX%\%%i
)

rem Files...
for %%i in (.\*) do (
    if not %%~ni==build_env_setup.bat (
    if not %%~ni==conda_build.bat (
    if not %%~ni==metadata_conda_debug.yaml (
        xcopy %%i %PREFIX%
    )
    )
    )
)
