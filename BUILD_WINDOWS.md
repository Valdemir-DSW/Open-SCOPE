# Building OpenScope on Windows

Run `build.bat` from a normal Command Prompt.

The script:

1. Finds Python 3 (`py -3` or `python`).
2. Installs runtime and build dependencies.
3. Builds a standalone OpenScope distribution with Nuitka.
4. Searches for the Inno Setup compiler (`ISCC.exe`) in PATH, the common Program Files locations, LocalAppData, Windows uninstall registry entries, and finally by recursive search under Program Files.
5. If Inno Setup is found, compiles `installer\OpenScope.iss` and writes `release\OpenScope-Setup.exe`.

`resources\OpenScope.ico` is bundled and is used automatically by the source, Nuitka build and Inno Setup installer. `resources\OpenScope_logo.png` remains optional; when added later, it is used in the graph overlay and PNG exports.
