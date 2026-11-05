' SignGesture AI - Silent Launcher
' Runs the dashboard directly without showing a background console window.

Set objFSO = CreateObject("Scripting.FileSystemObject")
Set objShell = CreateObject("WScript.Shell")

scriptDir = objFSO.GetParentFolderName(WScript.ScriptFullName)
objShell.CurrentDirectory = scriptDir

localAppData = objShell.ExpandEnvironmentStrings("%LOCALAPPDATA%")
pywPath = localAppData & "\Programs\Python\Python311\pythonw.exe"

If objFSO.FileExists(pywPath) Then
    objShell.Run """" & pywPath & """ """ & scriptDir & "\realtime_detector.py""", 0, False
Else
    ' Fallback to standard bat runner
    objShell.Run """" & scriptDir & "\RUN_SIGN_GESTURE.bat""", 1, False
End If
