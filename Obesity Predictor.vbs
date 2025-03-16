Set WshShell = CreateObject("WScript.Shell") 
WshShell.Run Chr(34) & WshShell.CurrentDirectory & "\scripts\app.bat" & Chr(34), 0
Set WshShell = Nothing
