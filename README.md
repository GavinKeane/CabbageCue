### Setup:
Run setup_env.ps1 in Powershell

### Usage:
`run_aligner.bat --video "D:\Depos\Session1.mp4" "D:\Depos\Session2.mp4" "D:\Depos\Session*.mp4 --transcript "D:\Depos\Transcript.txt" --out "D:\Output\Outputs" --first-name "John" --last-name "Smith" --date "12/31/2025"`

The `--video` argument accepts 1 or more video files. It also supports wildcard characters.\
The `--transcript` argument accepts only 1 txt file.\
The `--out` argument points to a folder. An mdb and a SQLite file will be created in this location.

### Requirements:
- NVIDIA 30, 40, or 50 series
- 32 GB RAM