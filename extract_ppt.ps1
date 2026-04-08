$pptPath = "c:\Users\ASUS\xai-workbench\XAI_Dashboard_Review3 (1) (1).pptx"
$outputPath = "c:\Users\ASUS\xai-workbench\ppt_content.txt"

# Create a PowerPoint application object
try {
    $pptApp = New-Object -ComObject PowerPoint.Application
    $presentation = $pptApp.Presentations.Open($pptPath, [Microsoft.Office.Core.MsoTriState]::msoTrue, [Microsoft.Office.Core.MsoTriState]::msoFalse, [Microsoft.Office.Core.MsoTriState]::msoFalse)

    $content = ""
    foreach ($slide in $presentation.Slides) {
        $content += "`n--- Slide $($slide.SlideNumber) ---`n"
        foreach ($shape in $slide.Shapes) {
            if ($shape.HasTextFrame -eq -1 -and $shape.TextFrame.HasText -eq -1) {
                $content += $shape.TextFrame.TextRange.Text + "`n"
            }
        }
    }

    $content | Out-File -FilePath $outputPath -Encoding utf8
    $presentation.Close()
    $pptApp.Quit()
    Write-Host "Content extracted to $outputPath"
} catch {
    Write-Error "Failed to extract content: $_"
    if ($pptApp) { $pptApp.Quit() }
}
