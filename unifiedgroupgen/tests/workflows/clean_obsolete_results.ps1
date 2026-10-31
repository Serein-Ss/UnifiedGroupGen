param([Parameter(Mandatory=$true)][string]$ProjectRoot)
$ErrorActionPreference = 'Stop'
$cleanupRoot = (Resolve-Path -LiteralPath $ProjectRoot).Path.TrimEnd('\')
$planFile = Join-Path $cleanupRoot 'docs/validation/cleanup_plan.json'
$reportFile = Join-Path $cleanupRoot 'docs/validation/cleanup_execution.json'
$cleanupPlan = Get-Content -LiteralPath $planFile -Raw | ConvertFrom-Json
$cleanupReport = [ordered]@{state='running'; project_root=$cleanupRoot; deleted_targets=@(); planned_bytes=$cleanupPlan.total_bytes}

try {
    foreach ($entry in $cleanupPlan.targets) {
        $cleanupTarget = [System.IO.Path]::GetFullPath((Join-Path $cleanupRoot $entry))
        if (-not $cleanupTarget.StartsWith($cleanupRoot+'\', [StringComparison]::OrdinalIgnoreCase)) {
            throw "Cleanup target escapes project: $cleanupTarget"
        }
        $relativeTarget = $cleanupTarget.Substring($cleanupRoot.Length+1)
        $cacheName = Split-Path -Leaf $cleanupTarget
        $allowed = ($relativeTarget -eq 'old_results') -or
                   $relativeTarget.StartsWith('intermediate\', [StringComparison]::OrdinalIgnoreCase) -or
                   ($relativeTarget -eq 'web\symmetry-lab\qa') -or
                   (($cacheName -in @('__pycache__','.pytest_cache','.ipynb_checkpoints')) -and
                    ($relativeTarget -match '^(unifiedgroupgen|scripts|assets|examples|web)\\'))
        if (-not $allowed) { throw "Unapproved cleanup scope: $relativeTarget" }
        if ($relativeTarget -match '^(outputs|data|\.git)(\\|$)') {
            throw "Protected results, data or repository metadata: $relativeTarget"
        }
        $cleanupReport['current'] = $relativeTarget
        $cleanupReport | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportFile -Encoding utf8
        if (Test-Path -LiteralPath $cleanupTarget) {
            Write-Output "[DELETE] $relativeTarget"
            $longTarget = '\\?\'+$cleanupTarget
            try {
                Remove-Item -LiteralPath $longTarget -Recurse -Force
            } catch {
                # The PowerShell provider can miss deeply nested test fixtures on Windows.
                # The plan has already ruled out reparse points and checked this absolute root.
                if ([System.IO.Directory]::Exists($longTarget)) {
                    foreach ($file in [System.IO.Directory]::EnumerateFiles($longTarget, '*', [System.IO.SearchOption]::AllDirectories)) {
                        [System.IO.File]::SetAttributes($file, [System.IO.FileAttributes]::Normal)
                    }
                    [System.IO.Directory]::Delete($longTarget, $true)
                } else { throw }
            }
        }
        $cleanupReport.deleted_targets += $relativeTarget
    }
    $cleanupReport.state = 'complete'
    $cleanupReport['current'] = $null
} catch {
    $cleanupReport.state = 'failed'
    $cleanupReport['error'] = $_.Exception.Message
    throw
} finally {
    $cleanupReport | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $reportFile -Encoding utf8
}
