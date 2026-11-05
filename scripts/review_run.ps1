param([Parameter(Mandatory)][string]$Dir, [string]$Packet = "$Dir\packet-01.txt", [string]$JulesRepo = "tomiya7688/Image_drawer")
# 外部レビュー(Luna/Continue/Qwen/Antigravity/Grok + Jules)を並列に起動し、結果の状態を一覧する。
# レビューのみ・編集禁止の資料(packet)を-PublicPacketConfirmedで送る。再試行・課金・モデル変更はしない。
$local = 'E:\projects\.review-tools\Invoke-LocalReview.ps1'
$free = 'E:\projects\.review-tools\Invoke-FreeCliReview.ps1'
$jobs = @()
$jobs += Start-Job -Name Luna { & $using:local -Reviewer Luna -InputFile $using:Packet -OutputFile "$using:Dir\luna.json" -PublicPacketConfirmed 2>&1 | Out-String }
foreach ($r in 'Continue', 'Qwen', 'Antigravity', 'Grok') {
    $jobs += Start-Job -Name $r { & $using:free -Reviewer $using:r -InputFile $using:Packet -OutputFile "$using:Dir\$(($using:r).ToLower()).json" -PublicPacketConfirmed 2>&1 | Out-String }
}
$jobs += Start-Job -Name Ori { & 'E:\projects\.review-tools\Invoke-ExtraFreeReview.ps1' -Reviewer Ori -InputFile $using:Packet -OutputFile "$using:Dir\ori.json" -PublicPacketConfirmed 2>&1 | Out-String }
$jules = Get-Content -LiteralPath $Packet -Raw -Encoding UTF8 | jules new --repo $JulesRepo 2>&1 | Out-String
$id = if ($jules -match 'ID:\s*(\d+)') { $Matches[1] } else { '' }
$null = Wait-Job $jobs -Timeout 190
foreach ($j in $jobs) { if ($j.State -ne 'Completed') { Stop-Job $j } }
$rows = foreach ($n in 'luna', 'continue', 'qwen', 'antigravity', 'grok', 'ori') {
    $f = "$Dir\$n.json"
    if (Test-Path -LiteralPath $f) { $o = Get-Content -LiteralPath $f -Raw | ConvertFrom-Json; "{0}: {1} ({2})" -f $n, $o.status, $o.reason } else { "${n}: ファイルなし" }
}
$rows + "jules: session $id 起動(CLIでは本文を取得できない場合がある)" | Set-Content -LiteralPath "$Dir\summary.txt" -Encoding UTF8
Get-Content -LiteralPath "$Dir\summary.txt"
