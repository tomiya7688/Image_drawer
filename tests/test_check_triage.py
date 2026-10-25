"""優先度の読み取りと、PR・Issueの作業順を確認する。"""

import pytest

from scripts import check_local, check_triage


# {
#   責務: [
#     test_priority_sources: ラベル・タイトル・起票フォームのいずれからも優先度を読み取る
#   ]
#   引数: [
#     issue: Issue一覧の1件, expected: 期待する優先度
#   ]
# }
@pytest.mark.parametrize(
    ("issue", "expected"),
    [
        ({"number": 1, "title": "x", "labels": [{"name": "priority: P0"}]}, "P0"),
        ({"number": 2, "title": "[P1] x", "labels": []}, "P1"),
        ({"number": 3, "title": "x", "labels": [], "body": "### 優先度\n\nP2\n\n### 背景"}, "P2"),
        ({"number": 4, "title": "x", "labels": [{"name": "bug"}], "body": None}, None),
        ({"number": 5, "title": "[P9] x", "labels": []}, None),
    ],
)
def test_priority_sources(issue, expected):
    assert check_triage.priority_of(issue) == expected


# {
#   責務: [
#     test_order_issues: 優先度の高い順、同じ優先度は番号順、未設定は末尾に並べる
#   ]
# }
def test_order_issues():
    issues = [
        {"number": 9, "title": "x", "labels": []},
        {"number": 7, "title": "[P2] x", "labels": []},
        {"number": 5, "title": "[P0] x", "labels": []},
        {"number": 3, "title": "[P2] x", "labels": []},
    ]
    ordered = [(priority, issue["number"]) for priority, issue in check_triage.order_issues(issues)]
    assert ordered == [("P0", 5), ("P2", 3), ("P2", 7), (None, 9)]


# {
#   責務: [
#     test_exit_codes: ghが使えない場合、PRや未設定がある場合、問題がない場合の終了値を分ける
#   ]
#   引数: [
#     pull_requests: 取得したPR, issues: 取得したIssue, expected: 期待する終了値
#   ]
# }
@pytest.mark.parametrize(
    ("pull_requests", "issues", "expected"),
    [
        ([], [{"number": 1, "title": "[P1] x", "labels": []}], check_triage.EXIT_OK),
        ([{"number": 2, "title": "pr"}], [], check_triage.EXIT_ATTENTION),
        ([], [{"number": 3, "title": "x", "labels": []}], check_triage.EXIT_ATTENTION),
    ],
)
def test_exit_codes(monkeypatch, capsys, pull_requests, issues, expected):
    responses = iter([pull_requests, issues])
    monkeypatch.setattr(check_triage, "run_gh_json", lambda arguments: next(responses))
    assert check_triage.main() == expected
    assert "オープンPR" in capsys.readouterr().out


# {
#   責務: [
#     test_unavailable_gh: ghが使えない場合はSKIP相当の終了値を返し、成功扱いにしない
#   ]
# }
def test_unavailable_gh(monkeypatch, capsys):
    def fail(arguments):
        raise RuntimeError("ghなし")

    monkeypatch.setattr(check_triage, "run_gh_json", fail)
    assert check_triage.main() == check_triage.EXIT_UNAVAILABLE
    assert "ghなし" in capsys.readouterr().out


# {
#   責務: [
#     test_local_check_maps_warning: check_localがtriageの終了値をWARN・SKIPに対応させ、失敗終了にしない
#   ]
# }
@pytest.mark.parametrize(("code", "status"), [(0, "PASS"), (3, "WARN"), (2, "SKIP"), (1, "FAIL")])
def test_local_check_maps_warning(tmp_path, code, status):
    checker = check_local.LocalCheck(tmp_path)
    checker.run(
        "triage",
        [check_local.sys.executable, "-c", f"raise SystemExit({code})"],
        statuses={2: "SKIP", 3: "WARN"},
    )
    assert checker.results[0][1] == status
    assert checker.summarize(0.0) == (1 if status == "FAIL" else 0)


# {
#   責務: [
#     test_gh_output_is_decoded_as_utf8: Windowsの既定文字コードに依存せず、日本語を含むghの出力を読む
#   ]
#   処理: [
#     subprocess.runへ渡される引数を捕捉し、UTF-8が明示されていることを確認する
#   ]
# }
def test_gh_output_is_decoded_as_utf8(monkeypatch):
    captured = {}

    def fake_run(command, **options):
        captured.update(options)
        return check_triage.subprocess.CompletedProcess(command, 0, stdout="[]", stderr="")

    monkeypatch.setattr(check_triage.subprocess, "run", fake_run)
    assert check_triage.run_gh_json(["issue", "list"]) == []
    assert captured["encoding"] == "utf-8"
