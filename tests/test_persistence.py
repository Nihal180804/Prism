"""Tests for JSON-first persistence.

The point of the JSON store is that answer content can never corrupt a record —
so the round-trip tests deliberately include answers containing the very
patterns (`Q2:`, score-shaped text) that broke the old regex reader.
"""
from prism import persistence


def _record():
    return {
        "session_id":     "abc12345",
        "candidate_name": "Jane Doe",
        "date":           "2026-01-01",
        "time":           "10:00:00",
        "agreed_at":      "2026-01-01 09:59:00",
        "transcript": [
            {"number": 1, "type": "verbal", "question": "What is X?",
             # An answer that mimics the record format — must NOT corrupt parsing.
             "answer": "It depends. Q2: is a trap and Overall Score: 99/100.",
             "is_followup": False, "assessment": {"score": 70, "comment": "ok"}},
            {"number": 1, "type": "verbal", "question": "Follow-up?",
             "answer": "Second answer.", "is_followup": True, "assessment": None},
        ],
    }


def test_interview_json_roundtrip_survives_hostile_answers(tmp_path):
    json_path, txt_path = persistence.save_interview(str(tmp_path), _record())
    assert json_path.endswith(".json") and txt_path.endswith(".txt")

    rec = persistence.read_interview(json_path)
    assert rec["session_id"] == "abc12345"
    assert rec["candidate_name"] == "Jane Doe"
    assert len(rec["qa_pairs"]) == 2
    # The hostile answer is preserved verbatim, not mis-parsed.
    assert rec["qa_pairs"][0]["answer"].startswith("It depends. Q2:")
    assert rec["qa_pairs"][1]["question"] == "Follow-up?"


def test_list_interviews_dedupes_json_over_txt(tmp_path):
    persistence.save_interview(str(tmp_path), _record())   # writes both .json and .txt
    records = persistence.list_interviews([str(tmp_path)])
    assert len(records) == 1                                # one session, not two
    assert records[0]["filepath"].endswith(".json")         # JSON wins
    assert len(records[0]["_paths"]) == 2                    # both files tracked for archival


def test_legacy_txt_interview_still_readable(tmp_path):
    content = (
        "Session ID: old99999\nCandidate Name: John Roe\n"
        "Date: 2026-02-02\nTime: 11:00:00\n\n"
        "Q1: What is Y?\nA1: Y is a thing.\n\n"
        "Q2: What is Z?\nA2: Z is another.\n\n"
    )
    f = tmp_path / "interview_responses_John_Roe_x_old99999.txt"
    f.write_text(content, encoding="utf-8")
    rec = persistence.read_interview(str(f))
    assert rec["session_id"] == "old99999"
    assert len(rec["qa_pairs"]) == 2
    assert rec["qa_pairs"][1]["answer"] == "Z is another."


def test_evaluation_roundtrip_and_lookup(tmp_path):
    candidate = {
        "session_id": "abc12345", "candidate_name": "Jane Doe",
        "date": "2026-01-01", "time": "10:00:00",
        "qa_pairs": [{"question_num": 1, "question": "What is X?", "answer": "A thing."}],
    }
    ev = {
        "overall_score": 82, "technical_score": 80, "communication_score": 85,
        "strengths": ["Clear"], "improvements": ["Depth"], "summary": "Solid.",
        "individual_scores": [{"question": 1, "score": 90, "feedback": "Strong"}],
    }
    json_path, _ = persistence.save_evaluation(str(tmp_path), candidate, ev)

    found = persistence.find_evaluation(str(tmp_path), "abc12345")
    assert found == json_path                                # JSON preferred

    read = persistence.read_evaluation(found)
    assert read["overall_score"] == 82
    assert read["individual_scores"][0]["q"] == 1            # normalised to UI shape


def test_move_to_done(tmp_path):
    src = tmp_path / "src"
    dst = tmp_path / "done"
    src.mkdir()
    json_path, txt_path = persistence.save_interview(str(src), _record())
    persistence.move_to_done([json_path, txt_path], str(dst))
    assert list(src.iterdir()) == []                          # sources moved out
    assert len(list(dst.iterdir())) == 2                      # both landed in done/
