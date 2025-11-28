from video_analyst.rag import _reconstruct_chapter_text

def test_reconstruct_chapter_text():
    """Test that we correctly grab words within the time window."""
    mock_transcript = [
        {
            "words": [
                {"word": "Hello", "start": 0.5, "end": 0.9},
                {"word": "world", "start": 1.5, "end": 1.9}, # Inside
                {"word": "this", "start": 2.5, "end": 2.9},  # Inside
                {"word": "is", "start": 5.0, "end": 5.5},    # Outside
            ]
        }
    ]

    # Window: 1.0s to 3.0s
    text = _reconstruct_chapter_text(1.0, 3.0, mock_transcript)

    assert text == "world this"
    words = text.split()
    assert "Hello" not in words
    assert "is" not in words
