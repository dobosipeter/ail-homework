from video_analyst.segmentation import Chapter

def test_chapter_time_validation_swap():
    """Test that end_time < start_time is auto-corrected by swapping."""
    # Intentional error: start=100, end=50
    chapter = Chapter(
        title="Test",
        summary="Test summary",
        start_time=100.0,
        end_time=50.0,
        topic_keywords=["test"]
    )

    # Assert it swapped them
    assert chapter.start_time == 50.0
    assert chapter.end_time == 100.0

def test_chapter_time_validation_equal():
    """Test that end_time == start_time is fixed."""
    chapter = Chapter(
        title="Test",
        summary="Summary",
        start_time=10.0,
        end_time=10.0,
        topic_keywords=["test"]
    )
    # Should add 1 second
    assert chapter.end_time == 11.0
