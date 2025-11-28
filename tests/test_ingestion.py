from unittest.mock import patch, MagicMock
from pathlib import Path
from video_analyst.ingestion import extract_audio, split_audio

@patch("os.system")
@patch("pathlib.Path.exists")
def test_extract_audio_success(mock_exists, mock_system):
    """Test audio extraction command generation when file doesn't exist."""
    # Setup: File does NOT exist
    mock_exists.return_value = False
    mock_system.return_value = 0 # Success return code

    # Action
    output = extract_audio("test_video.mp4", output_dir=Path("/tmp"))

    # Assert
    assert str(output) == "/tmp/test_video.flac"
    mock_system.assert_called_once()

    # Check that the command contains input and output paths
    cmd = mock_system.call_args[0][0]
    assert '-i "test_video.mp4"' in cmd
    assert '"/tmp/test_video.flac"' in cmd

@patch("os.system")
@patch("pathlib.Path.exists")
def test_extract_audio_cached(mock_exists, mock_system):
    """Test that we skip extraction if the file already exists."""
    # Setup: File DOES exist
    mock_exists.return_value = True

    output = extract_audio("test_video.mp4", output_dir=Path("/tmp"))

    assert str(output) == "/tmp/test_video.flac"
    # Should NOT call ffmpeg
    mock_system.assert_not_called()

@patch("video_analyst.ingestion.AudioSegment")
@patch("pathlib.Path.exists") 
def test_split_audio_logic(mock_exists, mock_audio_segment):
    """Test that audio is correctly split into chunks."""
    # Setup: Path.exists returns False so we trigger splitting
    mock_exists.return_value = False

    # Mock AudioSegment to behave like a 25-minute file
    mock_audio = MagicMock()
    mock_audio.__len__.return_value = 25 * 60 * 1000 # 25 minutes in ms
    mock_audio.__getitem__.return_value = MagicMock() # Allow slicing [start:end]

    mock_audio_segment.from_file.return_value = mock_audio

    # Action: Split into 10-minute chunks
    chunks = split_audio("test.flac", chunk_length_ms=10 * 60 * 1000)

    # Assert
    # 25 mins / 10 mins = 3 chunks (10, 10, 5)
    assert len(chunks) == 3
    assert "test_part0.flac" in chunks[0]
    assert "test_part1.flac" in chunks[1]
    assert "test_part2.flac" in chunks[2]
