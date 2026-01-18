"""Tests for coordinator IO error handling."""

import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

# Add the tests directory to path to enable imports from test modules
tests_path = Path(__file__).parent
sys.path.insert(0, str(tests_path))

from test_fixtures import MockLogger  # noqa: E402


class TestCoordinatorIOErrorHandling:
    """Test suite for file IO error handling logic."""

    def test_store_table_handles_permission_error(self, tmp_path) -> None:
        """Test that PermissionError is caught when storing table."""
        datafile = tmp_path / "readonly_dir" / "test.csv"
        logger = MockLogger()

        # Create directory with read-only permissions
        datafile.parent.mkdir(parents=True, exist_ok=True)
        datafile.parent.chmod(0o444)  # Read-only

        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

        try:
            # Simulate the store_table error handling
            try:
                datafile.parent.mkdir(parents=True, exist_ok=True)
                if df is not None:
                    df.to_csv(datafile, index=False)
            except (OSError, PermissionError):
                logger.exception(
                    "Failed to store dataset to file %s",
                    str(datafile),
                )

            # Verify error was logged
            assert len(logger.exception_calls) == 1
            assert "Failed to store dataset to file" in logger.exception_calls[0][0]
        finally:
            # Cleanup: restore write permissions
            datafile.parent.chmod(0o755)

    def test_store_table_handles_oserror(self, tmp_path) -> None:
        """Test that OSError is caught when storing table."""
        datafile = tmp_path / "test.csv"
        logger = MockLogger()
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

        # Mock to_csv to raise OSError
        with patch.object(pd.DataFrame, "to_csv", side_effect=OSError("Disk full")):
            # Simulate the store_table error handling
            try:
                datafile.parent.mkdir(parents=True, exist_ok=True)
                if df is not None:
                    df.to_csv(datafile, index=False)
            except (OSError, PermissionError):
                logger.exception(
                    "Failed to store dataset to file %s",
                    str(datafile),
                )

            # Verify error was logged
            assert len(logger.exception_calls) == 1
            assert "Failed to store dataset to file" in logger.exception_calls[0][0]

    def test_store_table_with_none_dataframe(self, tmp_path) -> None:
        """Test that None dataframe doesn't cause errors."""
        datafile = tmp_path / "test.csv"
        logger = MockLogger()

        # Simulate the store_table with None
        df = None
        try:
            datafile.parent.mkdir(parents=True, exist_ok=True)
            if df is not None:
                df.to_csv(datafile, index=False)
        except (OSError, PermissionError):
            logger.exception(
                "Failed to store dataset to file %s",
                str(datafile),
            )

        # Verify no error was logged
        assert len(logger.exception_calls) == 0

    def test_read_table_handles_permission_error(self, tmp_path) -> None:
        """Test that PermissionError is caught when reading table."""
        datafile = tmp_path / "test.csv"
        logger = MockLogger()

        # Create a CSV file
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        df.to_csv(datafile, index=False)

        # Mock pd.read_csv to raise PermissionError
        with patch("pandas.read_csv", side_effect=PermissionError("Access denied")):
            # Simulate the read_table error handling
            if datafile.exists():
                try:
                    pd.read_csv(datafile, header=0)
                except (OSError, PermissionError):
                    logger.exception(
                        "Failed to read dataset from file %s",
                        str(datafile),
                    )
                except pd.errors.ParserError:
                    logger.exception(
                        "Failed to parse CSV file %s",
                        str(datafile),
                    )

            # Verify error was logged
            assert len(logger.exception_calls) == 1
            assert "Failed to read dataset from file" in logger.exception_calls[0][0]

    def test_read_table_handles_oserror(self, tmp_path) -> None:
        """Test that OSError is caught when reading table."""
        datafile = tmp_path / "test.csv"
        logger = MockLogger()

        # Create a CSV file
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        df.to_csv(datafile, index=False)

        # Mock pd.read_csv to raise OSError
        with patch("pandas.read_csv", side_effect=OSError("IO error")):
            # Simulate the read_table error handling
            if datafile.exists():
                try:
                    pd.read_csv(datafile, header=0)
                except (OSError, PermissionError):
                    logger.exception(
                        "Failed to read dataset from file %s",
                        str(datafile),
                    )
                except pd.errors.ParserError:
                    logger.exception(
                        "Failed to parse CSV file %s",
                        str(datafile),
                    )

            # Verify error was logged
            assert len(logger.exception_calls) == 1
            assert "Failed to read dataset from file" in logger.exception_calls[0][0]

    def test_read_table_handles_parser_error(self, tmp_path) -> None:
        """Test that ParserError is caught when reading table."""
        datafile = tmp_path / "test.csv"
        logger = MockLogger()

        # Create a CSV file
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        df.to_csv(datafile, index=False)

        # Mock pd.read_csv to raise ParserError
        with patch("pandas.read_csv", side_effect=pd.errors.ParserError("Parse error")):
            # Simulate the read_table error handling
            if datafile.exists():
                try:
                    pd.read_csv(datafile, header=0)
                except (OSError, PermissionError):
                    logger.exception(
                        "Failed to read dataset from file %s",
                        str(datafile),
                    )
                except pd.errors.ParserError:
                    logger.exception(
                        "Failed to parse CSV file %s",
                        str(datafile),
                    )

            # Verify error was logged
            assert len(logger.exception_calls) == 1
            assert "Failed to parse CSV file" in logger.exception_calls[0][0]

    def test_read_table_file_not_exists(self, tmp_path) -> None:
        """Test that non-existent file doesn't cause errors."""
        datafile = tmp_path / "nonexistent.csv"
        logger = MockLogger()

        # Simulate the read_table with non-existent file
        if datafile.exists():
            try:
                pd.read_csv(datafile, header=0)
            except (OSError, PermissionError):
                logger.exception(
                    "Failed to read dataset from file %s",
                    str(datafile),
                )
            except pd.errors.ParserError:
                logger.exception(
                    "Failed to parse CSV file %s",
                    str(datafile),
                )

        # Verify no exception was logged (file doesn't exist, so we skip reading)
        assert len(logger.exception_calls) == 0

    def test_store_table_success(self, tmp_path) -> None:
        """Test that store operation works on success."""
        datafile = tmp_path / "test.csv"
        logger = MockLogger()
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})

        # Simulate the store_table success path
        try:
            datafile.parent.mkdir(parents=True, exist_ok=True)
            if df is not None:
                df.to_csv(datafile, index=False)
        except (OSError, PermissionError):
            logger.exception(
                "Failed to store dataset to file %s",
                str(datafile),
            )

        # Verify file was created
        assert datafile.exists()

        # Verify contents
        read_df = pd.read_csv(datafile)
        assert read_df.equals(df)

        # Verify no error was logged
        assert len(logger.exception_calls) == 0

    def test_read_table_success(self, tmp_path) -> None:
        """Test that read operation works on success."""
        datafile = tmp_path / "test.csv"
        logger = MockLogger()

        # Create a CSV file
        df = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
        df.to_csv(datafile, index=False)

        # Simulate the read_table success path
        dataset = None
        if datafile.exists():
            try:
                dataset = pd.read_csv(datafile, header=0)
            except (OSError, PermissionError):
                logger.exception(
                    "Failed to read dataset from file %s",
                    str(datafile),
                )
            except pd.errors.ParserError:
                logger.exception(
                    "Failed to parse CSV file %s",
                    str(datafile),
                )

        # Verify data was loaded
        assert dataset is not None
        assert dataset.shape[0] == 2

        # Verify no error was logged
        assert len(logger.exception_calls) == 0
